use super::u256_simd::*;
use crate::bitset::*;
use std::collections::HashSet;
use std::cmp::*;
use rand::Rng;
use rand_pcg::Pcg64Mcg;

pub struct Params {
    pub n_nodes: usize,
    pub n_accounts: usize,
    pub byz_frac: usize,  // Fraction of nodes that are Byzantine.
    pub byz_tol: usize,   // Fraction of Byzantine nodes that can be tolerated.
    pub shard_size: usize,
    pub reps: usize,
    pub iter: usize,
    pub seed: u128,
}

impl Params {
    pub fn with_shard_size(&self, shard_size: usize) -> Self {
        Params {
            shard_size,
            ..*self
        }
    }
    pub fn with_seed(&self, seed: u128) -> Self {
        Params {
            seed,
            ..*self
        }
    }

    pub fn with_n_accounts(&self, n_accounts: usize) -> Self {
        Params {
            n_accounts,
            ..*self
        }
    }
}

// This function is needed so that we can compare the two implementations.
// That way, they will both use the RNG in exactly the same way and generate the same
// nodes, accounts, and, more importantly, Byzantine nodes.
pub fn random_hashset(nodes: &Vec<U256>, m: usize, rng: &mut Pcg64Mcg) -> HashSet<U256> {
    let n = nodes.len();
    assert!(n >= m);
    let mut subset = HashSet::with_capacity(m);
    let mut count = 0;
    for i in 0..n {
        let x: f64 = rng.gen();
        if x < (m - count) as f64 / (n - i) as f64 {
            subset.insert(nodes[i]);
            count += 1;
            if count == m {
                return subset
            }
        }
    }
    panic!("only made {} < {}", count, m)
}

// Uses from_sorted(Vec) bulk create.
pub fn random_bitset(n: usize, m: usize, rng: &mut Pcg64Mcg) -> BitSet {
    assert!(n >= m);
    let mut indices = Vec::with_capacity(m);
    let mut count = 0;
    for i in 0..n {
        let x: f64 = rng.gen();
        if x < (m - count) as f64 / (n - i) as f64 {
            indices.push(i);
            count += 1;
            if count == m {
                indices.sort();
                return BitSet::from_sorted(&indices);
            }
        }
    }
    panic!("only made {} < {}", count, m)
}

// Adds one index at a time to the bitset.
pub fn random_bitset_serial(n: usize, m: usize, rng: &mut Pcg64Mcg) -> BitSet {
    assert!(n >= m);
    let mut subset = BitSet::new();
    let mut count = 0;
    for i in 0..n {
        let x: f64 = rng.gen();
        if x < (m - count) as f64 / (n - i) as f64 {
            subset.insert(i);
            count += 1;
            if count == m {
                return subset
            }
        }
    }
    panic!("only made {} < {}", count, m)
}

pub fn tolerance(n: usize, frac: usize) -> usize {
    (n-1) / frac
}

pub fn shard_hashset(id: &U256, mut nodes: Vec<U256>, k: usize) -> HashSet<U256> {
    nodes.sort_by(|a, b| id.compare(a, b));
    nodes.into_iter().take(k).collect()
}


pub struct Sharder {
    indices: Vec<usize>,
    nodes: Vec<U256>,
}

impl Sharder {
    pub fn new(nodes: Vec<U256>) -> Self {
        let indices: Vec<usize> = Vec::with_capacity(nodes.len());
        Sharder { indices, nodes, }
    }

    pub fn make_shard_indices(&mut self, id: U256, r: usize) -> Vec<usize> {
        self.indices.truncate(0);
        for i in 0..r {
            self.indices.push(i);
        }
        assert_eq!(self.indices.len(), r);
        self.indices.sort_by(|i, j| id.compare(&self.nodes[*i], &self.nodes[*j]));
        let mut maxdist = &id ^ &self.nodes[self.indices[r-1]];
        for i in r..self.nodes.len() {
            let targetdist = &id ^ &self.nodes[i];
            if targetdist.cmp(&maxdist) == Ordering::Greater {
                continue
            }
            let index = index_of(&self.nodes, &self.indices, &id, &targetdist);
            if index == r-1 {
                self.indices[index] = i;
                maxdist = targetdist;
            } else {
                self.indices.insert(index, i);
                self.indices.truncate(r);
            }
            assert_eq!(self.indices.len(), r);
        }
        self.indices.clone()
    }

    pub fn make_shard(&mut self, id: U256, r: usize) -> BitSet {
        self.indices.truncate(0);
        for i in 0..r {
            self.indices.push(i);
        }
        assert_eq!(self.indices.len(), r);
        // Note: We only care about the sort order because it help to keep track of the r closest.
        self.indices.sort_by(|i, j| id.compare(&self.nodes[*i], &self.nodes[*j]));
        let mut maxdist = &id ^ &self.nodes[self.indices[r-1]];
        for i in r..self.nodes.len() {
            let targetdist = &id ^ &self.nodes[i];
            if targetdist.cmp(&maxdist) == Ordering::Greater {
                continue
            }
            let index = index_of(&self.nodes, &self.indices, &id, &targetdist);
            if index == r-1 {
                self.indices[index] = i;
                maxdist = targetdist;
            } else {
                self.indices.insert(index, i);
                self.indices.truncate(r);
            }
        }
        assert_eq!(self.indices.len(), r);
        let out = if r > 200 && self.nodes.len() > 4000 {
            self.indices.sort();
            BitSet::from_sorted(&self.indices)
        } else {
            let mut out = BitSet::new();
            for i in &self.indices {
                out.insert(*i);
            }
            out
        };
        assert_eq!(out.len(), r);
        out
    }

}

// NOTE: Using secondary buckets is only a tiny bit faster (like 1 %).
pub struct Sharder3 {
    indices: Vec<usize>,
    nodes: Vec<U256>,
    buckets: Vec<Vec<usize>>,
    prefix_bits: usize,
    secondary_buckets: Vec<Vec<usize>>,
    secondary_bits: usize,
}

impl Sharder3 {
    pub fn new(nodes: Vec<U256>, r: usize) -> Self {
        let indices: Vec<usize> = Vec::with_capacity(nodes.len());
        // We want to look at a number of bits such that the number of prefixes (buckets) is such
        // that nodes/buckets > 2r.
        // That way, we shouldn't have a problem with buckets that are too small.
        let prefix_bits = min(7, max(1, nodes.len()/r - 1).ilog2()) as usize;
        let secondary_bits = if prefix_bits == 0 {
            0
        } else {
            prefix_bits - 1
        };
        let prefix_count = 2_usize.pow(prefix_bits as u32);
        let secondary_count = 2_usize.pow(secondary_bits as u32);
        let mut buckets = Vec::with_capacity(prefix_count);
        let mut secondary_buckets = if prefix_bits > 0 {
            Vec::with_capacity(secondary_count)
        } else {
            Vec::new()
        };
        for _ in 0..prefix_count {
            buckets.push(Vec::with_capacity(nodes.len() / prefix_count + 1));
            if prefix_bits > 0 {
                secondary_buckets.push(Vec::with_capacity(nodes.len() / secondary_count + 1));
            }
        }
        for (i, id) in nodes.iter().enumerate() {
            let prefix = id.prefix(prefix_bits);
            buckets[prefix].push(i);
            if prefix_bits > 0 {
                let prefix = id.prefix(secondary_bits);
                secondary_buckets[prefix].push(i);
            }
        }
        Sharder3 { indices, nodes, prefix_bits, buckets, secondary_buckets, secondary_bits }
    }

    // A mix of fast and clever is best
    pub fn make_shard(&mut self, id: U256, r: usize) -> BitSet {
        if self.nodes.len() >= 4000 { // and something like  || r > 200 ??
            self.make_shard_faster(id, r)
        } else {
            self.make_shard_clever(id, r)
        }
    }

    pub fn make_shard_clever(&mut self, id: U256, r: usize) -> BitSet {
        let bucket = {
            let prefix = id.prefix(self.prefix_bits);
            let bucket = &self.buckets[prefix];
            if bucket.len() >= r {
                bucket
            } else {
                let prefix = id.prefix(self.secondary_bits);
                let bucket = &self.secondary_buckets[prefix];
                assert!(bucket.len() >= r);
                bucket
            }
        };
        unsafe {
            self.indices.set_len(r);
        }
        self.indices[0..r].copy_from_slice(&bucket[0..r]);
        assert_eq!(self.indices.len(), r);
        // Note: We only care about the sort order because it helps to keep track of the r closest.
        self.indices.sort_by(|i, j| id.compare(&self.nodes[*i], &self.nodes[*j]));
        let mut maxdist = &id ^ &self.nodes[self.indices[r-1]];
        for i in r..bucket.len() {
            let i = bucket[i];
            let targetdist = &id ^ &self.nodes[i];
            if targetdist.cmp(&maxdist) == Ordering::Greater {
                continue
            }
            let index = index_of(&self.nodes, &self.indices, &id, &targetdist);
            if index == r-1 {
                self.indices[index] = i;
                maxdist = targetdist;
            } else {
                self.indices.insert(index, i);
                self.indices.truncate(r);
            }
        }
        assert_eq!(self.indices.len(), r);
        let out = if r > 200 && self.nodes.len() > 4000 {
            self.indices.sort();
            BitSet::from_sorted(&self.indices)
        } else {
            let mut out = BitSet::new();
            for i in &self.indices {
                out.insert(*i);
            }
            out
        };
        assert_eq!(out.len(), r);
        out
    }

    pub fn make_shard_faster(&mut self, id: U256, r: usize) -> BitSet {
        let bucket = {
            let prefix = id.prefix(self.prefix_bits);
            let bucket = &mut self.buckets[prefix];
            if bucket.len() > r {
                bucket
            } else {
                let prefix = id.prefix(self.secondary_bits);
                let bucket = &mut self.secondary_buckets[prefix];
                assert!(bucket.len() >= r);
                bucket
            }
        };
        // Sort all nodes in the bucket
        let (smaller, _, _) = bucket.select_nth_unstable_by(r, |i, j| id.compare(&self.nodes[*i], &self.nodes[*j]));
        // from_sorted is faster than insert
        smaller.sort();
        BitSet::from_sorted_slice(smaller)
    }

}

fn index_of(nodes: &Vec<U256>, indices: &Vec<usize>, id: &U256, x_dist: &U256) -> usize {
    index_of_bin(nodes, indices, id, x_dist)
}

#[allow(dead_code)]
fn index_of_lin(nodes: &Vec<U256>, indices: &Vec<usize>, id: &U256, x_dist: &U256) -> usize {
    for j in 0..indices.len() {
        let dist = id ^ &nodes[indices[j]];
        if x_dist.cmp(&dist) == Ordering::Less {
            return j;
        }
    }
    indices.len()
}

// This is much faster. So Rust apparently wasn't able to automatically convert
// to binary search for me.
fn index_of_bin(nodes: &Vec<U256>, indices: &Vec<usize>, id: &U256, x_dist: &U256) -> usize {
    let mut low = 0;
    let mut high = indices.len();
    let mut mid = low + (high - low)/2;
    while low < high {
        let dist = id ^ &nodes[indices[mid]];
        match x_dist.cmp(&dist) {
            Ordering::Less => {
                high = mid;
            }
            Ordering::Greater => {
                low = mid+1;
            }
            Ordering::Equal => {
                return mid;
            }
        }
        mid = low + (high - low)/2;
    }
    mid
}


#[cfg(test)]
mod test {
    use super::*;

    // Make sure random_hashset and random_bitset select the same elements.
    #[test]
    fn random_subset() {
        for seed in vec![123456789, 234567891, 345678912, 456789123] {
            let n = 1000;
            let mut rng = Pcg64Mcg::new(seed);
            let nodes = generate_ids(n, &mut rng);
            let hashset = random_hashset(&nodes, 100, &mut rng);
            let mut rng = Pcg64Mcg::new(seed);
            let nodes = generate_ids(n, &mut rng);
            let bitset = random_bitset(n, 100, &mut rng);
            for i in 0..n {
                assert_eq!(hashset.contains(&nodes[i]), bitset.contains(i));
            }
        }
    }

    #[test]
    fn compare_sharding() {
        for seed in vec![123456789, 234567891, 345678912, 456789123] {
            let n = 1000;
            let m = 2000;
            let r = 51;
            let mut rng = Pcg64Mcg::new(seed);
            let nodes = generate_ids(n, &mut rng);
            let accounts = generate_ids(m, &mut rng);
            let mut shards1: Vec<HashSet<U256>> = Vec::with_capacity(m);
            let mut shards2: Vec<BitSet> = Vec::with_capacity(m);
            let mut shards3: Vec<BitSet> = Vec::with_capacity(m);
            let mut sharder1 = Sharder::new(nodes.clone());
            let mut sharder3 = Sharder3::new(nodes.clone(), r);
            for account in accounts {
                let shard = shard_hashset(&account, nodes.clone(), r);
                assert_eq!(shard.len(), r);
                shards1.push(shard);
                let shard = sharder1.make_shard(account, r);
                assert_eq!(shard.len(), r);
                shards2.push(shard);
                let shard = sharder3.make_shard(account, r);
                assert_eq!(shard.len(), r);
                shards3.push(shard);
            }
            assert_eq!(shards1.len(), m);
            assert_eq!(shards2.len(), m);
            assert_eq!(shards3.len(), m);
            for i in 0..m {
                let shard1 = &shards1[i];
                let shard2 = &shards2[i];
                let shard3 = &shards3[i];
                for j in 0..n {
                    assert_eq!(shard1.contains(&nodes[j]), shard2.contains(j));
                    assert_eq!(shard1.contains(&nodes[j]), shard3.contains(j));
                }
            }
        }
    }

    #[test]
    fn random_bitset_serial_and_bulk_equivalence() {
        for seed in vec![123456789, 234567891, 345678912, 456789123] {
            let n = 8000;
            let b = 500;
            for _ in 0..9999 {
                let mut rng1 = Pcg64Mcg::new(seed);
                let mut rng2 = Pcg64Mcg::new(seed);
                let bitset1 = random_bitset(n, b, &mut rng1);
                assert_eq!(bitset1.len(), b);
                let bitset2 = random_bitset_serial(n, b, &mut rng2);
                assert_eq!(bitset2.len(), b);
                for i in 0..n {
                    assert_eq!(bitset1.contains(i), bitset2.contains(i));
                }
            }
        }
    }

}
