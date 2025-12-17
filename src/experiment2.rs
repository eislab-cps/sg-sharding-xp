use crate::u256_simd::*;
use crate::bitset::*;
use crate::util::*;
use std::collections::HashSet;
use std::time::Instant;
use rand::Rng;
use rand_pcg::Pcg64Mcg;

// ============================================================================
// Real (BitSet-based) implementation
// ----------------------------------------------------------------------------
// ============================================================================

pub fn experiment2(params: &Params, mut rs: Vec<usize>) -> Vec<Vec<usize>> {
    rs.reverse();
    let n_nodes = params.n_nodes;
    let n_accounts = params.n_accounts;
    let max_r = rs[0];
    let mut results: Vec<Vec<usize>> = (0..rs.len()).map(|_| vec![0].repeat(params.reps)).collect();
    let mut shard_cache: Vec<Vec<usize>> = Vec::with_capacity(n_accounts);
    let mut rng = Pcg64Mcg::new(params.seed);
    for i in 0..params.reps {
        let mut rng = Pcg64Mcg::new(rng.gen());
        shard_cache.truncate(0);
        let nodes = generate_ids(n_nodes, &mut rng);
        let accounts = generate_ids(n_accounts, &mut rng);
        let mut sharder = Sharder::new(nodes);
        for account in accounts {
            let shard = sharder.make_shard_indices(account, max_r);
            shard_cache.push(shard);
        }
        for (j,r) in rs.iter().enumerate() {
            let params = params.with_shard_size(*r);
            let shards = make_shards(&shard_cache, *r);
            results[j][i] += count_compromised_iter(&shards, &params, rng.gen())
        }
    }
    results.reverse();
    results
}

// There is a very slight optimization possible, but it may not be significant enough
// to be worth the effort: instead of rebuilding the shard (bitset) from scratch for each
// r, it might be slightly more efficient to build all of them at once. When done building
// a shard for one r, clone it and continue adding to the next one, etc.
/// Takes the largest shards (in terms of node indices) and truncates them to `r` to form
/// shards in terms of bit sets.
fn make_shards(cache: &Vec<Vec<usize>>, r: usize) -> Vec<BitSet> {
    let mut shards: Vec<BitSet> = Vec::with_capacity(cache.len());
    for shard in cache {
        let mut bitset = BitSet::new();
        for i in 0..r {
            bitset.insert(shard[i]);
        }
        shards.push(bitset);
    }
    shards
}

/// Count **the number of iterations** where there is *at least one* compromised shard.
fn count_compromised_iter(shards: &Vec<BitSet>, params: &Params, seed: u128) -> usize {
    let mut rng = Pcg64Mcg::new(seed);
    let n_nodes = params.n_nodes;
    let n_byz = n_nodes / params.byz_frac;
    let shard_tol = tolerance(shards[0].len(), params.byz_tol);
    let mut count = 0;
    for _ in 0..params.iter {
        let byz: BitSet = random_bitset(n_nodes, n_byz, &mut rng);
        for shard in shards {
            if byz.intersection_size(shard) > shard_tol {
                count += 1;
                break;
            }
        }
    }
    count
}

// --- single ---

/// Check both individual shards and their unions and find the failure probability for a given
/// shard size.
pub fn experiment2_single(params: &Params, rs: Vec<usize>) -> Vec<Vec<usize>> {
    let timer = Instant::now();
    assert_eq!(rs.len(), 1);
    let mut rng = Pcg64Mcg::new(params.seed);
    let r = rs[0];
    let params = params.with_shard_size(r);
    let mut results: Vec<usize> = Vec::with_capacity(params.reps);
    for _ in 0..params.reps {
        results.push(test_r_single(&params, rng.gen()));
        println!("F = 1/{}, f = 1/{}, N = {}, r = {}, fails = {:?}", params.byz_frac, params.byz_tol, params.n_nodes, params.shard_size, results);
    }
    let secs = timer.elapsed().as_secs();
    println!("After {} sec:", secs);
    vec![results]
}

fn test_r_single(params: &Params, seed: u128) -> usize {
    let mut rng = Pcg64Mcg::new(seed);
    let n_nodes = params.n_nodes;
    let n_accounts = params.n_accounts;
    // Generate nodes and shards
    let nodes = generate_ids(n_nodes, &mut rng);
    let mut sharder = Sharder::new(nodes);
    let mut shards: Vec<BitSet> = Vec::with_capacity(n_accounts);
    for _ in 0..n_accounts {
        let account = U256::new(rng.gen());
        let shard = sharder.make_shard(account, params.shard_size);
        shards.push(shard);
    }
    // Now test the shards...
    count_compromised_iter(&shards, &params, rng.gen())
}

// ============================================================================
// Naive reference implementation
// ----------------------------------------------------------------------------
// This is the naive and straightforward implementation that is "obviously correct".
// ============================================================================

pub fn experiment2_naive(params: &Params, mut rs: Vec<usize>) -> Vec<Vec<usize>> {
    let timer = Instant::now();
    rs.reverse();
    let n_nodes = params.n_nodes;
    let n_accounts = params.n_accounts;
    let mut rng = Pcg64Mcg::new(params.seed);
    let mut results: Vec<Vec<usize>> = (0..rs.len()).map(|_| vec![0].repeat(params.reps)).collect();
    for i in 0..params.reps {
        let mut rng = Pcg64Mcg::new(rng.gen());
        // Generate nodes and shards
        let nodes = generate_ids(n_nodes, &mut rng);
        let accounts = generate_ids(n_accounts, &mut rng);
        for (j,r) in rs.iter().enumerate() {
            let params = &params.with_shard_size(*r);
            let mut shards: Vec<HashSet<U256>> = Vec::with_capacity(n_accounts);
            for account in &accounts {
                let shard = shard_hashset(&account, nodes.clone(), params.shard_size);
                shards.push(shard);
            }
            results[j][i] += count_compromised_naive(&shards, &nodes, &params, rng.gen())
        }
    }
    let secs = timer.elapsed().as_secs();
    println!("After {} sec:", secs);
    results.reverse();
    results
}

/// Count **the number of iterations** where there is at least one compromised shard.
fn count_compromised_naive(shards: &Vec<HashSet<U256>>, nodes: &Vec<U256>, params: &Params, seed: u128) -> usize {
    assert_eq!(nodes.len(), params.n_nodes);
    let mut rng = Pcg64Mcg::new(seed);
    let n_byz = nodes.len() / params.byz_frac;
    let tolerance = tolerance(shards[0].len(), params.byz_tol);
    let mut failed = 0;
    for _ in 0..params.iter {
        let byz: HashSet<U256> = random_hashset(nodes, n_byz, &mut rng);
        for shard in shards {
            let intersection: HashSet<&U256> = byz.intersection(shard).collect();
            if intersection.len() > tolerance {
                failed += 1;
                break;
            }
        }
    }
    failed
}

// --- single ---

pub fn experiment2_single_naive(params: &Params, rs: Vec<usize>) -> Vec<Vec<usize>> {
    let timer = Instant::now();
    assert_eq!(rs.len(), 1);
    let mut rng = Pcg64Mcg::new(params.seed);
    let r = rs[0];
    let params = params.with_shard_size(r);
    let mut results: Vec<usize> = Vec::with_capacity(params.reps);
    for _ in 0..params.reps {
        results.push(test_r_single_naive(&params, rng.gen()));
        println!("F = 1/{}, f = 1/{}, N = {}, r = {}, fails = {:?}", params.byz_frac, params.byz_tol, params.n_nodes, params.shard_size, results);
    }
    let secs = timer.elapsed().as_secs();
    println!("After {} sec:", secs);
    vec![results]
}

fn test_r_single_naive(params: &Params, seed: u128) -> usize {
    let mut rng = Pcg64Mcg::new(seed);
    let n_nodes = params.n_nodes;
    let n_accounts = params.n_accounts;
    // Generate nodes and shards
    let nodes = generate_ids(n_nodes, &mut rng);   // OPT: reuse memory
    let mut shards: Vec<HashSet<U256>> = Vec::with_capacity(n_accounts);
    for _ in 0..n_accounts {
        let account = U256::new(rng.gen());
        let shard = shard_hashset(&account, nodes.clone(), params.shard_size);
        shards.push(shard);
    }
    // Now test the shards...
    count_compromised_naive(&shards, &nodes, &params, rng.gen())
}


#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn compare_to_ref_single() {
        let n_nodes = 500;
        let n_accounts = 100;
        let shard_sizes = vec![61];
        for seed in vec![123456789, 234567891, 345678912, 456789123, 567891234] {
            let params = Params {
                n_nodes,
                n_accounts,
                byz_frac: 4,
                byz_tol: 3,
                shard_size: 0,  // ignored
                reps: 10,
                iter: 10,
                seed,
            };
            let counts_bulk = experiment2(&params, shard_sizes.clone());
            let counts_single = experiment2_single(&params, shard_sizes.clone());
            let counts_bulk_naive = experiment2_naive(&params, shard_sizes.clone());
            let counts_single_naive = experiment2_single_naive(&params, shard_sizes.clone());
            assert_eq!(counts_bulk, counts_single);
            assert_eq!(counts_bulk, counts_bulk_naive);
            assert_eq!(counts_bulk, counts_single_naive);
        }
    }

    #[test]
    fn compare_to_ref_bulk() {
        let n_nodes = 500;
        let n_accounts = 100;
        let shard_sizes = vec![31, 51, 71, 81];
        for seed in vec![123456789, 234567891, 345678912, 456789123, 567891234] {
            let params = Params {
                n_nodes,
                n_accounts,
                byz_frac: 4,
                byz_tol: 3,
                shard_size: 0,  // ignored
                reps: 10,
                iter: 10,
                seed,
            };
            let counts_bulk = experiment2(&params, shard_sizes.clone());
            let counts_bulk_naive = experiment2_naive(&params, shard_sizes.clone());
            assert_eq!(counts_bulk, counts_bulk_naive);
        }
    }

    // Failure (compromised shards) is impossible.
    #[test]
    fn sanity_check_no_fail() {
        let shard_sizes = vec![100];
        let params = Params {
            n_nodes: 100,
            n_accounts: 10,
            byz_frac: 4,
            byz_tol: 3,
            shard_size: 0,  // ignored
            reps: 10,
            iter: 100,
            seed: 123456789,
        };
        let count = experiment2(&params, shard_sizes.clone());
        let expected = vec![vec![0].repeat(params.reps)];
        assert_eq!(count, expected);
    }

    // Failure (compromised shards) is guaranteed.
    #[test]
    fn sanity_check_fail() {
        let shard_sizes = vec![11];
        let params = Params {
            n_nodes: 100,
            n_accounts: 1,
            byz_frac: 1,
            byz_tol: 3,
            shard_size: 0,  // ignored
            reps: 2,
            iter: 5,
            seed: 123456789,
        };
        let count = experiment2(&params, shard_sizes.clone());
        let expected = vec![vec![params.iter].repeat(params.reps)];
        assert_eq!(count, expected);
    }

}
