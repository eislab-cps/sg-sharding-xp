use std::cmp::*;

pub type BitSet = FixedBitSet128;


/// A FixedBitSet stores usize ints in the range [0, 16383].
/// It is limited to 128 chunks/offsets and therefore to 128*128 ints.
pub struct FixedBitSet128 {
    xs: Vec<u128>,
    offs: u128,
    sz: usize,
}

impl FixedBitSet128 {
    pub fn new() -> Self {
        FixedBitSet128 {
            xs: Vec::new(),
            offs: 0,
            sz: 0,
        }
    }

    /// Reserves space to hold elements up to `max_elem`.
    pub fn with_capacity(max_elem: usize) -> Self {
        FixedBitSet128 {
            xs: Vec::with_capacity(max_elem / 128 + 1),
            offs: 0,
            sz: 0,
        }
    }

    /// Panics if `elems` is empty.
    pub fn from_sorted_slice(elems: &[usize]) -> Self {
        let mut blips = [(0_usize, 0_u128); 128];
        let mut index: usize = 0;
        let mut off: usize = elems[0] / 128;
        let mut bits = 0_u128;
        let mut offs = 0_u128;
        for i in elems {
            if *i / 128 > off {
                blips[index] = (off, bits);
                index += 1;
                offs |= 1_u128 << off;
                bits = 0;
                off = *i / 128;
            }
            bits |= 1_u128 << (*i % 128);
        }
        if bits > 0 {
            blips[index] = (off, bits);
            index += 1;
            offs |= 1_u128 << off;
        }
        let mut chunks: Vec<u128> = Vec::with_capacity(index);
        unsafe { chunks.set_len(index); }
        for i in 0..index {
            let (_off, bits) = blips[i];
            chunks[i] = bits;
        }
        FixedBitSet128 {
            xs: chunks,
            offs,
            sz: elems.len(),
        }
    }

    /// Panics if `elems` is empty.
    pub fn from_sorted(elems: &Vec<usize>) -> Self {
        let mut blips = [(0_usize, 0_u128); 128];
        let mut index: usize = 0;
        let mut off: usize = elems[0] / 128;
        let mut bits = 0_u128;
        let mut offs = 0_u128;
        for i in elems {
            if *i / 128 > off {
                blips[index] = (off, bits);
                index += 1;
                offs |= 1_u128 << off;
                bits = 0;
                off = *i / 128;
            }
            bits |= 1_u128 << (*i % 128);
        }
        if bits > 0 {
            blips[index] = (off, bits);
            index += 1;
            offs |= 1_u128 << off;
        }
        let mut chunks: Vec<u128> = Vec::with_capacity(index);
        unsafe { chunks.set_len(index); }
        for i in 0..index {
            let (_off, bits) = blips[i];
            chunks[i] = bits;
        }
        FixedBitSet128 {
            xs: chunks,
            offs,
            sz: elems.len(),
        }
    }

    pub fn len(&self) -> usize {
        self.sz
    }

    pub fn clear(&mut self) {
        self.xs.truncate(0);
        self.offs = 0;
        self.sz = 0;
    }

    pub fn contains(&self, x: usize) -> bool {
        let off = x / 128;
        let mask = 1u128 << (x % 128);
        let (found_chunk, index) = self.chunk_index(off);
        found_chunk && self.xs[index] & mask == mask
    }

    pub fn insert(&mut self, x: usize) {
        let off = x / 128;
        let (found_chunk, index) = self.chunk_index(off);
        let mask = 1u128 << (x % 128);
        let found_bit = found_chunk && self.xs[index] & mask == mask;
        if found_bit {
            return;
        }
        // Found the chunk but it didn't have the bit.
        if found_chunk {
            self.xs[index] |= mask;
            self.sz += 1;
            return;
        }
        // Didn't find the chunk (nor the bit).
        let chunk = mask;
        self.xs.insert(index, chunk);
        self.offs |= 1_u128 << off;
        self.sz += 1;
    }

    /// Compute the size of the intersection between `self` and `other` without
    /// actually realizing the set. In cases where only knowing the size is important,
    /// `set1.intersection_size(set2)` is faster than `set1.intersection(set2).len()`.
    pub fn intersection_size(&self, other: &FixedBitSet128) -> usize {
        let mut sz = 0;
        let offs = self.offs & other.offs;
        {
            let mut offs: u64 = offs as u64;
            while offs > 0 {
                let off = offs.trailing_zeros() as usize;
                let i = self.known_chunk_index(off);
                let j = other.known_chunk_index(off);
                let bits1 = &self.xs[i];
                let bits2 = &other.xs[j];
                let and = bits1 & bits2;
                sz += count_ones(and);
                offs = offs ^ (1_u64 << off); // clear bit
            }
        }
        {
            let mut offs: u64 = (offs >> 64) as u64;
            while offs > 0 {
                let off = offs.trailing_zeros() as usize;
                let i = self.known_chunk_index(off + 64);
                let j = other.known_chunk_index(off + 64);
                let bits1 = &self.xs[i];
                let bits2 = &other.xs[j];
                let and = bits1 & bits2;
                sz += count_ones(and);
                offs = offs ^ (1_u64 << off); // clear bit
            }
        }
        sz
    }

    fn chunk_index(&self, off: usize) -> (bool, usize) {
        let found_mask = 1_u128 << off;
        let found = (found_mask & self.offs) == found_mask;
        let mask = !(0xFFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFF_u128 << off);
        let indices = count_ones(mask & self.offs);
        (found, indices)
    }

    fn known_chunk_index(&self, off: usize) -> usize {
        let mask = !(0xFFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFF_u128 << off);
        count_ones(mask & self.offs)
    }

}



#[derive(Clone)]
pub struct Bits128 {
    bits: u128,
    off: i32,
    sz: usize,
}

/// A BitSet stores usize ints.
pub struct BitSet128 {
    xs: Vec<Bits128>,
    sz: usize,
}

impl BitSet128 {
    pub fn new() -> Self {
        BitSet128 {
            xs: Vec::new(),
            sz: 0,
        }
    }

    /// Reserves space to hold elements up to `max_elem`.
    pub fn with_capacity(max_elem: usize) -> Self {
        BitSet128 {
            xs: Vec::with_capacity(max_elem / 128 + 1),
            sz: 0,
        }
    }

    /// Panics if `elems` is empty.
    pub fn from_sorted(elems: &Vec<usize>) -> Self {
        let mut blips = [(0_usize, 0_u128); 128];
        let mut index: usize = 0;
        let mut off: usize = elems[0] / 128;
        let mut bits = 0_u128;
        for i in elems {
            if *i / 128 > off {
                blips[index] = (off, bits);
                index += 1;
                off = *i / 128;
                bits = 0;
            }
            bits |= 1_u128 << (*i % 128);
        }
        if bits > 0 {
            blips[index] = (off, bits);
            index += 1;
        }
        let mut chunks: Vec<Bits128> = Vec::with_capacity(index);
        unsafe { chunks.set_len(index); }
        for i in 0..index {
            let (off, bits) = blips[i];
            chunks[i] = Bits128 {
                bits,
                off: off as i32,
                sz: count_ones(bits),
            };
        }
        BitSet128 {
            xs: chunks,
            sz: elems.len(),
        }
    }

    pub fn len(&self) -> usize {
        self.sz
    }

    pub fn clear(&mut self) {
        self.xs.truncate(0);
        self.sz = 0;
    }

    pub fn contains(&self, x: usize) -> bool {
        let off = (x / 128) as i32;
        let mask = 1u128 << (x % 128);
        let index = chunk_index(&self.xs, off);
        let found = index < self.xs.len() && self.xs[index].off == off;
        found && self.xs[index].bits & mask > 0
    }

    pub fn insert(&mut self, x: usize) {
        let off = (x / 128) as i32;
        let mask = 1u128 << (x % 128);
        let index = chunk_index(&self.xs, off);
        let found = index < self.xs.len() && self.xs[index].off == off;
        if !found {
            let bits = Bits128 {
                bits: mask,
                off,
                sz: 1,
            };
            self.xs.insert(index, bits);
            self.sz += 1;
        } else if self.xs[index].bits & mask == 0 {
            let bits = &mut self.xs[index];
            bits.bits |= mask;
            bits.sz += 1;
            self.sz += 1;
        }
    }

    // This is a special-purpose function specifically for simulating sharding and
    // probably has very limited general utility.
    /// Compute the size of `intersection(union(self, uset), iset)` without forming
    /// either the union or the intersection. Returns `(union_size, intersection_size)`.
    pub fn union_intersection_size(&self, uset: &BitSet128, iset: &BitSet128) -> (usize, usize) {
        let n1 = self.xs.len();
        let n2 = uset.xs.len();
        let n3 = iset.xs.len();
        let mut sz1 = 0;
        let mut sz2 = 0;
        let mut usz = 0;
        let mut isz = 0;
        let mut i = 0;
        let mut j = 0;
        let mut k = 0;
        while i < n1 && j < n2 {
            let bits1 = &self.xs[i];
            let bits2 = &uset.xs[j];
            let (or, orsize) = match bits1.off.cmp(&bits2.off) {
                Ordering::Less => {
                    sz1 += bits1.sz;
                    i += 1;
                    (bits1.bits, bits1.sz)
                }
                Ordering::Greater => {
                    sz2 += bits2.sz;
                    j += 1;
                    (bits2.bits, bits2.sz)
                }
                Ordering::Equal => {
                    sz1 += bits1.sz;
                    sz2 += bits2.sz;
                    let or = bits1.bits | bits2.bits;
                    let orsize = if bits1.bits & bits2.bits == 0 {
                        bits1.sz + bits2.sz
                    } else {
                        count_ones(or)
                    };
                    i += 1;
                    j += 1;
                    (or, orsize)
                }
            };
            usz += orsize;
            let minoff = min(bits1.off, bits2.off);
            while k < n3 && iset.xs[k].off < minoff {
                k += 1;
            }
            if k < n3 {
                let bits3 = &iset.xs[k];
                let and = or & bits3.bits;
                if and > 0 {
                    isz += count_ones(and);
                }
                k += 1;
            }
        }
        if i < n1 {
            usz += self.sz - sz1;
        }
        if j < n2 {
            usz += uset.sz - sz2;
        }
        (usz, isz)
    }

    /// Form a BitSet that is the intersection of `self` and `other`.
    /// Allocates a new BitSet to hold the result.
    pub fn intersection(&self, other: &BitSet128) -> BitSet128 {
        let mut out = BitSet128::new();
        self.intersection_inplace(other, &mut out);
        out
    }

    /// Form a BitSet that is the intersection of `self` and `other`.
    /// Reuses an existing BitSet (`out`) to hold the result.
    pub fn intersection_inplace(&self, other: &BitSet128, out: &mut BitSet128) {
        let n1 = self.xs.len();
        let n2 = other.xs.len();
        out.xs.reserve(min(n1, n2));
        out.xs.truncate(0);
        let mut sz = 0;
        let mut i = 0;
        let mut j = 0;
        while i < n1 && j < n2 {
            let bits1 = &self.xs[i];
            let bits2 = &other.xs[j];
            match bits1.off.cmp(&bits2.off) {
                Ordering::Less => {
                    i += 1;
                }
                Ordering::Greater => {
                    j += 1;
                }
                _ => {
                    let and = bits1.bits & bits2.bits;
                    let ones = count_ones(and);
                    let bits = Bits128 {
                        bits: and,
                        off: bits1.off,
                        sz: ones,
                    };
                    out.xs.push(bits);
                    sz += ones;
                    i += 1;
                    j += 1;
                }
            }
        }
        out.sz = sz;
    }

    /// Compute the size of the intersection between `self` and `other` without
    /// actually realizing the set. In cases where only knowing the size is important,
    /// `set1.intersection_size(set2)` is faster than `set1.intersection(set2).len()`.
    pub fn intersection_size(&self, other: &BitSet128) -> usize {
        let n1 = self.xs.len();
        let n2 = other.xs.len();
        let mut sz = 0;
        let mut i = 0;
        let mut j = 0;
        while i < n1 && j < n2 {
            let bits1 = &self.xs[i];
            let bits2 = &other.xs[j];
            match bits1.off.cmp(&bits2.off) {
                Ordering::Less => { i += 1; }
                Ordering::Greater => { j += 1; }
                _ => {
                    let and = bits1.bits & bits2.bits;
                    sz += count_ones(and);
                    i += 1;
                    j += 1;
                }
            }
        }
        sz
    }

    pub fn union(&self, other: &BitSet128) -> BitSet128 {
        let n1 = self.xs.len();
        let n2 = other.xs.len();
        let mut xs: Vec<Bits128> = Vec::with_capacity(min(n1, n2));
        let mut sz = 0;
        let mut i = 0;
        let mut j = 0;
        while i < n1 && j < n2 {
            let bits1 = &self.xs[i];
            let bits2 = &other.xs[j];
            match bits1.off.cmp(&bits2.off) {
                Ordering::Less => {
                    xs.push(bits1.clone());
                    sz += bits1.sz;
                    i += 1;
                }
                Ordering::Greater => {
                    xs.push(bits2.clone());
                    sz += bits2.sz;
                    j += 1;
                }
                _ => {
                    let or = bits1.bits | bits2.bits;
                    let ones = count_ones(or);
                    let bits = Bits128 {
                        bits: or,
                        off: bits1.off,
                        sz: ones,
                    };
                    xs.push(bits);
                    sz += ones;
                    i += 1;
                    j += 1;
                }
            }
        }
        while i < n1 {
            xs.push(self.xs[i].clone());
            sz += self.xs[i].sz;
            i += 1;
        }
        while j < n2 {
            xs.push(other.xs[j].clone());
            sz += other.xs[j].sz;
            j += 1;
        }
        BitSet128 { xs, sz, }
    }

    pub fn union_size(&self, other: &BitSet128) -> usize {
        let n1 = self.xs.len();
        let n2 = other.xs.len();
        let mut sz1 = 0;
        let mut sz2 = 0;
        let mut usz = 0;
        let mut i = 0;
        let mut j = 0;
        while i < n1 && j < n2 {
            let bits1 = &self.xs[i];
            let bits2 = &other.xs[j];
            match bits1.off.cmp(&bits2.off) {
                Ordering::Less => {
                    usz += bits1.sz;
                    sz1 += bits1.sz;
                    i += 1;
                }
                Ordering::Greater => {
                    usz += bits2.sz;
                    sz2 += bits2.sz;
                    j += 1;
                }
                Ordering::Equal => {
                    if bits1.bits & bits2.bits == 0 {
                        usz += bits1.sz + bits2.sz;
                    } else {
                        usz += count_ones(bits1.bits | bits2.bits);
                    }
                    sz1 += bits1.sz;
                    sz2 += bits2.sz;
                    i += 1;
                    j += 1;
                }
            }
        }
        if i < n1 {
            usz += self.sz - sz1;
        }
        if j < n2 {
            usz += other.sz - sz2;
        }
        usz
    }

    // Reference implementation. Does not try to be clever.
    /// Compute the size of intersection(union(self, uset), iset) without forming
    /// either the union or the intersection.
    pub fn union_intersection_size_ref(&self, uset: &BitSet128, iset: &BitSet128) -> (usize, usize) {
        let n1 = self.xs.len();
        let n2 = uset.xs.len();
        let n3 = iset.xs.len();
        let mut usz = 0;
        let mut isz = 0;
        let mut i = 0;
        let mut j = 0;
        let mut k = 0;
        while i < n1 && j < n2 && k < n3 {
            let bits1 = &self.xs[i];
            let bits2 = &uset.xs[j];
            let bits3 = &iset.xs[k];
            match bits1.off.cmp(&bits2.off) {
                Ordering::Less => {
                    match bits3.off.cmp(&bits1.off) {
                        Ordering::Less => {
                            k += 1;
                        }
                        Ordering::Greater => {
                            usz += bits1.sz;
                            i += 1;
                        }
                        Ordering::Equal => {
                            usz += bits1.sz;
                            isz += count_ones(bits1.bits & bits3.bits);
                            i += 1;
                            k += 1;
                        }
                    }
                }
                Ordering::Greater => {
                    match bits3.off.cmp(&bits2.off) {
                        Ordering::Less => {
                            k += 1;
                        }
                        Ordering::Greater => {
                            usz += bits2.sz;
                            j += 1;
                        }
                        Ordering::Equal => {
                            usz += bits2.sz;
                            isz += count_ones(bits2.bits & bits3.bits);
                            j += 1;
                            k += 1;
                        }
                    }
                }
                Ordering::Equal => {
                    match bits3.off.cmp(&bits1.off) {
                        Ordering::Less => {
                            k += 1;
                        }
                        Ordering::Greater => {
                            let or = bits1.bits | bits2.bits;
                            usz += count_ones(or);
                            i += 1;
                            j += 1;
                        }
                        Ordering::Equal => {
                            let or = bits1.bits | bits2.bits;
                            usz += count_ones(or);
                            isz += count_ones(or & bits3.bits);
                            i += 1;
                            j += 1;
                            k += 1;
                        }
                    }
                }
            }
        }
        while i < n1 && k < n3 {
            let bits1 = &self.xs[i];
            let bits3 = &iset.xs[k];
            match bits3.off.cmp(&bits1.off) {
                Ordering::Less => {
                    k += 1;
                }
                Ordering::Greater => {
                    usz += bits1.sz;
                    i += 1;
                }
                Ordering::Equal => {
                    usz += bits1.sz;
                    isz += count_ones(bits1.bits & bits3.bits);
                    i += 1;
                    k += 1;
                }
            }
        }
        while j < n2 && k < n3 {
            let bits2 = &uset.xs[j];
            let bits3 = &iset.xs[k];
            match bits3.off.cmp(&bits2.off) {
                Ordering::Less => {
                    k += 1;
                }
                Ordering::Greater => {
                    usz += bits2.sz;
                    j += 1;
                }
                Ordering::Equal => {
                    usz += bits2.sz;
                    isz += count_ones(bits2.bits & bits3.bits);
                    j += 1;
                    k += 1;
                }
            }
        }
        while i < n1 && j < n2 {
            let bits1 = &self.xs[i];
            let bits2 = &uset.xs[j];
            match bits1.off.cmp(&bits2.off) {
                Ordering::Less => {
                    usz += bits1.sz;
                    i += 1;
                }
                Ordering::Greater => {
                    usz += bits2.sz;
                    j += 1;
                }
                Ordering::Equal => {
                    let or = bits1.bits | bits2.bits;
                    usz += count_ones(or);
                    i += 1;
                    j += 1;
                }
            }
        }
        while i < n1 {
            let bits1 = &self.xs[i];
            usz += bits1.sz;
            i += 1;
        }
        while j < n2 {
            let bits2 = &self.xs[j];
            usz += bits2.sz;
            j += 1;
        }
        (usz, isz)
    }

}

// Linear search. This is optimized to a binary search by the compiler.
fn chunk_index(xs: &Vec<Bits128>, off: i32) -> usize {
    for i in 0..xs.len() {
        if off <= xs[i].off {
            return i;
        }
    }
    xs.len()
}

#[inline(always)]
fn count_ones(x: u128) -> usize {
    x.count_ones() as usize
}



#[cfg(test)]
mod test_fixed_bitset {
    use super::*;
    use rand::Rng;
    use rand_pcg::Pcg64Mcg;

    #[test]
    fn contains_insert() {
        let n = 2000;
        for seed in vec![123456789, 234567891, 345678912, 456789123] {
            let mut rng = Pcg64Mcg::new(seed);
            let mut bitset1 = BitSet128::new();
            let mut bitset2 = FixedBitSet128::new();
            for _ in 0..n {
                let k = rng.gen_range(0..n);
                assert_eq!(bitset1.contains(k), bitset2.contains(k));
                bitset1.insert(k);
                bitset2.insert(k);
            }
            for k in 0..n {
                assert_eq!(bitset1.contains(k), bitset2.contains(k));
            }
        }
    }

    #[test]
    fn from_sorted() {
        let n = 2000;
        for seed in vec![123456789, 234567891, 345678912, 456789123] {
            let mut rng = Pcg64Mcg::new(seed);
            let mut xs: Vec<usize> = Vec::with_capacity(n);
            let mut bitset0 = BitSet128::new();
            for _ in 0..n {
                let k = rng.gen_range(0..n);
                xs.push(k);
                bitset0.insert(k);
            }
            xs.sort();
            let bitset1 = BitSet128::from_sorted(&xs);
            let bitset2 = FixedBitSet128::from_sorted(&xs);
            for k in 0..n {
                assert_eq!(bitset0.contains(k), bitset1.contains(k));
                assert_eq!(bitset1.contains(k), bitset2.contains(k));
            }
        }
    }

    #[test]
    fn intersection_size() {
        let n = 2000;
        for seed in vec![123456789, 234567891, 345678912, 456789123] {
            let mut rng = Pcg64Mcg::new(seed);
            let mut bitset_128_1 = BitSet128::new();
            for _ in 0..n {
                bitset_128_1.insert(rng.gen_range(0..n));
            }
            let mut bitset_128_2 = BitSet128::new();
            for _ in 0..n {
                bitset_128_2.insert(rng.gen_range(0..n));
            }
            let mut rng = Pcg64Mcg::new(seed);
            let mut bitset_fixed_1 = FixedBitSet128::new();
            for _ in 0..n {
                bitset_fixed_1.insert(rng.gen_range(0..n));
            }
            let mut bitset_fixed_2 = FixedBitSet128::new();
            for _ in 0..n {
                bitset_fixed_2.insert(rng.gen_range(0..n));
            }
            let size_128 = bitset_128_1.intersection_size(&bitset_128_2);
            let size_fixed = bitset_fixed_1.intersection_size(&bitset_fixed_2);
            assert_eq!(size_128, size_fixed);
        }
    }

}


#[cfg(test)]
mod test {
    use super::*;
    use std::collections::HashSet;
    use rand::Rng;
    use rand_pcg::Pcg64Mcg;

    type BitSet = BitSet128;

    // Make sure BitSet behaves correctly by comparing it to HashSet.
    #[test]
    fn hashset_bitset() {
        for seed in vec![123456789, 234567891, 345678912, 456789123, 567891234] {
            let mut rng  = Pcg64Mcg::new(seed);
            for _ in 0..10 {
                let n = rng.gen_range(1000..2000);
                // Populate sets 1
                let mut hashset1 = HashSet::with_capacity(n);
                let mut bitset1 = BitSet::new();
                for _ in 0..n {
                    let x: usize = rng.gen_range(0..n);
                    hashset1.insert(x);
                    bitset1.insert(x);
                    assert_eq!(hashset1.contains(&x), bitset1.contains(x));
                }
                assert_eq!(hashset1.len(), bitset1.len());
                // Populate sets 2
                let mut hashset2 = HashSet::with_capacity(n);
                let mut bitset2 = BitSet::new();
                for _ in 0..n {
                    let x: usize = rng.gen_range(0..n);
                    hashset2.insert(x);
                    bitset2.insert(x);
                    assert_eq!(hashset2.contains(&x), bitset2.contains(x));
                }
                assert_eq!(hashset2.len(), bitset2.len());
                // Check that all pairs contain the same elements
                for i in 0..n {
                    assert_eq!(hashset1.contains(&i), bitset1.contains(i));
                    assert_eq!(hashset2.contains(&i), bitset2.contains(i));
                }
                // Check the union of 1 and 2
                let hashset: HashSet<&usize> = hashset1.union(&hashset2).collect();
                let bitset = bitset1.union(&bitset2);
                assert_eq!(hashset.len(), bitset.len());
                assert_eq!(bitset1.union_size(&bitset2), bitset.len());
                for i in 0..n {
                    assert_eq!(hashset.contains(&i), bitset.contains(i));
                }
                // Check the intersection of 1 and 2
                let hashset: HashSet<&usize> = hashset1.intersection(&hashset2).collect();
                let bitset = bitset1.intersection(&bitset2);
                assert_eq!(hashset.len(), bitset.len());
                assert_eq!(bitset1.intersection_size(&bitset2), bitset.len());
                for i in 0..n {
                    assert_eq!(hashset.contains(&i), bitset.contains(i));
                }
            }
        }
    }

    #[test]
    fn intersection_inplace() {
        let n = 1000;
        for seed in vec![123456789, 234567891, 345678912, 456789123] {
            let mut rng = Pcg64Mcg::new(seed);
            let mut bitset1 = BitSet::new();
            for _ in 0..n {
                bitset1.insert(rng.gen_range(0..n));
            }
            let mut bitset2 = BitSet::new();
            for _ in 0..n {
                bitset2.insert(rng.gen_range(0..n));
            }
            let intersection = bitset1.intersection(&bitset2);
            let mut inplace = BitSet::new();
            bitset1.intersection_inplace(&bitset2, &mut inplace);
            assert_eq!(inplace.len(), intersection.len());
        }
    }

    #[test]
    fn union_intersection_size() {
        let n = 1000;
        for seed in vec![123456789, 234567891, 345678912, 456789123] {
            let mut rng = Pcg64Mcg::new(seed);
            let mut bitset1 = BitSet::new();
            for _ in 0..n {
                bitset1.insert(rng.gen_range(0..n));
            }
            let mut bitset2 = BitSet::new();
            for _ in 0..n {
                bitset2.insert(rng.gen_range(0..n));
            }
            let mut bitset3 = BitSet::new();
            for _ in 0..n {
                bitset3.insert(rng.gen_range(0..n));
            }
            let union = bitset1.union(&bitset2);
            let intersection = union.intersection(&bitset3);
            let (usz, isz) = bitset1.union_intersection_size(&bitset2, &bitset3);
            assert_eq!(usz, union.len());
            assert_eq!(isz, intersection.len());
        }
    }

    #[test]
    fn intersection_size() {
        let n = 2000;
        for seed in vec![123456789, 234567891, 345678912, 456789123] {
            let mut rng = Pcg64Mcg::new(seed);
            let mut bitset1 = BitSet::new();
            for _ in 0..n {
                bitset1.insert(rng.gen_range(0..n));
            }
            let mut bitset2 = BitSet::new();
            for _ in 0..n {
                bitset2.insert(rng.gen_range(0..n));
            }
            let intersection = bitset1.intersection(&bitset2);
            let size = bitset1.intersection_size(&bitset2);
            assert_eq!(size, intersection.len());
        }
    }

    #[test]
    fn union_size() {
        let n = 2000;
        for seed in vec![123456789, 234567891, 345678912, 456789123] {
            let mut rng = Pcg64Mcg::new(seed);
            let mut bitset1 = BitSet128::new();
            for _ in 0..n {
                bitset1.insert(rng.gen_range(0..n));
            }
            let mut bitset2 = BitSet128::new();
            for _ in 0..n {
                bitset2.insert(rng.gen_range(0..n));
            }
            let union = bitset1.union(&bitset2);
            let size = bitset1.union_size(&bitset2);
            assert_eq!(size, union.len());
        }
    }

    /// Should have |union(A,B)| = |A|+|B| - |intersection(A,B)|.
    #[test]
    fn union_size_shortcut() {
        for seed in vec![123456789, 234567891, 345678912, 456789123, 567891234] {
            let mut rng = Pcg64Mcg::new(seed);
            let n = rng.gen_range(10..1000);
            for _ in 0..100 {
                let mut bitset1 = BitSet::new();
                for _ in 0..n {
                    bitset1.insert(rng.gen_range(0..n));
                }
                let mut bitset2 = BitSet::new();
                for _ in 0..n {
                    bitset2.insert(rng.gen_range(0..n));
                }
                let union = bitset1.union(&bitset2);
                assert_eq!(union.len(), bitset1.union_size(&bitset2));
                let intersection = bitset1.intersection(&bitset2);
                assert_eq!(union.len(), bitset1.len() + bitset2.len() - intersection.len());
                let intersection_size = bitset1.intersection_size(&bitset2);
                assert_eq!(union.len(), bitset1.len() + bitset2.len() - intersection_size);
            }
        }
    }

}
