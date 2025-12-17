use std::simd::u64x4;
use std::{fmt, cmp::Ordering, hash::{Hash, Hasher}, ops::BitXor};
use rand_pcg::Pcg64Mcg;
use rand::Rng;

pub type U256 = U256simd;

pub const BITS: usize = 256;

//pub type U256Arr = [u128; 2];
type U256Arr = [u64; 4];

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct U256simd {
    words: u64x4
}

// Proves that generating [u128; 2] and [u64; 4] are exactly the same.
#[test]
fn test_rng_gen_u128x2_u64x4_equivalence() {
    let mut rng1 = Pcg64Mcg::new(123);
    let mut rng2 = Pcg64Mcg::new(123);
    for _ in 0..99999 {
        let u128x2: [u128; 2] = rng1.gen();
        let u64x4: [u64; 4] = rng2.gen();
        assert_eq!(u128x2, u64x4_array_to_u128x2_array(u64x4));
        assert_eq!(u64x4, u128x2_array_to_u64x4_array(u128x2));
    }
}

// Test conversion between [u128; 2] and [u64; 4].
#[test]
fn test_u128x2_u64x4_conversion() {
    let mut rng = Pcg64Mcg::new(123);
    for _ in 0..99999 {
        let u128x2: [u128; 2] = rng.gen();
        let u64x4: [u64; 4] = u128x2_array_to_u64x4_array(u128x2);
        let u128x2_alt: [u128; 2] = u64x4_array_to_u128x2_array(u64x4);
        assert_eq!(u128x2, u128x2_alt);
        let u64x4: [u64; 4] = rng.gen();
        let u128x2: [u128; 2] = u64x4_array_to_u128x2_array(u64x4);
        let u64x4_alt: [u64; 4] = u128x2_array_to_u64x4_array(u128x2);
        assert_eq!(u64x4, u64x4_alt);
    }
}

#[test]
fn test_i256_u256_equivalence() {
    use crate::u256_arr::U256arr;
    let mut rng1 = Pcg64Mcg::new(123);
    let mut rng2 = Pcg64Mcg::new(123);
    for _ in 0..99999 {
        let u256a = U256simd::new(rng1.gen());
        let u256b = U256simd::new(rng1.gen());
        let u256c = U256simd::new(rng1.gen());
        let i256a = U256arr::new(rng2.gen());
        let i256b = U256arr::new(rng2.gen());
        let i256c = U256arr::new(rng2.gen());
        assert_eq!(u256a < u256b, i256a < i256b);
        assert_eq!((&u256a ^ &u256b) < u256c, (&i256a ^ &i256b) < i256c);
    }
}

#[allow(unused)]
fn u128x2_array_to_u64x4_array(u128x2: [u128; 2]) -> [u64; 4] {
    [u128x2[0] as u64, (u128x2[0] >> 64) as u64, u128x2[1] as u64, (u128x2[1] >> 64) as u64]
}

#[allow(unused)]
fn u64x4_array_to_u128x2_array(u64x4: [u64; 4]) -> [u128; 2] {
    [(u64x4[0] as u128) | ((u64x4[1] as u128) << 64), (u64x4[2] as u128) | ((u64x4[3] as u128) << 64)]
}

fn u64x4_array_to_u64x4_simd(xs: [u64; 4]) -> u64x4 {
    u64x4::from_array([xs[1], xs[0], xs[3], xs[2]])
}

#[allow(unused)]
fn u128x2_array_to_u64x4_simd(xs: [u128; 2]) -> u64x4 {
    u64x4::from_array([(xs[0] >> 64) as u64, xs[0] as u64, (xs[1] >> 64) as u64, xs[1] as u64])
}

impl U256simd {
    pub fn new(xs: U256Arr) -> Self {
        //let words = u128x2_array_to_u64x4_simd(xs);
        let words = u64x4_array_to_u64x4_simd(xs);
        U256simd { words }
    }

    pub fn distance(&self, other: &U256simd) -> Self {
        self ^ other
    }

    pub fn compare(&self, a: &U256simd, b: &U256simd) -> Ordering {
        self.distance(a).cmp(&self.distance(b))
    }

    /// `b <= 64`
    pub fn prefix(&self, b: usize) -> usize {
        if b == 0 {
            0
        } else if b <= 64 {
            (self.words[0] >> (64 - b)) as usize
        } else {
            panic!()
        }
    }

}

// TODO: Check if it matters whether we only look at the first 128 bits or all.
impl Hash for U256simd {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.words[0].hash(state);
    }
}

impl From<U256Arr> for U256simd {
    fn from(xs: U256Arr) -> Self {
        U256simd::new(xs)
    }
}

impl From<u64x4> for U256simd {
    fn from(words: u64x4) -> Self {
        U256simd { words }
    }
}

impl BitXor<&U256simd> for &U256simd {
    type Output = U256simd;
    fn bitxor(self, right: &U256simd) -> Self::Output {
        let xs = self.words ^ right.words;
        U256simd::from(xs)
    }
}

impl fmt::Display for U256simd {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.words)
    }
}

pub fn generate_ids(count: usize, rng: &mut Pcg64Mcg) -> Vec<U256simd> {
    let mut ids: Vec<U256simd> = Vec::with_capacity(count);
    for _ in 0..count {
        let n: U256Arr = rng.gen();
        ids.push(U256simd::new(n));
    }
    ids
}

pub fn generate_ids_inplace(count: usize, rng: &mut Pcg64Mcg, ids: &mut Vec<U256simd>) {
    ids.truncate(0);
    for _ in 0..count {
        ids.push(U256simd::new(rng.gen()));
    }
}
