use std::{fmt, cmp::Ordering, hash::{Hash, Hasher}, ops::BitXor};
use rand_pcg::Pcg64Mcg;
use rand::Rng;

pub const BITS: usize = 256;
pub const DIGITS: usize = 2;

type U256Arr = [u128; DIGITS];

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct U256arr {
    words: U256Arr
}

impl U256arr {
    pub fn new(words: U256Arr) -> Self {
        U256arr { words }
    }

    pub fn distance(&self, other: &U256arr) -> Self {
        self ^ other
    }

    pub fn compare(&self, a: &U256arr, b: &U256arr) -> Ordering {
        self.distance(a).cmp(&self.distance(b))
    }

}

// TODO: Check if it matters whether we only look at the first 128 bits or all.
impl Hash for U256arr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.words[0].hash(state);
    }
}

impl From<U256Arr> for U256arr {
    fn from(xs: U256Arr) -> Self {
        U256arr::new(xs)
    }
}

impl BitXor<&U256arr> for &U256arr {
    type Output = U256arr;
    fn bitxor(self, right: &U256arr) -> Self::Output {
        let xs = [self.words[0] ^ right.words[0], self.words[1] ^ right.words[1]];
        U256arr::from(xs)
    }
}

impl fmt::Display for U256arr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.words)
    }
}

pub fn generate_ids(count: usize, rng: &mut Pcg64Mcg) -> Vec<U256arr> {
    let mut ids: Vec<U256arr> = Vec::with_capacity(count);
    for _ in 0..count {
        let n: U256Arr = rng.gen();
        ids.push(U256arr::new(n));
    }
    ids
}

pub fn generate_ids_inplace(count: usize, rng: &mut Pcg64Mcg, ids: &mut Vec<U256arr>) {
    ids.truncate(0);
    for _ in 0..count {
        ids.push(U256arr::new(rng.gen()));
    }
}
