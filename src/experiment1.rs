use crate::u256_simd::*;
use crate::bitset::*;
use crate::util::*;
use std::collections::HashSet;
use std::time::Instant;
use rand::Rng;
use rand_pcg::Pcg64Mcg;

// ============================================================================
// "Real" (BitSet-based extra clever) implementation
// ----------------------------------------------------------------------------
// ============================================================================

pub fn experiment1(params: &Params, step: usize) -> usize {
    let mut r = params.shard_size;
    let mut restart = true;
    while restart {
        restart = false;
        let mut rng = Pcg64Mcg::new(params.seed);
        for _ in 0..params.reps {
            restart = !test_r(&params, r, &mut rng);
            if restart {
                println!("FAIL r = {}", r);
                r += step;
                break;
            }
        }
    }
    r
}

pub fn test_r(params: &Params, r: usize, mut rng: &mut Pcg64Mcg) -> bool {
    let n_nodes = params.n_nodes;
    let n_accounts = params.n_accounts;
    // Generate nodes and shards
    let nodes = generate_ids(n_nodes, &mut rng);
    let mut sharder = Sharder3::new(nodes, r);
    let mut shards: Vec<BitSet> = Vec::with_capacity(n_accounts);
    for _ in 0..n_accounts {
        let account = U256::new(rng.gen());
        let shard = sharder.make_shard(account, r);
        shards.push(shard);
    }
    // Now test the shards...
    all_shards_safe(&shards, &params, rng.gen())
}

fn all_shards_safe(shards: &Vec<BitSet>, params: &Params, seed: u128) -> bool {
    let mut rng = Pcg64Mcg::new(seed);
    let n_nodes = params.n_nodes;
    let n_byz = n_nodes / params.byz_frac;
    let shard_tol = tolerance(shards[0].len(), params.byz_tol);
    for _ in 0..params.iter {
        let byz: BitSet = random_bitset(n_nodes, n_byz, &mut rng);
        for shard in shards {
            if byz.intersection_size(shard) > shard_tol {
                return false;
            }
        }
    }
    true
}


// ============================================================================
// Naive reference implementation
// ----------------------------------------------------------------------------
// This is the naive and straightforward implementation that is "obviously correct".
// ============================================================================

pub fn experiment1_naive(params: &Params, step: usize) -> usize {
    let timer = Instant::now();
    let mut r = params.shard_size;
    let mut restart = true;
    while restart {
        restart = false;
        let mut rng = Pcg64Mcg::new(params.seed);
        for _ in 0..params.reps {
            restart = !test_r_naive(&params, r, &mut rng);
            if restart {
                println!("FAIL r = {}", r);
                r += step;
                break;
            }
        }
    }
    let secs = timer.elapsed().as_secs();
    println!("After {} sec:", secs);
    println!("F = 1/{}, f = 1/{}, N = {}, r = {}", params.byz_frac, params.byz_tol, params.n_nodes, r);
    r
}

fn test_r_naive(params: &Params, r: usize, mut rng: &mut Pcg64Mcg) -> bool {
    let n_nodes = params.n_nodes;
    let n_accounts = params.n_accounts;
    // Generate nodes and shards
    let nodes = generate_ids(n_nodes, &mut rng);   // OPT: reuse memory
    let mut shards: Vec<HashSet<U256>> = Vec::with_capacity(n_accounts);
    for _ in 0..n_accounts {
        let account = U256::new(rng.gen());
        let shard = shard_hashset(&account, nodes.clone(), r);
        shards.push(shard);
    }
    // Now test the shards...
    all_shards_safe_naive(&shards, &nodes, &params, rng.gen())
}

fn all_shards_safe_naive(shards: &Vec<HashSet<U256>>, nodes: &Vec<U256>, params: &Params, seed: u128) -> bool {
    assert_eq!(nodes.len(), params.n_nodes);
    let mut rng = Pcg64Mcg::new(seed);
    let n_byz = nodes.len() / params.byz_frac;
    let tolerance = tolerance(shards[0].len(), params.byz_tol);
    for _ in 0..params.iter {
        let byz: HashSet<U256> = random_hashset(nodes, n_byz, &mut rng);
        for shard in shards {
            let intersection: HashSet<&U256> = byz.intersection(shard).collect();
            if intersection.len() > tolerance {
                return false;
            }
        }
    }
    true
}


#[cfg(test)]
mod test {
    use super::*;

    #[test] // r = 71-81
    fn compare_to_naive_1a() {
        let n_nodes = 100;
        for seed in vec![123456789, 234567891, 345678912, 456789123] {
            let params = Params {
                n_nodes,
                n_accounts: 2*n_nodes,
                byz_frac: 4,
                byz_tol: 3,
                shard_size: 21,
                reps: 5,
                iter: 10,
                seed,
            };
            let reference = experiment1_naive(&params, 12);
            let result = experiment1(&params, 12);
            assert_eq!(result, reference);
        }
    }

    #[test] // 171-211
    fn compare_to_naive_1b() {
        let n_nodes = 500;
        for seed in vec![123456789, 234567891, 345678912, 456789123] {
            let params = Params {
                n_nodes,
                n_accounts: 2*n_nodes,
                byz_frac: 4,
                byz_tol: 3,
                shard_size: 11,
                reps: 5,
                iter: 10,
                seed,
            };
            let reference = experiment1_naive(&params, 30);
            let result = experiment1(&params, 30);
            assert_eq!(result, reference);
        }
    }

    #[test] // r = 51-61
    fn compare_to_naive_2a() {
        let n_nodes = 100;
        for seed in vec![123456789, 234567891, 345678912, 456789123] {
            let params = Params {
                n_nodes,
                n_accounts: 2*n_nodes,
                byz_frac: 5,
                byz_tol: 3,
                shard_size: 11,
                reps: 5,
                iter: 10,
                seed,
            };
            let reference = experiment1_naive(&params, 10);
            let result = experiment1(&params, 10);
            assert_eq!(result, reference);
        }
    }

    #[test] // 91-121
    fn compare_to_naive_2b() {
        let n_nodes = 500;
        for seed in vec![123456789, 234567891, 345678912, 456789123] {
            let params = Params {
                n_nodes,
                n_accounts: 2*n_nodes,
                byz_frac: 5,
                byz_tol: 3,
                shard_size: 11,
                reps: 5,
                iter: 10,
                seed,
            };
            let reference = experiment1_naive(&params, 10);
            let result = experiment1(&params, 10);
            assert_eq!(result, reference);
        }
    }

    #[test] // r = 41-51
    fn compare_to_naive_3a() {
        let n_nodes = 100;
        for seed in vec![123456789, 234567891, 345678912, 456789123] {
            let params = Params {
                n_nodes,
                n_accounts: 2*n_nodes,
                byz_frac: 3,
                byz_tol: 2,
                shard_size: 11,
                reps: 5,
                iter: 10,
                seed,
            };
            let reference = experiment1_naive(&params, 8);
            let result = experiment1(&params, 8);
            assert_eq!(result, reference);
        }
    }

    #[test] // 81-101
    fn compare_to_naive_3b() {
        let n_nodes = 500;
        for seed in vec![123456789, 234567891, 345678912, 456789123] {
            let params = Params {
                n_nodes,
                n_accounts: 2*n_nodes,
                byz_frac: 3,
                byz_tol: 2,
                shard_size: 11,
                reps: 5,
                iter: 10,
                seed,
            };
            let reference = experiment1_naive(&params, 20);
            let result = experiment1(&params, 20);
            assert_eq!(result, reference);
        }
    }

    #[test] // r = 21-31
    fn compare_to_naive_4a() {
        let n_nodes = 100;
        for seed in vec![123456789, 234567891, 345678912, 456789123] {
            let params = Params {
                n_nodes,
                n_accounts: 2*n_nodes,
                byz_frac: 4,
                byz_tol: 2,
                shard_size: 7,
                reps: 5,
                iter: 10,
                seed,
            };
            let reference = experiment1_naive(&params, 6);
            let result = experiment1(&params, 6);
            assert_eq!(result, reference);
        }
    }

    #[test] // 31-41
    fn compare_to_naive_4b() {
        let n_nodes = 500;
        for seed in vec![123456789, 234567891, 345678912, 456789123] {
            let params = Params {
                n_nodes,
                n_accounts: 2*n_nodes,
                byz_frac: 5,
                byz_tol: 2,
                shard_size: 11,
                reps: 5,
                iter: 10,
                seed,
            };
            let reference = experiment1_naive(&params, 6);
            let result = experiment1(&params, 6);
            assert_eq!(result, reference);
        }
    }

    #[test] // r = 15-23
    fn compare_to_naive_5a() {
        let n_nodes = 100;
        for seed in vec![123456789, 234567891, 345678912, 456789123] {
            let params = Params {
                n_nodes,
                n_accounts: 2*n_nodes,
                byz_frac: 5,
                byz_tol: 2,
                shard_size: 7,
                reps: 5,
                iter: 10,
                seed,
            };
            let reference = experiment1_naive(&params, 4);
            let result = experiment1(&params, 4);
            assert_eq!(result, reference);
        }
    }

    #[test]
    fn compare_to_naive_5b() {
        let n_nodes = 300;
        for seed in vec![123456789, 234567891, 345678912, 456789123] {
            let params = Params {
                n_nodes,
                n_accounts: 2*n_nodes,
                byz_frac: 5,
                byz_tol: 2,
                shard_size: 11,
                reps: 5,
                iter: 10,
                seed,
            };
            let reference = experiment1_naive(&params, 4);
            let result = experiment1(&params, 4);
            assert_eq!(result, reference);
        }
    }

}
