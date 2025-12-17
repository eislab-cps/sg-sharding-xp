use crate::u256_simd::*;
use crate::bitset::*;
use crate::util::*;
use std::collections::HashSet;
use std::time::Instant;
use rand::Rng;
use rand_pcg::Pcg64Mcg;


// ============================================================================
// Real implementation
// ----------------------------------------------------------------------------
// Because we avoid rebuilding overlapping shards from scratch and instead
// start with the largest size and truncate, we generate IDs in a different
// order. Therefore, the results are not bit-for-bit identical to the ref/naive
// implementations. But if the averages are approximately equal (total difference
// close to 0), then the two implementations are doing the same thing and are
// equally correct. (See test case `compare_to_ref_1`)
// ============================================================================

/// All parameters are fixed, except for the number of accounts (or shards).
pub fn experiment3(params: &Params, mut ms: Vec<usize>) -> Vec<Vec<usize>> {
    ms.reverse();
    let n_nodes = params.n_nodes;
    let shard_size = params.shard_size;
    let max_accounts = ms[0];
    let mut results: Vec<Vec<usize>> = (0..ms.len()).map(|_| vec![0].repeat(params.reps)).collect();
    let mut rng = Pcg64Mcg::new(params.seed);
    for i in 0..params.reps {
        let mut rng = Pcg64Mcg::new(rng.gen());
        let nodes = generate_ids(n_nodes, &mut rng);
        let accounts = generate_ids(max_accounts, &mut rng);
        let mut sharder = Sharder3::new(nodes, shard_size);
        let mut shards: Vec<BitSet> = Vec::with_capacity(max_accounts);
        for account in accounts {
            let shard = sharder.make_shard(account, shard_size);
            shards.push(shard);
        }
        for (j,m) in ms.iter().enumerate() {
            shards.truncate(*m);
            let params = params.with_n_accounts(*m);
            results[j][i] += count_compromised_iter(&shards, &params, rng.gen())
        }
    }
    results.reverse();
    results
}

/// Count **the number of iterations** where there is *at least one* compromised shard.
fn count_compromised_iter(shards: &Vec<BitSet>, params: &Params, seed: u128) -> usize {
    let mut rng = Pcg64Mcg::new(seed);
    let n_nodes = params.n_nodes;
    let n_byz = n_nodes / params.byz_frac;
    let mut count = 0;
    for _ in 0..params.iter {
        let byz: BitSet = random_bitset(n_nodes, n_byz, &mut rng);
        count += any_compromised(shards, params, &byz);
    }
    count
}

/// Returns `1` as soon as a compromised shard is found, or `0` if all were OK.
pub fn any_compromised(shards: &Vec<BitSet>, params: &Params, byz: &BitSet) -> usize {
    let shard_tol = tolerance(shards[0].len(), params.byz_tol);
    for shard in shards {
        if byz.intersection_size(shard) > shard_tol {
            return 1;
        }
    }
    0
}


// ============================================================================
// Reference implementation
// ----------------------------------------------------------------------------
// Uses Sharder (not Sharder3) and does not reuse work.
// ============================================================================

/// All parameters are fixed, except for the number of accounts (or shards).
pub fn experiment3_ref(params: &Params, ms: Vec<usize>) -> Vec<Vec<usize>> {
    let timer = Instant::now();
    let n_nodes = params.n_nodes;
    let shard_size = params.shard_size;
    let mut results: Vec<Vec<usize>> = (0..ms.len()).map(|_| vec![0].repeat(params.reps)).collect();
    let mut rng = Pcg64Mcg::new(params.seed);
    for i in 0..params.reps {
        let mut rng = Pcg64Mcg::new(rng.gen());
        let nodes = generate_ids(n_nodes, &mut rng);
        let mut sharder = Sharder::new(nodes);
        for (j,m) in ms.iter().enumerate() {
            let params = params.with_n_accounts(*m);
            let mut shards: Vec<BitSet> = Vec::with_capacity(*m);
            let accounts = generate_ids(*m, &mut rng);
            for account in accounts {
                let shard = sharder.make_shard(account, shard_size);
                shards.push(shard);
            }
            results[j][i] += count_compromised_iter(&shards, &params, rng.gen())
        }
    }
    let secs = timer.elapsed().as_secs();
    for i in 0..params.reps {
        for (j,m) in ms.iter().enumerate() {
            println!("F = 1/{}, f = 1/{}, N = {}, m = {}, r = {}, fails = {:?}", params.byz_frac, params.byz_tol, params.n_nodes, m, params.shard_size, results[j][i]);
        }
    }
    println!("After {} sec", secs);
    results
}


// ============================================================================
// Naive implementation
// ----------------------------------------------------------------------------
// ============================================================================

/// All parameters are fixed, except for the number of accounts (or shards).
pub fn experiment3_naive(params: &Params, ms: Vec<usize>) -> Vec<Vec<usize>> {
    let timer = Instant::now();
    let n_nodes = params.n_nodes;
    let shard_size = params.shard_size;
    let mut results: Vec<Vec<usize>> = (0..ms.len())
        .map(|_| vec![0].repeat(params.reps)).collect();
    let mut rng = Pcg64Mcg::new(params.seed);
    for i in 0..params.reps {
        let mut rng = Pcg64Mcg::new(rng.gen());
        let nodes = generate_ids(n_nodes, &mut rng);
        for (j,m) in ms.iter().enumerate() {
            let params = params.with_n_accounts(*m);
            let mut shards: Vec<HashSet<U256>> = Vec::with_capacity(*m);
            let accounts = generate_ids(*m, &mut rng);
            for account in accounts {
                let shard = shard_hashset(&account, nodes.clone(), shard_size);
                shards.push(shard);
            }
            results[j][i] += count_compromised_naive_iter(&shards, &nodes, &params, rng.gen())
        }
    }
    let secs = timer.elapsed().as_secs();
    for i in 0..params.reps {
        for (j,m) in ms.iter().enumerate() {
            println!("F = 1/{}, f = 1/{}, N = {}, m = {}, r = {}, fails = {:?}", params.byz_frac, params.byz_tol, params.n_nodes, m, params.shard_size, results[j][i]);
        }
    }
    println!("After {} sec", secs);
    results
}

/// Count **the number of iterations** where there is *at least one* compromised shard.
fn count_compromised_naive_iter(shards: &Vec<HashSet<U256>>, nodes: &Vec<U256>, params: &Params, seed: u128) -> usize {
    let mut rng = Pcg64Mcg::new(seed);
    let n_nodes = params.n_nodes;
    let n_byz = n_nodes / params.byz_frac;
    let mut count = 0;
    for _ in 0..params.iter {
        let byz: HashSet<U256> = random_hashset(nodes, n_byz, &mut rng);
        count += any_compromised_naive(shards, params, &byz);
    }
    count
}

/// Returns `1` as soon as a compromised shard is found, or `0` if all were OK.
pub fn any_compromised_naive(shards: &Vec<HashSet<U256>>, params: &Params, byz: &HashSet<U256>) -> usize {
    let shard_tol = tolerance(shards[0].len(), params.byz_tol);
    for shard in shards {
        let intersection: HashSet<&U256> = byz.intersection(shard).collect();
        if intersection.len() > shard_tol {
            return 1;
        }
    }
    0
}


#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn compare_to_naive_1() {
        let n_nodes = 500;
        for seed in vec![123456789, 234567891, 345678912, 456789123] {
            let params = Params {
                n_nodes,
                n_accounts: 2*n_nodes,
                byz_frac: 5,
                byz_tol: 3,
                shard_size: 51,
                reps: 5,
                iter: 10,
                seed,
            };
            let reference = experiment3_naive(&params, vec![100, 500, 1000]);
            let result = experiment3_ref(&params, vec![100, 500, 1000]);
            assert_eq!(result, reference);
        }
    }

    #[test]
    fn compare_to_naive_2() {
        let n_nodes = 500;
        for seed in vec![123456789, 234567891, 345678912, 456789123] {
            let params = Params {
                n_nodes,
                n_accounts: 2*n_nodes,
                byz_frac: 3,
                byz_tol: 2,
                shard_size: 41,
                reps: 5,
                iter: 10,
                seed,
            };
            let reference = experiment3_naive(&params, vec![100, 500, 1000]);
            let result = experiment3_ref(&params, vec![100, 500, 1000]);
            assert_eq!(result, reference);
        }
    }

    #[test]
    fn compare_to_ref_1() {
        let n_nodes = 500;
        let ms = vec![100, 500, 1000];
        let mut all_diffs: Vec<f64> = Vec::with_capacity(ms.len());
        for seed in vec![123456789, 234567891, 345678912, 456789123] {
            let params = Params {
                n_nodes,
                n_accounts: 0,
                byz_frac: 3,
                byz_tol: 2,
                shard_size: 41,
                reps: 5,
                iter: 5000,
                seed,
            };
            let reference = experiment3_ref(&params, ms.clone());
            let result = experiment3(&params, ms.clone());
            let tot_it = params.reps*params.iter;
            let sums_ref: Vec<f64> = reference.iter()
                .map(|v| v.iter().sum())
                .map(|s: usize| s as f64 / tot_it as f64)
                .collect();
            let sums: Vec<f64> = result.iter()
                .map(|v| v.iter().sum())
                .map(|s: usize| s as f64 / tot_it as f64)
                .collect();
            println!("sums_ref: {:?}", sums_ref);
            println!("sums: {:?}", sums);
            // The results are not bit-for-bit identical, but we don't need that.
            // As long as they are close, that means they are equally correct!
            for i in 0..ms.len() {
                assert!((sums_ref[i] * 0.975) <= (sums[i] as f64));
                assert!(sums[i] <= (sums_ref[i] * 1.025));
                let diff = sums_ref[i] - sums[i];
                println!("diff: {:?}", diff);
                assert!(diff.abs() <= 0.01);
                all_diffs.push(diff);
            }
        }
        // Averaging over all differences.
        let diff_sum: f64 = all_diffs.iter().sum();
        println!("diff_sum: {:?}", diff_sum);
        assert!(diff_sum.abs() <= 0.002);
    }


}
