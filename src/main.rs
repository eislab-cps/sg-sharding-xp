#![feature(portable_simd)]

use sg_sharding_xp::experiment1::*;
use sg_sharding_xp::experiment2::*;
use sg_sharding_xp::experiment3::*;
use sg_sharding_xp::util::Params;
use std::time::Instant;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(name = "sg-sharding-xp", about = "ScaleGraph sharding simulation experiments")]
struct Opt {
    #[structopt(short, long)]
    experiment: usize,
    #[structopt(short, long)]
    seed: Option<u128>,
    #[structopt(short, long)]
    reps: Option<usize>,
    #[structopt(short, long)]
    iter: Option<usize>,
    #[structopt(long)]
    shard_size: Option<usize>,
    #[structopt(short, long)]
    nodes: Option<usize>,
}

fn main() {
    let seeds = vec![
        0123456789,
        1234567890,
        2345678901,
        3456789012,
        4567890123,
        5678901234,
        6789012345,
        7890123456,
        8901234567,
        9012345678,
        01234567890,
        12345678900,
        23456789010,
        34567890120,
        45678901230,
        56789012340,
        67890123450,
        78901234560,
        89012345670,
        90123456780,
    ];
    let opt = Opt::from_args();
    let seed = match opt.seed {
        Some(n) => {
            if n < seeds.len() as u128 {
                seeds[n as usize]
            } else {
                n
            }
        }
        None => seeds[0],
    };
    match opt.experiment {
        1 => start_experiment1(seed, opt.reps, opt.iter),
        2 => start_experiment2(seed, opt.reps, opt.iter, opt.nodes),
        3 => start_experiment3(seed, opt.reps, opt.iter, opt.nodes, opt.shard_size),
        31 => start_experiment3_sup(seed, opt.reps, opt.iter, opt.nodes, opt.shard_size),
        _ => {
            panic!()
        }
    }
}


// ----- Experiment 1 -----

fn start_experiment1(seed: u128, reps: Option<usize>, iter: Option<usize>) {
    let step = 10;
    let reps = reps.unwrap_or(1_000);
    let iter = iter.unwrap_or(5_000);
    let n_nodes = vec![125, 250, 500, 1_000, 2_000, 4_000];
    run_experiment1(&n_nodes, 5, 2, reps, iter, step, seed);
    run_experiment1(&n_nodes, 4, 2, reps, iter, step, seed);
    run_experiment1(&n_nodes, 3, 2, reps, iter, step, seed);
    run_experiment1(&n_nodes, 5, 3, reps, iter, step, seed);
    let n_nodes = vec![125, 250, 500, 1_000, 2_000];
    run_experiment1(&n_nodes, 4, 3, reps, iter, step, seed);
    let n_nodes = vec![8_000, 16_000];
    run_experiment1(&n_nodes, 5, 2, reps, iter, step, seed);
    run_experiment1(&n_nodes, 4, 2, reps, iter, step, seed);
    run_experiment1(&n_nodes, 3, 2, reps, iter, step, seed);
}

fn run_experiment1(n_nodes: &Vec<usize>, byz_frac: usize, byz_tol: usize, reps: usize, iter: usize, step: usize, seed: u128) {
    for n_nodes in n_nodes.clone() {
        let n_accounts = 2*n_nodes;
        let params = Params {
            n_nodes,
            n_accounts,
            byz_frac,
            byz_tol,
            shard_size: 11, // Starting size
            reps,
            iter,
            seed,
        };
        let timer = Instant::now();
        println!("experiment_1({}, {}, {}, {}, {})...", n_nodes, byz_frac, reps, iter, seed);
        let r = experiment1(&params, step);
        let ms = timer.elapsed().as_millis();
        println!("N={}, F=1/{}, f=1/{}, m={}, r={}", n_nodes, byz_frac, byz_tol, n_accounts, r);
        println!("Took {} sec", (ms as f64) / 1000.0);
    }
}

// ----- Experiment 2 -----

fn start_experiment2(seed: u128, reps: Option<usize>, iter: Option<usize>, n_nodes: Option<usize>) {
    let n_nodes = n_nodes.unwrap_or(2_000);
    let reps = reps.unwrap_or(1_000);
    let iter = iter.unwrap_or(5_000);
    let shard_sizes = vec![11, 21, 31, 41, 51, 61, 71, 81, 91, 101];
    run_experiment2(shard_sizes, n_nodes, 4, 2, reps, iter, seed);
}

fn run_experiment2(shard_sizes: Vec<usize>, n_nodes: usize, byz_frac: usize, byz_tol: usize, reps: usize, iter: usize, seed: u128) {
    let timer = Instant::now();
    let n_accounts = 2*n_nodes;
    let params = Params {
        n_nodes,
        n_accounts,
        byz_frac,
        byz_tol,
        shard_size: 11,
        reps,
        iter,
        seed,
    };
    println!("experiment_2({}, {}, {}, {}, {}, {:?})...", n_nodes, byz_frac, reps, iter, seed, shard_sizes);
    let results = experiment2(&params, shard_sizes.clone());
    let secs = timer.elapsed().as_secs();
    for i in 0..params.reps {
        for (j,r) in shard_sizes.iter().enumerate() {
            println!("N={}, F=1/{}, f=1/{}, m={}, r={}, fails={:?}", n_nodes, byz_frac, byz_tol, n_accounts, r, results[j][i]);
        }
    }
    println!("Took {} sec", secs);
}

// ----- Experiment 3 -----

fn start_experiment3(seed: u128, reps: Option<usize>, iter: Option<usize>, n_nodes: Option<usize>, shard_size: Option<usize>) {
    let n_nodes = n_nodes.unwrap_or(4_000);
    let shard_size = shard_size.unwrap_or(61);
    let reps = reps.unwrap_or(1_000);
    let iter = iter.unwrap_or(5_000);
    run_experiment3(n_nodes, shard_size, reps, iter, seed);
}

fn run_experiment3(n_nodes: usize, shard_size: usize, reps: usize, iter: usize, seed: u128) {
    let timer = Instant::now();
    let byz_frac = 4;
    let byz_tol = 2;
    let n_accounts = vec![n_nodes/4, n_nodes/3, n_nodes/2, n_nodes, n_nodes*2, n_nodes*3, n_nodes*4];
    // Use a fixed number of shards, regardless of the number of nodes.
    //let n_accounts = vec![4_000/4, 4_000/3, 4_000/2, 4_000, 4_000*2, 4_000*3, 4_000*4];
    let params = Params {
        n_nodes,
        n_accounts: 0,  // ignored
        byz_frac,
        byz_tol,
        shard_size,
        reps,
        iter,
        seed,
    };
    println!("experiment_3({}, {}, {}, {}, {}, {}, {:?})...", n_nodes, byz_frac, shard_size, reps, iter, seed, n_accounts);
    let results = experiment3(&params, n_accounts.clone());
    let secs = timer.elapsed().as_secs();
    for i in 0..params.reps {
        for (j,m) in n_accounts.iter().enumerate() {
            println!("N={}, F=1/{}, f=1/{}, m={}, r={}, fails={:?}", n_nodes, byz_frac, byz_tol, m, shard_size, results[j][i]);
        }
    }
    println!("Took {} sec", secs);
}

// ----- Supplemental -----

fn start_experiment3_sup(seed: u128, reps: Option<usize>, iter: Option<usize>, n_nodes: Option<usize>, shard_size: Option<usize>) {
    let n_nodes = n_nodes.unwrap_or(4_000);
    let shard_size = shard_size.unwrap_or(61);
    let reps = reps.unwrap_or(200);
    let iter = iter.unwrap_or(5_000);
    run_experiment3_sup(n_nodes, shard_size, reps, iter, seed);
}

fn run_experiment3_sup(n_nodes: usize, shard_size: usize, reps: usize, iter: usize, seed: u128) {
    let timer = Instant::now();
    let byz_frac = 4;
    let byz_tol = 2;
    let n_accounts = vec![125, 250, 333, 400, 500, 666, 800, 10000, 12000, 14000, 16000, 20000, 24000, 28000, 32000];
    let params = Params {
        n_nodes,
        n_accounts: 0,  // ignored
        byz_frac,
        byz_tol,
        shard_size,
        reps,
        iter,
        seed,
    };
    println!("experiment_3_sup({}, {}, {}, {}, {}, {}, {:?})...", n_nodes, byz_frac, shard_size, reps, iter, seed, n_accounts);
    let results = experiment3(&params, n_accounts.clone());
    let secs = timer.elapsed().as_secs();
    for i in 0..params.reps {
        for (j,m) in n_accounts.iter().enumerate() {
            println!("N={}, F=1/{}, f=1/{}, m={}, r={}, fails={:?}", n_nodes, byz_frac, byz_tol, m, shard_size, results[j][i]);
        }
    }
    println!("Took {} sec", secs);
}
