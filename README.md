# ScaleGraph sharding experiments

This repo contains the code used to run the sharding simulation experiments and
generate the results for the ScaleGraph paper. The generated raw results (and
CSV versions) are also included.


## Recreating the results

The parameters used in the experiments are the defaults.
Only the seed and the experiment must be specified.
To re-run the experiments with matching parameters:

```
$ cargo run -- --seed 1 --experiment 1
$ cargo run -- --seed 1 --experiment 21     # 2a
$ cargo run -- --seed 1 --experiment 22     # 2a
$ cargo run -- --seed 1 --experiment 31     # 3a
$ cargo run -- --seed 1 --experiment 32     # 3b
$ cargo run -- --seed 1 --experiment 33     # 3c
$ cargo run -- --seed 1 --experiment 34     # 3d
```

(Note that some results (for certain parameter combinations) may appear in a
different order. But the actual results should be identical to those in the CSV
files.)


## Testing

A straightforward and simple implementation is extremely slow and limits the
scale at which experiments can be run. This version has been optimized to be at
least an order of magnitude faster. To make sure this implementation is not
"too clever", each experiment has at least 2 implementations; one simple and
one optimized. Multiple test cases check that the implementations produce the
same results (for smaller-scale runs).

The same principle is applied to lower-level building blocks, such as the
implementation of the `U256` (i.e. 256-bit unsigned) type.
In addition, sanity checks are made to check cases with known results (such
as scenarios where finding compromised shards is impossible/inevitable).


## TODO

- Add the code for analyzing the data and generating the graphs.
