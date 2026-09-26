#!/bin/bash
# Runs every experiment except E18's learning curves (see run_e18.sh).
cd "$(dirname "$0")"
export PYTHONPATH=.
for env in random_oldp pointing qubit_prep; do
  python3 experiments/e18_transfer.py --env $env >> results/e18_transfer_log.txt 2>&1
done
python3 experiments/e17_exact_checks.py   > results/e17.txt 2>&1
python3 experiments/e17b_extension_checks.py > results/e17b.txt 2>&1
python3 experiments/e09_commutative_reduction.py > results/e09.txt 2>&1
python3 experiments/e12_monotonicity.py  > results/e12.txt 2>&1
python3 experiments/e10_depth_profile.py > results/e10.txt 2>&1
python3 experiments/e07_lse_lipschitz.py > results/e07.txt 2>&1
for env in orth5 so3 se3 boson; do
  python3 experiments/rep_benchmark.py --env $env > results/rep_$env.txt 2>&1
done
echo ALLDONE > results/rest_done.txt
