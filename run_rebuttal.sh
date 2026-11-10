#!/bin/bash
# Experiments added in the revision (E20-E24). Run after run_e18.sh / run_rest.sh, or alone.
# Approximate wall-clock times on 2 cores: E20 6 min, E21 30 min, E22 20 min, E23 20 min, E24 25 min.
# E19 (model-matched baselines and planners) is run by run_e19.py; see README section 3.
cd "$(dirname "$0")"
export PYTHONPATH=.
export NPROC=${NPROC:-2}
mkdir -p results/e20 results/e21 results/e22 results/e23 results/e24
python3 experiments/e20_tightening.py   > results/e20/e20_log.txt 2>&1
python3 experiments/e21_calibration.py  > results/e21/e21_log.txt 2>&1
python3 experiments/e22_beta_sweep.py   > results/e22/e22.log 2>&1
python3 experiments/e23_hidden_latent.py > results/e23/e23.log 2>&1
python3 experiments/e24_convergence.py  > results/e24/e24.log 2>&1
echo REBUTTAL_DONE > results/rebuttal_done.txt
