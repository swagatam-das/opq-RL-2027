#!/bin/bash
# E18: end-to-end control (all environments, both regimes, 5 seeds)
cd "$(dirname "$0")"
for env in random_oldp pointing qubit_prep; do
  for regime in fixed random; do
    PYTHONPATH=. python3 experiments/e18_control.py --env $env --regime $regime --seeds 5 --episodes 3000
  done
done
