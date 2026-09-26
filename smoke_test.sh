#!/bin/bash
# Quick installation check (~1 minute). Writes only to a temporary directory.
set -e
cd "$(dirname "$0")"
export PYTHONPATH=.
TMP=$(mktemp -d)
echo "== E9: commutative reduction"
python3 experiments/e09_commutative_reduction.py
echo "== E18 (short): SO(3) pointing, 100 episodes, 1 seed"
python3 experiments/e18_control.py --env pointing --regime fixed --seeds 1 --episodes 100 \
        --agents opq_soft,scalar_state,dqn --out "$TMP"
rm -rf "$TMP"
echo "== smoke test passed"
