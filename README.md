# Operator-Valued Q-Learning (OpQ)

Code accompanying the ICLR 2027 submission
*Operator-Valued Q-Learning: Compositional Bellman Recursions for Order-Sensitive Decision Making*.

Every number, table and figure in the paper is produced by the scripts in this
repository. The whole suite runs on a single CPU core in about three hours;
no GPU is needed.

## 1. Install

```bash
python -m venv .venv && source .venv/bin/activate      # optional
pip install -r requirements.txt
```

Tested with Python 3.12.3, NumPy 2.4.4, SciPy 1.17.1, mpmath 1.3.0,
Matplotlib 3.10.8, JAX 0.11.2 (CPU) and Optax 0.2.8. Everything runs on a
CPU; no GPU is needed.

## 2. Quick check (about 1 minute)

```bash
bash smoke_test.sh
```

This runs the commutative-reduction check (E9) and a 100-episode control run on
SO(3) pointing with three agents. It writes nothing into `results/`.

## 3. Reproduce the paper

```bash
bash run_e18.sh               # E18 learning curves: 3 envs x 2 regimes x 7 agents x 5 seeds  (~70 min)
bash run_rest.sh             # E18 transfer, E17, E9, E12, E10, E7, representation benchmark   (~110 min)
python make_figures.py   # figures -> Images/, LaTeX tables -> results/table_*.tex
```

The raw outputs of the runs reported in the paper are already in `results/`,
so `python make_figures.py` alone regenerates every figure and table.

### Experiments

| Paper | Script | What it tests |
|---|---|---|
| E17 | `experiments/e17_exact_checks.py` | Thm 1 (exact representation), Thm 2 (operator TD), Thm 3 (Löwner certificate), Prop 2, Thm 4 (aliasing), Thm 6 (Hankel rank) |
| E17b | `experiments/e17b_extension_checks.py` | Appendix E extensions: noisy/random latent dynamics (Prop 7), robust improvement (Prop 6), structure-refined dimension (Prop 8) |
| E7  | `experiments/e07_lse_lipschitz.py` | Prop 4: Lipschitz behaviour of the operator log-sum-exp (80-digit arithmetic) |
| E9  | `experiments/e09_commutative_reduction.py` | Prop 2: exact reduction to soft Q-learning |
| E12 | `experiments/e12_monotonicity.py` | Prop 5: monotonicity holds only on the commuting sector |
| E10 | `experiments/e10_depth_profile.py` | Thm 5: nilpotent (n3) vs non-nilpotent (so(3)) control algebras |
| E8, E13–E16 | `experiments/rep_benchmark.py --env {orth5,so3,se3,boson}` | Thm 6 / Cor 1: operator dimension vs GRU/LSTM/attention/CNN baselines |
| E18 | `experiments/e18_control.py --env {random_oldp,pointing,qubit_prep} --regime {fixed,random}` | end-to-end control |
| E18 (transfer) | `experiments/e18_transfer.py --env ...` | zero-shot transfer to unseen initial latents |

Scripts can be run from the repository root, e.g.
`python experiments/e18_control.py --env pointing --regime fixed --seeds 5`.

### Figures

| Paper | File | Produced by |
|---|---|---|
| Fig. 1 (overview) | `Images/fig1_overview.pdf` | `make_figures.py` (`overview`) |
| Fig. 2 (App. A, folding example) | `Images/origami_opq_compare.pdf` | `make_figures.py` (`origami`) |
| Fig. 3 (E18 learning curves) | `Images/fig_e18_control.pdf` | `make_figures.py` from `results/e18/*.json` |
| Fig. 4 (representation benchmark) | `Images/fig_rep_benchmark.pdf` | `make_figures.py` from `results/rep/*.json` |
| Fig. 5 (E10) | `Images/fig2_e10_horizon.pdf` | `make_figures.py` from `results/e10.json` |

## 4. Package layout

```
opq/core.py         normalised trace, operator log-sum-exp (float64 and mpmath), random operators
opq/envs.py         operator-latent decision processes (OLDPs): random OLDP, SO(3) pointing,
                    two-qubit state preparation; exact optimal values by dynamic programming;
                    exact policy evaluation
opq/agents.py       OpQ-soft, OpQ-PI, exact operator recursions (Eqs. 3-4), tabular scalar Q-learning
opq/deep_agents.py  DQN (MLP on rho_t) and DRQN (GRU on history) in JAX
opq/train.py        episodic training loop with exact evaluation checkpoints
opq/rep_models.py   operator model OpQ(d) and sequence baselines (GRU, LSTM, attention, CNN)
experiments/        one script per experiment (see table above)
results/            raw outputs of the runs reported in the paper (see results/README.md)
Images/             figures used in the paper
```

## 5. Minimal example

```python
from opq.envs import pointing
from opq.agents import OpQSoft
from opq.train import train

env = pointing(seed=0)
agent = OpQSoft(env, beta=5.0)                  # learns R(m, a) from scalar rewards
vstar = env.optimal_values()[1]                 # exact optimum by dynamic programming
curve, _ = train(env, agent, 1000, lambda pol: env.evaluate(pol) / vstar, [100, 1000])
print(curve)                                    # fraction of the optimal return
```

## License

MIT (see `LICENSE`).
