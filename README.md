# Operator-Valued Q-Learning (OpQ)

Code accompanying the ICLR 2027 submission
*Operator-Valued Q-Learning: Compositional Bellman Recursions for Order-Sensitive Decision Making*.

Every number, table and figure in the paper is produced by the scripts in this
repository. The original suite (E7–E18) runs on a single CPU core in about three
hours; the experiments added in the revision (E20–E24) take about 100 minutes on
two cores, plus E19. No GPU is needed.

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

Experiments added in the revision:

```bash
python run_e19.py --jobs 4  # E19: model-matched baselines, planners, horizons (~8 CPU-hours; see README_E19.md)     (see note below)
bash run_rebuttal.sh      # E20-E24: tightening, miscalibration, beta, hidden latent, convergence (~100 min, 2 cores)
```

`NPROC` sets the number of worker processes for E22-E24 (default 2).

### Experiments

Theorem numbers refer to the revised paper.

| Paper | Script | What it tests |
|---|---|---|
| E17 | `experiments/e17_exact_checks.py` | Prop 1 (exact representation), Thm 1 (operator TD), Thm 2 (Löwner certificate), Prop 3 (commutative reduction), Prop 8 (aliasing), Prop 10 (Hankel rank) |
| E17b | `experiments/e17b_extension_checks.py` | extensions: robust improvement (Prop 11), noisy/random latent dynamics (Prop 12), structure-refined dimension (Prop 13) |
| E7  | `experiments/e07_lse_lipschitz.py` | Prop 6: Lipschitz behaviour of the operator log-sum-exp (80-digit arithmetic) |
| E9  | `experiments/e09_commutative_reduction.py` | Prop 3: exact reduction to soft Q-learning |
| E12 | `experiments/e12_monotonicity.py` | Prop 7: monotonicity holds only on the commuting sector |
| E10 | `experiments/e10_depth_profile.py` | Prop 9: nilpotent (n3) vs non-nilpotent (so(3)) control algebras |
| E8, E13–E16 | `experiments/rep_benchmark.py --env {orth5,so3,se3,boson}` | Prop 10 / Cor 1: operator dimension vs GRU/LSTM/attention/CNN baselines (supervised regression) |
| E18 | `experiments/e18_control.py --env {random_oldp,pointing,qubit_prep} --regime {fixed,random}` | end-to-end control |
| E18 (transfer) | `experiments/e18_transfer.py --env ...` | zero-shot transfer to unseen initial latents |
| E19 | `run_e19.py`, `experiments/e19_planners.py`, `experiments/e19_horizon.py` | 2×2 study (linear vs nonlinear × with/without the action maps), model-matched DQN, exhaustive search and UCT planners, horizons up to 20, decision cost |
| E20 | `experiments/e20_tightening.py` | Cor 2 (any Löwner upper bound certifies; SDP aggregator), Prop 4 (slack spread), M-operator point-based family, certificate as a leaf value |
| E21 | `experiments/e21_calibration.py [--lower-bound-only]` | miscalibrated action operators; M-operator lower bound L_M and cost |
| E22 | `experiments/e22_beta_sweep.py` | sensitivity of the sampled soft update to β; drift and late greedy-action switching |
| E23 | `experiments/e23_hidden_latent.py` | hidden initial latent: belief-mean agents, exact plug-in at the belief mean, blind and latent-aware references |
| E24 | `experiments/e24_convergence.py` | Thm 3: convergence of the sampled soft update to the finite-horizon certificate, two step-size rules |

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
| Fig. 6 (App. N, E24) | `Images/fig_e24_convergence.pdf` | `make_figures.py` (`e24`) from `results/e24/e24_results.json` |

## 4. Package layout

```
opq/core.py         normalised trace, operator log-sum-exp (float64 and mpmath), random operators
opq/envs.py         operator-latent decision processes (OLDPs): random OLDP, SO(3) pointing,
                    two-qubit state preparation; exact optimal values by dynamic programming;
                    exact policy evaluation
opq/agents.py       OpQ-soft, OpQ-PI, exact operator recursions (Eqs. 3-4), tabular scalar Q-learning
opq/agents_ext.py   E19: Linear-Q (linear in rho, no K_a) and the exhaustive model-based planner agent
opq/deep_agents.py  DQN (MLP on rho_t), DRQN (GRU on history) and, for E19, DQN trained with K_a-generated transitions, in JAX
opq/planners.py     E19: exhaustive expectimax and UCT planners on a model learned from random episodes
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
