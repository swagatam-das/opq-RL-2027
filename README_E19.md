# E19 — model-matched baselines, planners and longer horizons

OpQ is given the action operators K_a; E19 gives the baselines and planners the same knowledge, so that the effect of
the representation can be separated from the effect of knowing K_a (Sec. 6 and App. H of the paper).
**No GPU is needed**: everything runs on CPU (JAX CPU for the small MLPs).

## Run

```bash
pip install -r requirements.txt
python run_e19.py --smoke --jobs 2        # ~2.5 min on 2 cores; tiny end-to-end check, writes results/e19_smoke
python run_e19.py --jobs 4                # full suite, resumable; logs in results/e19/logs/
python summarize_e19.py                   # tables -> results/e19/summary.md
```

`python run_e19.py --dry` lists the 24 jobs. Re-running skips finished jobs. Stages can be run alone:
`--stages 2x2,transfer,planners,horizon`. Use `--jobs` ≈ number of physical cores; each process needs
up to ~1 GB (exhaustive search over K^H histories). Estimated total ≈ 8 CPU-hours
(≈ 1 h on 8 cores, ≈ 2 h on 4, ≈ 4 h on 2); measured on a 2-core machine: one seed of the heaviest
learning-curve job ≈ 3.5 min, one H=20 horizon seed ≈ 3 min.
On Windows use the Python launcher (not the `.sh` scripts); if JAX will not install natively, use WSL.

## What is new

| stage | script | what it adds |
|---|---|---|
| 2x2 | `experiments/e18_control.py` (new agents) | learning curves, 3 envs × fixed/random, 5 seeds |
| transfer | `experiments/e18_transfer.py --agents …` | zero-shot transfer for the new agents |
| planners | `experiments/e19_planners.py` | model-based planners vs. number of data episodes, cost per decision |
| horizon | `experiments/e19_horizon.py` | longer horizons, all methods, certificate, timing |

**The 2×2** (value function linear/nonlinear in ρ × operators unknown/known):

| | K_a unknown | K_a known |
|---|---|---|
| nonlinear in ρ | `dqn` (E18) | `dqn_aug`: DQN whose batches are half real transitions, half transitions generated with the known K_a (fresh ρ, ρ'=K_aρK_a†, reward τ(R̂ρ') from the same least-squares reward model OpQ uses; mode kernel resampled from real data) |
| linear in ρ | `lin_q`: Q = ⟨θ_{t,m,a}, φ(ρ)⟩ by TD from observed (ρ, a, r, ρ'), no K_a (soft backup, β=5; `lin_q_max` uses max) | `opq_soft` (E18) |

Controls for tuning objections: `dqn_u4`, `dqn_aug_u4` (4 gradient updates per step), `dqn_aug_knownR` (true R),
`dqn_aug_reach` (synthetic ρ only on the reachable set, no extra state coverage).

**Planners that know what OpQ knows** (`opq/planners.py`): K_a known; R(m,a) by least squares and P(m'|m,a)
by counts from n uniformly random episodes (the same statistics OpQ accumulates).
`exhaustive` = exact expectimax over all K^(H−t) sequences with that model (`model_exh` in the 2×2 stage, evaluated
exactly); `mcts-S` = UCT, S simulations per decision, random rollouts (Monte-Carlo evaluation, common random numbers).
With the true model exhaustive search equals V* by construction (checked against `env.optimal_values`).

**Longer horizons**: C1 H∈{6,8,10,12}, C2 H∈{8,10,12,16,20}, C3 H∈{8,10,12} (target depth = H, so V*=1).
Exact V* only while K^H histories fit in `--exact_bytes` (default 400 MB); otherwise the table uses the best value
found (lower bound on V*) and the OpQ certificate (upper bound). JSON also stores decision time and the
LSE conditioning diagnostic (β·eigenvalue spread; the double-precision LSE is documented as reliable below ≈ 25).

## Checks already done
* synthetic transitions of `dqn_aug` reproduce the environment exactly (float32 precision) on all three envs;
* exhaustive planner with the true model equals `optimal_values` exactly on all three envs;
* Monte-Carlo evaluation agrees with exact evaluation (e.g. 6.1614±0.0015 vs 6.1597);
* full pipeline `run_e19.py --smoke` passes for every stage.

The outputs of the runs reported in the paper are in `results/e19/` (`summary.md` collects the tables).
