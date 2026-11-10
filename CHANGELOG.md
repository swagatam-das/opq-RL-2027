# Changelog

## v2 — revision (November 2026)

Code and results for the experiments added in the revised paper. The v1 modules of `opq/` are
unchanged (E19 only adds classes and modules), so every v1 number is reproduced exactly.

**New experiments**
- E19 — model-matched baselines: a 2×2 study (linear vs nonlinear value, with vs without the action
  maps), DQN trained on transitions generated with the action maps, exhaustive-search and UCT
  planners, horizons up to 20, decision cost. Code: `run_e19.py`, `summarize_e19.py`,
  `experiments/e19_planners.py`, `experiments/e19_horizon.py`, `opq/agents_ext.py`, `opq/planners.py`;
  new agents in `experiments/e18_control.py` (`lin_q`, `lin_q_max`, `dqn_u4`, `dqn_aug`, `dqn_aug_u4`,
  `dqn_aug_knownR`, `dqn_aug_reach`, `model_exh`) and `DQNAug` in `opq/deep_agents.py`;
  `--out` option in `experiments/e18_transfer.py`. See `README_E19.md`. Results: `results/e19/`.
- E20 — `experiments/e20_tightening.py`: any Löwner upper bound certifies (Cor. 2, SDP aggregator);
  slack spread (Prop. 4); M-operator point-based family; certificate as a leaf value.
- E21 — `experiments/e21_calibration.py`: miscalibrated action operators; M-operator lower bound and cost.
- E22 — `experiments/e22_beta_sweep.py`: sensitivity of the sampled soft update to β.
- E23 — `experiments/e23_hidden_latent.py`: hidden initial latent; belief-mean agents; exact plug-in.
- E24 — `experiments/e24_convergence.py`: numerical check of Thm. 3 (convergence of the sampled
  soft update to the finite-horizon certificate) under two step-size rules.

**Other changes**
- `run_rebuttal.sh` runs E20–E24.
- `make_figures.py`: Figure 1 labels follow the revised numbering (Prop. 1, 8, 9); new `e24()`
  writes `Images/fig_e24_convergence.pdf`.
- `requirements.txt`: adds `cvxpy` (SDP aggregator in E20).
- `README.md`, `results/README.md`: revised theorem numbering, new experiments and outputs.

## v1 — submission (September 2026)

E7–E18, representation benchmarks (E8, E13–E16), figures and tables of the submitted paper.
