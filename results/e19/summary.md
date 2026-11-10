## 2×2: value function (linear / nonlinear in ρ) × operators K_a (unknown / known)
Normalised return of the greedy policy (exact evaluation), mean±sd over seeds, after 100 / 3000 episodes.

### C1 random OLDP
| agent | class | K_a | fixed @100 | fixed @3000 | random @100 | random @3000 |
|---|---|---|---|---|---|---|
| **DQN on ρ_t** | nonlinear | unknown | 0.96±0.02 | 1.00±0.00 | 0.89±0.03 | 0.99±0.00 |
| **DQN + K_a-generated transitions** | nonlinear | known | 0.91±0.08 | 1.00±0.00 | 0.93±0.03 | 0.99±0.00 |
| **Linear-in-ρ Q-learning (soft)** | linear | unknown | 0.86±0.08 | 0.99±0.00 | 0.77±0.06 | 0.96±0.02 |
| **OpQ-soft (β=5)** | linear | known | 0.77±0.06 | 0.84±0.01 | 0.95±0.01 | 0.95±0.00 |
| Linear-in-ρ Q-learning (max) | | | 0.86±0.08 | 0.99±0.00 | 0.77±0.05 | 0.97±0.01 |
| DQN, 4 updates/step | | | 0.92±0.03 | 1.00±0.00 | 0.89±0.04 | 0.99±0.00 |
| DQN + K_a transitions, 4 updates/step | | | 0.96±0.02 | 1.00±0.00 | 0.96±0.02 | 1.00±0.00 |
| DQN + K_a transitions, true R | | | 0.96±0.04 | 1.00±0.00 | 0.94±0.02 | 0.99±0.00 |
| DQN + K_a transitions, on-support ρ only | | | 0.96±0.03 | 1.00±0.00 | 0.94±0.02 | 1.00±0.00 |
| Learned (R,P) + exhaustive planning | | | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 |

### C2 SO(3) pointing
| agent | class | K_a | fixed @100 | fixed @3000 | random @100 | random @3000 |
|---|---|---|---|---|---|---|
| **DQN on ρ_t** | nonlinear | unknown | 0.85±0.06 | 0.98±0.01 | 0.73±0.04 | 0.99±0.00 |
| **DQN + K_a-generated transitions** | nonlinear | known | 0.81±0.03 | 0.98±0.01 | 0.82±0.03 | 0.99±0.00 |
| **Linear-in-ρ Q-learning (soft)** | linear | unknown | 0.73±0.06 | 0.95±0.02 | 0.57±0.04 | 0.95±0.01 |
| **OpQ-soft (β=5)** | linear | known | 0.94±0.02 | 0.98±0.00 | 0.95±0.01 | 0.99±0.00 |
| Linear-in-ρ Q-learning (max) | | | 0.75±0.06 | 0.97±0.00 | 0.62±0.04 | 0.96±0.01 |
| DQN, 4 updates/step | | | 0.79±0.11 | 0.99±0.01 | 0.82±0.04 | 0.99±0.00 |
| DQN + K_a transitions, 4 updates/step | | | 0.93±0.02 | 0.99±0.00 | 0.90±0.03 | 1.00±0.00 |
| DQN + K_a transitions, true R | | | 0.84±0.05 | 0.99±0.01 | 0.81±0.01 | 0.99±0.00 |
| DQN + K_a transitions, on-support ρ only | | | 0.88±0.04 | 0.98±0.01 | 0.82±0.04 | 0.99±0.00 |
| Learned (R,P) + exhaustive planning | | | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 |

### C3 two-qubit prep.
| agent | class | K_a | fixed @100 | fixed @3000 | random @100 | random @3000 |
|---|---|---|---|---|---|---|
| **DQN on ρ_t** | nonlinear | unknown | -0.08±0.16 | 1.00±0.00 | 0.04±0.12 | 0.87±0.02 |
| **DQN + K_a-generated transitions** | nonlinear | known | 0.26±0.33 | 1.00±0.00 | 0.22±0.15 | 0.86±0.02 |
| **Linear-in-ρ Q-learning (soft)** | linear | unknown | 0.21±0.14 | 0.28±0.00 | 0.25±0.10 | 0.54±0.08 |
| **OpQ-soft (β=5)** | linear | known | 0.38±0.20 | 0.83±0.08 | 0.58±0.07 | 0.68±0.00 |
| Linear-in-ρ Q-learning (max) | | | 0.28±0.00 | 0.46±0.27 | 0.29±0.07 | 0.53±0.07 |
| DQN, 4 updates/step | | | 0.17±0.38 | 1.00±0.00 | 0.04±0.11 | 0.87±0.03 |
| DQN + K_a transitions, 4 updates/step | | | 0.36±0.22 | 1.00±0.00 | 0.38±0.09 | 0.90±0.01 |
| DQN + K_a transitions, true R | | | 0.24±0.32 | 1.00±0.00 | 0.19±0.14 | 0.89±0.02 |
| DQN + K_a transitions, on-support ρ only | | | 0.27±0.29 | 1.00±0.00 | 0.19±0.14 | 0.87±0.03 |
| Learned (R,P) + exhaustive planning | | | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 |

## Zero-shot transfer: trained at one latent, evaluated at 8 held-out latents (normalised return)
| env | agent | training ρ_0 | held-out ρ_0 |
|---|---|---|---|
| C1 random OLDP | opq_soft | 0.84±0.01 | 0.95±0.00 |
| C1 random OLDP | dqn | 1.00±0.00 | 0.86±0.04 |
| C1 random OLDP | lin_q | 0.99±0.00 | 0.75±0.03 |
| C1 random OLDP | dqn_aug | 1.00±0.00 | 0.99±0.00 |
| C1 random OLDP | dqn_aug_reach | 1.00±0.00 | 0.88±0.02 |
| C1 random OLDP | model_exh | 1.00±0.00 | 1.00±0.00 |
| C2 SO(3) pointing | opq_soft | 0.98±0.00 | 0.99±0.00 |
| C2 SO(3) pointing | dqn | 0.98±0.01 | 0.96±0.02 |
| C2 SO(3) pointing | lin_q | 0.95±0.02 | 0.78±0.02 |
| C2 SO(3) pointing | dqn_aug | 0.98±0.01 | 0.98±0.00 |
| C2 SO(3) pointing | dqn_aug_reach | 0.98±0.01 | 0.96±0.01 |
| C2 SO(3) pointing | model_exh | 1.00±0.00 | 1.00±0.00 |
| C3 two-qubit prep. | opq_soft | 0.83±0.08 | 0.68±0.01 |
| C3 two-qubit prep. | dqn | 1.00±0.00 | 0.46±0.11 |
| C3 two-qubit prep. | lin_q | 0.28±0.00 | 0.37±0.05 |
| C3 two-qubit prep. | dqn_aug | 1.00±0.00 | 0.84±0.03 |
| C3 two-qubit prep. | dqn_aug_reach | 1.00±0.00 | 0.59±0.04 |
| C3 two-qubit prep. | model_exh | 1.00±0.00 | 1.00±0.00 |

## Planners with the same knowledge as OpQ (K_a known, R and P learned from n random episodes)
Normalised return at the training latent / mean over 8 held-out latents; mean over seeds. MCTS-S = S simulations per decision.

### C1 random OLDP  (H=6, K=4; exhaustive search = 5,460 nodes, 0.00 s per decision at t=0)
| model data (episodes) | exhaustive | mcts-100 | mcts-1000 |
|---|---|---|---|
| 25 | 0.987 / 0.967 | 0.871 / 0.888 | 0.915 / 0.936 |
| 50 | 0.998 / 0.995 | 0.875 / 0.929 | 0.933 / 0.964 |
| 100 | 0.999 / 0.996 | 0.890 / 0.929 | 0.931 / 0.961 |
| 200 | 1.000 / 0.998 | 0.875 / 0.923 | 0.924 / 0.960 |
| 400 | 1.000 / 0.998 | 0.892 / 0.926 | 0.935 / 0.962 |
| exact model | 1.000 / 1.000 | 0.909 / 0.920 | 0.939 / 0.965 |

Cost per decision (s): exhaustive=0.00433, mcts-100=0.0202, mcts-1000=0.241

### C2 SO(3) pointing  (H=8, K=6; exhaustive search = 2,015,538 nodes, 1.60 s per decision at t=0)
| model data (episodes) | exhaustive | mcts-100 | mcts-1000 |
|---|---|---|---|
| 25 | 0.994 / 0.999 | 0.950 / 0.943 | 0.986 / 0.980 |
| 50 | 0.998 / 1.000 | 0.947 / 0.943 | 0.985 / 0.986 |
| 100 | 1.000 / 1.000 | 0.952 / 0.949 | 0.988 / 0.986 |
| 200 | 1.000 / 1.000 | 0.937 / 0.947 | 0.981 / 0.985 |
| 400 | 1.000 / 1.000 | 0.949 / 0.949 | 0.985 / 0.988 |
| exact model | 1.000 / 1.000 | 0.952 / 0.938 | 0.991 / 0.990 |

Cost per decision (s): exhaustive=1.6, mcts-100=0.0213, mcts-1000=0.241

### C3 two-qubit prep.  (H=8, K=5; exhaustive search = 488,280 nodes, 1.43 s per decision at t=0)
| model data (episodes) | exhaustive | mcts-100 | mcts-1000 |
|---|---|---|---|
| 25 | 0.832 / 0.879 | 0.363 / 0.489 | 0.566 / 0.654 |
| 50 | 0.958 / 0.942 | 0.570 / 0.556 | 0.587 / 0.728 |
| 100 | 1.000 / 0.997 | 0.384 / 0.613 | 0.832 / 0.733 |
| 200 | 1.000 / 1.000 | 0.629 / 0.690 | 0.832 / 0.757 |
| 400 | 1.000 / 1.000 | 0.486 / 0.643 | 0.731 / 0.765 |
| exact model | 1.000 / 1.000 | 0.283 / 0.648 | 0.790 / 0.812 |

Cost per decision (s): exhaustive=1.43, mcts-100=0.0232, mcts-1000=0.264

## Longer horizons
Value = expected discounted return (Monte Carlo over mode paths, common random numbers), mean over seeds. `score` = (V − V_rand)/(V_ref − V_rand) with V_ref = exact V* when it is computed (for two-qubit preparation V* = 1 analytically) and otherwise the best value found by any method (a lower bound on V*). `cert` = operator-LSE certificate / V_ref (an upper bound on V*, so ≥ 1). ms = wall-clock per decision at t=0. LSE-spread = max β·eigenvalue spread seen by OpQ (the double-precision LSE is reliable below about 25).

### C1 random OLDP
| H | exact V* | V_ref | cert/V_ref | dqn | dqn_aug | exhaustive_exact_model | exhaustive_learned_model | lin_q | mcts-1000_exact_model | mcts-1000_learned_model | mcts-100_exact_model | mcts-100_learned_model | opq_soft |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 6 | yes | 4.169 | 1.88 | 0.962 | 0.961 | 1.000 | 1.000 | 0.957 | 0.952 | 0.943 | 0.916 | 0.898 | 0.810 |
| 8 | yes | 5.254 | 1.95 | 0.944 | 0.955 | 1.000 | 1.000 | 0.942 | 0.965 | 0.963 | 0.876 | 0.819 | 0.809 |
| 10 | yes | 6.469 | 1.93 | 0.960 | 0.961 | 1.000 | 1.000 | 0.935 | 0.961 | 0.953 | 0.779 | 0.849 | 0.810 |
| 12 | no | 7.151 | 2.02 | 0.999 | 1.000 | – | – | 0.964 | 0.953 | 0.985 | 0.868 | 0.878 | 0.847 |

Zero-shot transfer (trained at the fixed latent; score on 2 held-out latents, same normalisation):
| H | dqn | dqn_aug | exhaustive_exact_model | exhaustive_learned_model | lin_q | mcts-1000_exact_model | mcts-1000_learned_model | mcts-100_exact_model | mcts-100_learned_model | opq_soft |
|---|---|---|---|---|---|---|---|---|---|---|
| 6 | 0.823 | 0.998 | 1.000 | 1.000 | 0.728 | 0.965 | 0.966 | 0.945 | 0.931 | 0.978 |
| 8 | 0.862 | 0.998 | 1.000 | 1.000 | 0.897 | 0.983 | 0.992 | 0.957 | 0.951 | 0.988 |
| 10 | 0.864 | 0.989 | 1.000 | 0.999 | 0.864 | 0.968 | 0.980 | 0.898 | 0.927 | 0.976 |
| 12 | 0.904 | 1.000 | – | – | 0.866 | 0.977 | 0.963 | 0.900 | 0.926 | 0.985 |

Decision time (ms) and OpQ LSE-spread:
| H | dqn | dqn_aug | exhaustive_exact_model | exhaustive_learned_model | lin_q | mcts-1000_exact_model | mcts-1000_learned_model | mcts-100_exact_model | mcts-100_learned_model | opq_soft | LSE-spread |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 6 | 0.1 | 0.11 | 4.3 | – | 0.039 | 2.3e+02 | 2.4e+02 | 21 | 21 | 0.026 | 22 |
| 8 | 0.11 | 0.12 | 58 | – | 0.038 | 2.6e+02 | 2.6e+02 | 22 | 23 | 0.025 | 22 |
| 10 | 0.11 | 0.11 | 9.4e+02 | – | 0.038 | 3e+02 | 3e+02 | 25 | 27 | 0.024 | 22 |
| 12 | 0.12 | 0.13 | – | – | 0.04 | 3.4e+02 | 3.4e+02 | 30 | 32 | 0.03 | 22 |

Exhaustive planning needs H=6: 5.46e+03, H=8: 8.74e+04, H=10: 1.40e+06, H=12: 2.24e+07 node expansions per decision.

### C2 SO(3) pointing
| H | exact V* | V_ref | cert/V_ref | dqn | dqn_aug | exhaustive_exact_model | exhaustive_learned_model | lin_q | mcts-1000_exact_model | mcts-1000_learned_model | mcts-100_exact_model | mcts-100_learned_model | opq_soft |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | yes | 6.200 | 1.32 | 0.977 | 0.981 | 1.000 | 1.000 | 0.963 | 0.996 | 0.980 | 0.946 | 0.944 | 0.980 |
| 10 | no | 7.380 | 1.33 | 0.997 | 1.000 | – | – | 0.945 | 0.974 | 0.979 | 0.950 | 0.950 | 0.991 |
| 12 | no | 8.489 | 1.33 | 0.990 | 1.000 | – | – | 0.939 | 0.971 | 0.966 | 0.894 | 0.927 | 0.986 |
| 16 | no | 10.398 | 1.33 | 1.000 | 0.981 | – | – | 0.884 | 0.950 | 0.957 | 0.907 | 0.920 | 0.984 |
| 20 | no | 11.954 | 1.33 | 1.000 | 0.996 | – | – | 0.909 | 0.960 | 0.958 | 0.890 | 0.910 | 0.976 |

Zero-shot transfer (trained at the fixed latent; score on 2 held-out latents, same normalisation):
| H | dqn | dqn_aug | exhaustive_exact_model | exhaustive_learned_model | lin_q | mcts-1000_exact_model | mcts-1000_learned_model | mcts-100_exact_model | mcts-100_learned_model | opq_soft |
|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 0.976 | 0.998 | 1.000 | 1.000 | 0.818 | 0.995 | 0.993 | 0.948 | 0.954 | 1.002 |
| 10 | 0.979 | 0.988 | – | – | 0.826 | 0.987 | 0.987 | 0.938 | 0.949 | 1.000 |
| 12 | 0.975 | 0.995 | – | – | 0.832 | 0.987 | 0.987 | 0.933 | 0.942 | 1.000 |
| 16 | 0.991 | 1.000 | – | – | 0.808 | 0.995 | 0.990 | 0.921 | 0.919 | 0.996 |
| 20 | 0.993 | 1.000 | – | – | 0.854 | 0.987 | 0.979 | 0.909 | 0.910 | 0.975 |

Decision time (ms) and OpQ LSE-spread:
| H | dqn | dqn_aug | exhaustive_exact_model | exhaustive_learned_model | lin_q | mcts-1000_exact_model | mcts-1000_learned_model | mcts-100_exact_model | mcts-100_learned_model | opq_soft | LSE-spread |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 0.11 | 0.11 | 1.5e+03 | – | 0.04 | 2.4e+02 | 2.5e+02 | 21 | 22 | 0.034 | 7 |
| 10 | 0.12 | 0.11 | – | – | 0.039 | 2.7e+02 | 2.8e+02 | 24 | 25 | 0.034 | 7 |
| 12 | 0.12 | 0.12 | – | – | 0.038 | 3e+02 | 3e+02 | 28 | 28 | 0.036 | 8 |
| 16 | 0.11 | 0.11 | – | – | 0.04 | 3.8e+02 | 3.7e+02 | 38 | 36 | 0.035 | 10 |
| 20 | 0.11 | 0.11 | – | – | 0.037 | 4.4e+02 | 4.2e+02 | 48 | 42 | 0.035 | 12 |

Exhaustive planning needs H=8: 2.02e+06, H=10: 7.26e+07, H=12: 2.61e+09, H=16: 3.39e+12, H=20: 4.39e+15 node expansions per decision.

### C3 two-qubit prep.
| H | exact V* | V_ref | cert/V_ref | dqn | dqn_aug | exhaustive_exact_model | exhaustive_learned_model | lin_q | mcts-1000_exact_model | mcts-1000_learned_model | mcts-100_exact_model | mcts-100_learned_model | opq_soft |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | yes | 1.000 | 3.02 | 1.000 | 1.000 | 1.000 | 1.000 | 0.283 | 0.790 | 0.860 | 0.283 | 0.452 | 0.860 |
| 10 | no | 1.000 | 3.67 | 1.000 | 0.942 | – | – | 0.606 | 0.606 | 0.606 | 0.606 | 0.606 | 0.606 |
| 12 | no | 1.000 | 4.30 | 1.000 | 1.000 | – | – | 0.611 | 1.000 | 0.930 | 0.611 | 0.860 | 0.731 |

Zero-shot transfer (trained at the fixed latent; score on 2 held-out latents, same normalisation):
| H | dqn | dqn_aug | exhaustive_exact_model | exhaustive_learned_model | lin_q | mcts-1000_exact_model | mcts-1000_learned_model | mcts-100_exact_model | mcts-100_learned_model | opq_soft |
|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 0.346 | 0.904 | 1.000 | 1.000 | 0.296 | 0.814 | 0.776 | 0.475 | 0.575 | 0.836 |
| 10 | 0.417 | 0.816 | – | – | 0.296 | 0.815 | 0.729 | 0.762 | 0.636 | 0.721 |
| 12 | 0.614 | 0.751 | – | – | 0.363 | 0.765 | 0.765 | 0.738 | 0.630 | 0.635 |

Decision time (ms) and OpQ LSE-spread:
| H | dqn | dqn_aug | exhaustive_exact_model | exhaustive_learned_model | lin_q | mcts-1000_exact_model | mcts-1000_learned_model | mcts-100_exact_model | mcts-100_learned_model | opq_soft | LSE-spread |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 0.17 | 0.17 | 1.4e+03 | – | 0.099 | 2.6e+02 | 2.6e+02 | 23 | 23 | 0.03 | 5 |
| 10 | 0.18 | 0.19 | – | – | 0.1 | 3e+02 | 3.5e+02 | 27 | 28 | 0.031 | 5 |
| 12 | 0.18 | 0.18 | – | – | 0.099 | 3.4e+02 | 3.3e+02 | 30 | 30 | 0.03 | 5 |

Exhaustive planning needs H=8: 4.88e+05, H=10: 1.22e+07, H=12: 3.05e+08 node expansions per decision.
