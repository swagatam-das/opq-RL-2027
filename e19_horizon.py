"""E19c: longer horizons.

For one environment and one horizon H this script trains the learning agents on the fixed
initial latent (as in E18), builds the model-based planners from n uniformly random episodes
(E19b), and evaluates everything at the training latent and on the 8 held-out latents.

Exact optimal values need K^H action histories, so they are computed only while that fits in
memory.  Otherwise the record contains (i) the best value found by any method, a lower bound on V*,
and (ii) the operator log-sum-exp certificate (Thm 3), an upper bound on V*.  All policies are
evaluated by Monte Carlo over the observable mode chain with common random numbers.

Usage: python experiments/e19_horizon.py --env pointing --H 12 --seeds 3
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from opq.envs import ENVS
from opq.core import tau
from opq.agents import op_soft_exact, greedy_from_ops
from opq.planners import (LearnedModel, ExhaustivePolicy, MCTSPolicy, evaluate_mc, decision_time)
from opq.train import train
from experiments.e18_control import AGENTS
from experiments.e19_planners import collect


def make_env(name, H):
    if name == "qubit_prep":
        return ENVS[name](seed=0, H=H, depth=H)          # target needs H gates
    return ENVS[name](seed=0, H=H)


def lse_spread(agent, beta):
    """max over (t, m) of beta * (largest - smallest eigenvalue among Q(t, m, .)); the double-precision
    LSE is documented as reliable while this is below about 25."""
    if not hasattr(agent, "Q"):
        return None
    worst = 0.0
    for t in range(len(agent.Q) - 1):
        for m in range(len(agent.Q[t])):
            ev = np.concatenate([np.linalg.eigvalsh((q + q.conj().T) / 2) for q in agent.Q[t][m]])
            worst = max(worst, beta * (ev.max() - ev.min()))
    return float(worst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="pointing")
    ap.add_argument("--H", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--agents", default="opq_soft,dqn,dqn_aug,lin_q")
    ap.add_argument("--sims", default="100,1000")
    ap.add_argument("--model_eps", type=int, default=400)
    ap.add_argument("--paths", type=int, default=32, help="mode paths for learned agents")
    ap.add_argument("--mcts_paths", type=int, default=12, help="mode paths for MCTS")
    ap.add_argument("--mcts_latents", type=int, default=3, help="latents for MCTS (train + first held-out)")
    ap.add_argument("--exact_bytes", type=float, default=4e8, help="max memory for exact DP / exhaustive")
    ap.add_argument("--n_test", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results", "e19"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    env = make_env(args.env, args.H)
    rng0 = np.random.default_rng(999)
    latents = [env.rho0] + [env.sample_rho0(rng0) for _ in range(args.n_test)]
    item = 16 if env.is_complex else 8
    exact_ok = env.K ** env.H * env.N * env.N * item <= args.exact_bytes
    sims = [int(s) for s in args.sims.split(",")]
    Rtrue = np.array([[env.R[m][a] for a in range(env.K)] for m in range(env.nM)])
    tag = f"{args.env}_H{args.H}"
    print(f"[{tag}] K={env.K} N={env.N} nM={env.nM} gamma={env.gamma}  exact DP: {exact_ok}", flush=True)

    res = {"env": args.env, "H": env.H, "K": env.K, "gamma": env.gamma, "exact": exact_ok,
           "n_latents": len(latents), "methods": {}}
    t0 = time.time()
    res["vrand"] = [float(env.random_value(r)) for r in latents]
    Qc = {b: op_soft_exact(env, b) for b in (2.0, 5.0)}
    res["certificate"] = {str(b): [float(sum(env.p0[m] * max(tau(r @ q) for q in Q[0][m])
                                        for m in range(env.nM) if env.p0[m] > 0)) for r in latents]
                          for b, Q in Qc.items()}
    res["vstar"] = [float(env.optimal_values(r)[1]) for r in latents] if exact_ok else None
    # two-qubit preparation: the target is the output of a word of length H, so V* = 1 exactly
    res["v_analytic"] = 1.0 if args.env == "qubit_prep" else None
    res["exhaustive_nodes"] = int(sum(env.K ** d for d in range(1, env.H + 1)))
    print(f"   references ({time.time() - t0:.0f}s): V*={'exact' if exact_ok else 'n/a (K^H too large)'}, "
          f"certificate(beta=5, train)={res['certificate']['5.0'][0]:.3f}", flush=True)

    def record(name, seed, vals, extra=None):
        d = res["methods"].setdefault(name, [])
        d.append({"seed": seed, "values": [float(v) for v in vals], **(extra or {})})
        print(f"   {name:22s} seed {seed}: train {vals[0]:.3f}  held-out {np.mean(vals[1:]):.3f}"
              + (f"  [{extra}]" if extra else ""), flush=True)

    # -- planners with the exact model (no learning) ------------------------------------------
    for S in sims:
        pm = MCTSPolicy(env, Rtrue, env.P, S, seed=1)
        k = args.mcts_latents
        vals = [evaluate_mc(env, pm, r, args.mcts_paths, seed=7).mean() for r in latents[:k]]
        res["methods"][f"mcts-{S}_exact_model"] = [{"seed": 0, "values": [float(v) for v in vals],
                                                   "n_latents_evaluated": k,
                                                   "decision_s": decision_time(pm, env, env.rho0, reps=3)}]
        print(f"   mcts-{S} exact model: train {vals[0]:.3f}  held-out {np.mean(vals[1:]):.3f} "
              f"({res['methods'][f'mcts-{S}_exact_model'][0]['decision_s'] * 1e3:.0f} ms/decision)", flush=True)
    if exact_ok:
        ex = ExhaustivePolicy(env)
        t1 = time.time(); ex(0, 0, env.rho0, ()); dts = time.time() - t1
        res["methods"]["exhaustive_exact_model"] = [{"seed": 0, "values": res["vstar"], "decision_s": dts}]
        print(f"   exhaustive exact model: V* by construction, {dts:.2f} s/decision", flush=True)

    # -- learning agents ---------------------------------------------------------------------
    for name in [a for a in args.agents.split(",") if a]:
        for s in range(args.seeds):
            ag = AGENTS[name](env, s)
            _, tt = train(env, ag, args.episodes, lambda p: 0.0, [args.episodes], seed=s)
            pol = ag.policy()
            vals = [evaluate_mc(env, pol, r, args.paths, seed=7).mean() for r in latents]
            dec = decision_time(pol, env, env.rho0, reps=20)
            record(name, s, vals, {"train_s": round(tt, 1), "decision_s": dec,
                                   **({"lse_beta_spread": lse_spread(ag, 5.0)} if name == "opq_soft" else {})})
        json.dump(res, open(os.path.join(args.out, f"horizon_{tag}.json"), "w"))

    # -- planners with a learned model (same data budget for every seed) -------------------------
    for s in range(args.seeds):
        model = LearnedModel(env)
        collect(env, model, args.model_eps, np.random.default_rng(s + 12345))
        Rh, Ph = model.R(), model.P()
        for S in sims:
            pm = MCTSPolicy(env, Rh, Ph, S, seed=s)
            k = args.mcts_latents
            vals = [evaluate_mc(env, pm, r, args.mcts_paths, seed=7).mean() for r in latents[:k]]
            record(f"mcts-{S}_learned_model", s, vals, {"n_latents_evaluated": k,
                                                       "decision_s": decision_time(pm, env, env.rho0, reps=3)})
        if exact_ok:
            pol = ExhaustivePolicy(env, Rh, Ph)
            vals = [env.evaluate(pol, r) for r in latents]
            record("exhaustive_learned_model", s, vals)
        json.dump(res, open(os.path.join(args.out, f"horizon_{tag}.json"), "w"))
    res["wall_s"] = time.time() - t0
    json.dump(res, open(os.path.join(args.out, f"horizon_{tag}.json"), "w"))
    print(f"done ({res['wall_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
