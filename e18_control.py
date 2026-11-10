"""E18: end-to-end control on operator-latent decision processes.

For each environment and regime we train every agent from scratch on sampled
episodes and evaluate its greedy policy EXACTLY (all mode branches enumerated,
mean rewards) at log-spaced checkpoints.  Scores are normalised per initial
latent as (V_agent - V_random) / (V* - V_random), where V* is the exact
latent-aware optimum computed by dynamic programming over action histories.

Regimes:
  fixed   : one fixed initial latent rho_0
  random  : rho_0 ~ random pure state at every episode (observed by the
            latent-aware agents; DRQN receives it at step 0); evaluation
            on 8 held-out initial latents.

Usage: python experiments/e18_control.py --env pointing --regime fixed --seeds 5
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from opq.envs import ENVS
from opq.agents import OpQSoft, OpQPI, TabularQ, op_soft_exact, op_policy_eval_exact, greedy_from_ops
from opq.deep_agents import DQN, DRQN, DQNAug
from opq.agents_ext import LinearQ, ModelExh
from opq.train import train
from opq.core import tau

AGENTS = {
    "opq_soft": lambda env, s: OpQSoft(env, beta=5.0, seed=s),
    "opq_soft_knownR": lambda env, s: OpQSoft(env, beta=5.0, known_R=True, seed=s),
    "opq_pi": lambda env, s: OpQPI(env, seed=s),
    "scalar_state": lambda env, s: TabularQ(env, "state", seed=s),
    "scalar_history": lambda env, s: TabularQ(env, "history", seed=s),
    "dqn": lambda env, s: DQN(env, seed=s),
    "drqn": lambda env, s: DRQN(env, seed=s),
    # ---- E19: 2x2 comparison (nonlinear/linear in rho  x  operators unknown/known) and planners
    "lin_q": lambda env, s: LinearQ(env, soft=True, seed=s),
    "lin_q_max": lambda env, s: LinearQ(env, soft=False, seed=s),
    "dqn_u4": lambda env, s: DQN(env, updates_per_step=4, seed=s),
    "dqn_aug": lambda env, s: DQNAug(env, seed=s),
    "dqn_aug_u4": lambda env, s: DQNAug(env, updates_per_step=4, seed=s),
    "dqn_aug_knownR": lambda env, s: DQNAug(env, known_R=True, seed=s),
    "dqn_aug_reach": lambda env, s: DQNAug(env, aug_rho="reach", seed=s),
    "model_exh": lambda env, s: ModelExh(env, seed=s),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="pointing")
    ap.add_argument("--regime", default="fixed", choices=["fixed", "random"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--agents", default=",".join(AGENTS))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results", "e18"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    env = ENVS[args.env](seed=0)
    if args.regime == "random":
        env.random_init = True
        rng = np.random.default_rng(999)
        rho0s = [env.sample_rho0(rng) for _ in range(8)]
    else:
        rho0s = [env.rho0]
    t0 = time.time()
    vstar = [env.optimal_values(r)[1] for r in rho0s]
    vrand = [env.random_value(r) for r in rho0s]
    print(f"[{args.env}/{args.regime}] V* = {np.round(vstar, 4)}, V_rand = {np.round(vrand, 4)} "
          f"({time.time() - t0:.1f}s)", flush=True)

    def score(policy):
        v = [env.evaluate(policy, r) for r in rho0s]
        return float(np.mean([(vi - vr) / (vs - vr) for vi, vs, vr in zip(v, vstar, vrand)])), \
            float(np.mean(v))

    # exact-planning references (known model): soft-operator greedy and one-step PI
    ref = {}
    for b in [1.0, 2.0, 5.0, 10.0]:
        Qb = op_soft_exact(env, b)
        cert = float(np.mean([sum(env.p0[m] * max(tau(r @ q) for q in Qb[0][m]) for m in range(env.nM))
                              for r in rho0s]))
        ref[f"opq_soft_exact_beta{b:g}"] = {"score": score(greedy_from_ops(Qb)), "certificate": cert}
    Qpi = op_policy_eval_exact(env, lambda t, m: np.ones(env.K) / env.K)
    ref["opq_pi_exact"] = {"score": score(greedy_from_ops(Qpi))}
    for k, v in ref.items():
        print(f"   ref {k:28s} score={v['score'][0]:.3f} value={v['score'][1]:.3f}"
              + (f" certificate={v['certificate']:.3f}" if "certificate" in v else ""), flush=True)

    checkpoints = [c for c in [25, 50, 100, 200, 400, 800, 1600, 3000, 6000] if c <= args.episodes]
    if args.episodes not in checkpoints:
        checkpoints.append(args.episodes)
    res = {"env": args.env, "regime": args.regime, "vstar": vstar, "vrand": vrand,
           "checkpoints": checkpoints, "reference": ref, "runs": {}}
    fpath = os.path.join(args.out, f"{args.env}_{args.regime}.json")
    if os.path.exists(fpath):                       # resume: keep agents already finished
        res["runs"] = json.load(open(fpath))["runs"]
    for name in args.agents.split(","):
        res["runs"][name] = []
        for s in range(args.seeds):
            agent = AGENTS[name](env, s)
            curve, tt = train(env, agent, args.episodes, score, checkpoints, seed=s)
            res["runs"][name].append({"seed": s, "curve": curve, "time": tt})
            print(f"   {name:16s} seed {s}: final score {curve[-1][1][0]:.3f} "
                  f"value {curve[-1][1][1]:.3f} ({tt:.0f}s)", flush=True)
        fs = [r["curve"][-1][1][0] for r in res["runs"][name]]
        print(f"== {name:16s} final score {np.mean(fs):.3f} +- {np.std(fs):.3f}", flush=True)
        with open(fpath, "w") as f:
            json.dump(res, f)


if __name__ == "__main__":
    main()
