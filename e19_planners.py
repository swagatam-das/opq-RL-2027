"""E19b: planners that know exactly what OpQ knows.

OpQ knows the action operators K_a and learns R(m, a) from scalar rewards and P(m'|m, a) from
samples.  Here a model-based agent collects the same statistics from n uniformly random episodes
(initial latent fixed, as in the 'fixed' regime of E18) and then PLANS with them:

  exhaustive : exact expectimax over all K^H action sequences (optimal for the model);
  mcts-S     : UCT with S simulations per decision and uniform-random rollouts.

Both are evaluated at the training latent and, without any retraining, on the 8 held-out latents
used in E18.  Exhaustive planning is evaluated exactly (all mode branches); MCTS by Monte Carlo
over mode paths with common random numbers.  Also records the cost per decision.

Usage: python experiments/e19_planners.py --env pointing --seeds 5
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from opq.envs import ENVS
from opq.planners import (LearnedModel, ExhaustivePolicy, MCTSPolicy, evaluate_mc, decision_time)


def collect(env, model, n_ep, rng):
    for _ in range(n_ep):
        t, m, rho = env.reset(rng)
        done = False
        while not done:
            a = int(rng.integers(env.K))
            r, t2, m2, rho2, done = env.step(t, m, rho, a, rng)
            model.update(m, a, rho2, r, m2)
            t, m, rho = t2, m2, rho2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="pointing")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--budgets", default="25,50,100,200,400", help="episodes of model-learning data")
    ap.add_argument("--sims", default="100,1000", help="MCTS simulations per decision")
    ap.add_argument("--paths", type=int, default=16, help="mode paths per latent for MCTS evaluation")
    ap.add_argument("--n_test", type=int, default=8, help="held-out latents")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results", "e19"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    env = ENVS[args.env](seed=0)
    rng0 = np.random.default_rng(999)
    test = [env.sample_rho0(rng0) for _ in range(args.n_test)]
    latents = [env.rho0] + test                       # index 0 = training latent
    vstar = [env.optimal_values(r)[1] for r in latents]
    vrand = [env.random_value(r) for r in latents]
    sims = [int(s) for s in args.sims.split(",")]
    budgets = [int(b) for b in args.budgets.split(",")]
    Rtrue = np.array([[env.R[m][a] for a in range(env.K)] for m in range(env.nM)])
    print(f"[{args.env}] H={env.H} K={env.K} V*(train)={vstar[0]:.4f}", flush=True)

    def score(vals):                                   # vals: per-latent expected returns
        s = [(v - r) / (a - r) for v, a, r in zip(vals, vstar, vrand)]
        return float(s[0]), float(np.mean(s[1:]))

    res = {"env": args.env, "H": env.H, "K": env.K, "vstar": vstar, "vrand": vrand,
           "budgets": budgets, "sims": sims, "rows": [], "cost": {}}

    # -- exact-model references -------------------------------------------------------
    exact_pol = ExhaustivePolicy(env)
    t0 = time.time(); exact_pol(0, 0, env.rho0, ()); res["cost"]["exhaustive_s"] = time.time() - t0
    res["cost"]["exhaustive_nodes"] = int(sum(env.K ** d for d in range(1, env.H + 1)))
    for S in sims:
        pols = MCTSPolicy(env, Rtrue, env.P, S, seed=1)
        vals = [evaluate_mc(env, pols, r, args.paths, seed=7).mean() for r in latents]
        sc = score(vals)
        res["rows"].append({"method": f"mcts-{S}", "model": "exact", "n_ep": None, "seed": 0,
                            "score_train": sc[0], "score_heldout": sc[1], "values": [float(v) for v in vals]})
        res["cost"][f"mcts-{S}_s"] = decision_time(pols, env, env.rho0, reps=3)
        print(f"   exact-model MCTS-{S}: train {sc[0]:.3f}  held-out {sc[1]:.3f}  "
              f"{res['cost'][f'mcts-{S}_s'] * 1e3:.1f} ms/decision", flush=True)
    print(f"   exact-model exhaustive: score 1.000 by construction ({res['cost']['exhaustive_s']:.2f} s/decision, "
          f"{res['cost']['exhaustive_nodes']:,} nodes)", flush=True)

    # -- learned models ----------------------------------------------------------------
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed + 12345)
        model, used = LearnedModel(env), 0
        for n_ep in budgets:
            collect(env, model, n_ep - used, rng)
            used = n_ep
            Rh, Ph = model.R(), model.P()
            pol = ExhaustivePolicy(env, Rh, Ph)
            vals = [env.evaluate(pol, r) for r in latents]
            sc = score(vals)
            res["rows"].append({"method": "exhaustive", "model": "learned", "n_ep": n_ep, "seed": seed,
                                "score_train": sc[0], "score_heldout": sc[1], "values": vals})
            line = f"   seed {seed} n={n_ep:4d}  exhaustive {sc[0]:.3f}/{sc[1]:.3f}"
            for S in sims:
                pm = MCTSPolicy(env, Rh, Ph, S, seed=seed)
                vals = [evaluate_mc(env, pm, r, args.paths, seed=7).mean() for r in latents]
                sc = score(vals)
                res["rows"].append({"method": f"mcts-{S}", "model": "learned", "n_ep": n_ep, "seed": seed,
                                    "score_train": sc[0], "score_heldout": sc[1], "values": [float(v) for v in vals]})
                line += f"  mcts-{S} {sc[0]:.3f}/{sc[1]:.3f}"
            print(line + "   (train/held-out)", flush=True)
        json.dump(res, open(os.path.join(args.out, f"planners_{args.env}.json"), "w"))
    res["complete"] = True
    json.dump(res, open(os.path.join(args.out, f"planners_{args.env}.json"), "w"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
