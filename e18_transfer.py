"""E18b: zero-shot transfer to unseen initial latents.

Agents are trained with ONE fixed initial latent rho_0 (as in the 'fixed' regime of
e18_control.py) and their greedy policies are then evaluated exactly on 8 held-out
random initial latents (the same 8 as in the 'random' regime).  Thm. 1 makes the
operator readout tau(rho Q(s, a)) valid for every rho, so the operator agents need
no retraining; the scalar agents must extrapolate.
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from opq.envs import ENVS
from opq.train import train
from experiments.e18_control import AGENTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="pointing")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--agents", default="opq_soft,opq_pi,dqn,drqn")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results", "e18"))
    args = ap.parse_args()
    env = ENVS[args.env](seed=0)
    rng = np.random.default_rng(999)
    test = [env.sample_rho0(rng) for _ in range(8)]
    vs = [env.optimal_values(r)[1] for r in test]
    vr = [env.random_value(r) for r in test]
    vs0, vr0 = env.optimal_values()[1], env.random_value()
    res = {"env": args.env, "runs": {}}
    for name in args.agents.split(","):
        res["runs"][name] = []
        for s in range(args.seeds):
            ag = AGENTS[name](env, s)
            train(env, ag, args.episodes, lambda p: 0.0, [args.episodes], seed=s)
            pol = ag.policy()
            ins = (env.evaluate(pol) - vr0) / (vs0 - vr0)
            tr = float(np.mean([(env.evaluate(pol, r) - b) / (a - b) for r, a, b in zip(test, vs, vr)]))
            res["runs"][name].append({"seed": s, "in_dist": ins, "transfer": tr})
            print(f"   {name:10s} seed {s}: training rho_0 {ins:.3f}   held-out rho_0 {tr:.3f}", flush=True)
        a = np.array([[r["in_dist"], r["transfer"]] for r in res["runs"][name]])
        print(f"== {args.env} {name:10s} training rho_0 {a[:, 0].mean():.3f} +- {a[:, 0].std():.3f}   "
              f"held-out {a[:, 1].mean():.3f} +- {a[:, 1].std():.3f}", flush=True)
    os.makedirs(args.out, exist_ok=True)
    json.dump(res, open(os.path.join(args.out, f"{args.env}_transfer.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
