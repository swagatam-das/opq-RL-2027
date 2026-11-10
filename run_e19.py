"""Launcher for the E19 experiments (revision): 2x2 comparison, transfer, planners, horizons.

    python run_e19.py --jobs 4                 # everything, 4 processes in parallel, resumable
    python run_e19.py --stages 2x2,transfer    # a subset
    python run_e19.py --smoke                  # tiny end-to-end check (a few minutes), writes results/e19_smoke
    python run_e19.py --dry                    # print the commands only

Stages
  2x2       E18-style learning curves: DQN, DQN+known K_a (variants), Linear-Q, OpQ-soft, exhaustive planner
  transfer  zero-shot transfer to 8 held-out latents for the same agents
  planners  model-based exhaustive / MCTS planners: sample efficiency and cost per decision
  horizon   longer horizons (learning agents + planners + certificate)

Finished jobs are detected from their JSON output and skipped, so an interrupted run can be restarted.
No GPU is needed.  Logs: <out>/logs/<job>.log
"""
import argparse, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.abspath(__file__))
ENVS = ["random_oldp", "pointing", "qubit_prep"]
CURVE_AGENTS = ["opq_soft", "dqn", "lin_q", "lin_q_max", "dqn_aug", "dqn_aug_u4", "dqn_aug_knownR",
                "dqn_aug_reach", "dqn_u4", "model_exh"]
TRANSFER_AGENTS = ["opq_soft", "dqn", "lin_q", "dqn_aug", "dqn_aug_reach", "model_exh"]
HORIZONS = {"random_oldp": [6, 8, 10, 12], "pointing": [8, 10, 12, 16, 20], "qubit_prep": [8, 10, 12]}


def load(path):
    try:
        return json.load(open(path))
    except Exception:
        return None


def jobs(args):
    out, py = args.out, sys.executable
    J = []
    if "2x2" in args.stages:
        for env in ENVS:
            for reg in ("fixed", "random"):
                f = os.path.join(out, f"{env}_{reg}.json")
                d = load(f)
                have = {k for k, v in (d or {}).get("runs", {}).items() if len(v) >= args.seeds}
                miss = [a for a in CURVE_AGENTS if a not in have]
                if miss:
                    J.append((f"2x2_{env}_{reg}", 3, [py, "experiments/e18_control.py", "--env", env,
                              "--regime", reg, "--seeds", str(args.seeds), "--episodes", str(args.episodes),
                              "--agents", ",".join(miss), "--out", out]))
    if "transfer" in args.stages:
        for env in ENVS:
            d = load(os.path.join(out, f"{env}_transfer.json"))
            if not d or any(len(d["runs"].get(a, [])) < args.seeds for a in TRANSFER_AGENTS):
                J.append((f"transfer_{env}", 2, [py, "experiments/e18_transfer.py", "--env", env,
                          "--seeds", str(args.seeds), "--episodes", str(args.episodes),
                          "--agents", ",".join(TRANSFER_AGENTS), "--out", out]))
    if "planners" in args.stages:
        for env in ENVS:
            d = load(os.path.join(out, f"planners_{env}.json"))
            if not (d and d.get("complete")):
                J.append((f"planners_{env}", 2, [py, "experiments/e19_planners.py", "--env", env,
                          "--seeds", str(args.seeds), "--budgets", args.budgets, "--sims", args.sims,
                          "--out", out]))
    if "horizon" in args.stages:
        for env in ENVS:
            for H in args.horizon_list.get(env, HORIZONS[env]):
                d = load(os.path.join(out, f"horizon_{env}_H{H}.json"))
                if not (d and "wall_s" in d):
                    J.append((f"horizon_{env}_H{H}", 1 if H >= 12 else 2,
                              [py, "experiments/e19_horizon.py", "--env", env, "--H", str(H),
                               "--seeds", str(args.hseeds), "--episodes", str(args.episodes),
                               "--sims", args.sims, "--model_eps", str(args.model_eps),
                               "--out", out]))
    J.sort(key=lambda j: -j[1])                     # heaviest first
    return J


def run(job, args):
    name, _, cmd = job
    os.makedirs(os.path.join(args.out, "logs"), exist_ok=True)
    env = dict(os.environ, PYTHONPATH=ROOT, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1",
               XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
    t0 = time.time()
    with open(os.path.join(args.out, "logs", name + ".log"), "a") as log:
        shown = ["python" if c == sys.executable else os.path.relpath(c, ROOT) if os.path.isabs(c) else c for c in cmd]
        log.write(f"\n=== {time.ctime()}  {' '.join(shown)}\n")  # relative paths: logs stay anonymous
        log.flush()
        rc = subprocess.call(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    return name, rc, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="2x2,transfer,planners,horizon")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--hseeds", type=int, default=3, help="seeds for the horizon stage")
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--budgets", default="25,50,100,200,400")
    ap.add_argument("--sims", default="100,1000")
    ap.add_argument("--model_eps", type=int, default=400)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "e19"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    args.stages = args.stages.split(",")
    args.horizon_list = {}
    if args.smoke:
        args.out = os.path.join(ROOT, "results", "e19_smoke")
        args.seeds, args.hseeds, args.episodes, args.budgets, args.sims, args.model_eps = 1, 1, 60, "25,60", "50", 40
        args.horizon_list = {"random_oldp": [7], "pointing": [10], "qubit_prep": [9]}
    os.makedirs(args.out, exist_ok=True)
    J = jobs(args)
    print(f"{len(J)} jobs, {args.jobs} in parallel, output -> {os.path.relpath(args.out, ROOT)}")
    for j in J:
        print("  ", j[0], "" if not args.dry else " ".join(j[2]))
    if args.dry or not J:
        return
    t0, bad = time.time(), []
    with ThreadPoolExecutor(args.jobs) as ex:
        futs = [ex.submit(run, j, args) for j in J]
        for f in as_completed(futs):
            name, rc, dt = f.result()
            print(f"[{time.time() - t0:7.0f}s] {'ok ' if rc == 0 else 'FAIL'} {name} ({dt:.0f}s)", flush=True)
            if rc:
                bad.append(name)
    print("failed:", bad if bad else "none")
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
