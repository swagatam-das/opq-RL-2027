"""Collect the E19 outputs into markdown tables (printed and written to <dir>/summary.md).

    python summarize_e19.py [results/e19]
"""
import glob, json, os, sys
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "e19")
ENVS = ["random_oldp", "pointing", "qubit_prep"]
LABEL = {"random_oldp": "C1 random OLDP", "pointing": "C2 SO(3) pointing", "qubit_prep": "C3 two-qubit prep."}
out = []


def P(s=""):
    print(s)
    out.append(s)


def ms(x, d=2):
    x = np.asarray(x, float)
    return f"{x.mean():.{d}f}±{x.std():.{d}f}" if len(x) > 1 else f"{x.mean():.{d}f}"


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


# ---------------------------------------------------------------------------------- 2x2
CELLS = [("dqn", "DQN on ρ_t", "nonlinear", "unknown"),
         ("dqn_aug", "DQN + K_a-generated transitions", "nonlinear", "known"),
         ("lin_q", "Linear-in-ρ Q-learning (soft)", "linear", "unknown"),
         ("opq_soft", "OpQ-soft (β=5)", "linear", "known")]
EXTRA = [("lin_q_max", "Linear-in-ρ Q-learning (max)"), ("dqn_u4", "DQN, 4 updates/step"),
         ("dqn_aug_u4", "DQN + K_a transitions, 4 updates/step"), ("dqn_aug_knownR", "DQN + K_a transitions, true R"),
         ("dqn_aug_reach", "DQN + K_a transitions, on-support ρ only"),
         ("model_exh", "Learned (R,P) + exhaustive planning")]


def at(run, ep):
    for e, (s, v) in run["curve"]:
        if e == ep:
            return s
    return run["curve"][-1][1][0]


P("## 2×2: value function (linear / nonlinear in ρ) × operators K_a (unknown / known)")
P("Normalised return of the greedy policy (exact evaluation), mean±sd over seeds, after 100 / 3000 episodes.\n")
for env in ENVS:
    rows = []
    for reg in ("fixed", "random"):
        d = load(os.path.join(D, f"{env}_{reg}.json"))
        if d:
            rows.append((reg, d))
    if not rows:
        continue
    P(f"### {LABEL[env]}")
    P("| agent | class | K_a | " + " | ".join(f"{r} @100 | {r} @3000" for r, _ in rows) + " |")
    P("|---|---|---|" + "---|---|" * len(rows))
    for key, lab, cls, ka in CELLS:
        cells = []
        for reg, d in rows:
            rs = d["runs"].get(key)
            cells += [ms([at(r, 100) for r in rs]), ms([r["curve"][-1][1][0] for r in rs])] if rs else ["–", "–"]
        P(f"| **{lab}** | {cls} | {ka} | " + " | ".join(cells) + " |")
    for key, lab in EXTRA:
        cells = []
        for reg, d in rows:
            rs = d["runs"].get(key)
            cells += [ms([at(r, 100) for r in rs]), ms([r["curve"][-1][1][0] for r in rs])] if rs else ["–", "–"]
        if any(c != "–" for c in cells):
            P(f"| {lab} | | | " + " | ".join(cells) + " |")
    P()

# ---------------------------------------------------------------------------------- transfer
P("## Zero-shot transfer: trained at one latent, evaluated at 8 held-out latents (normalised return)")
P("| env | agent | training ρ_0 | held-out ρ_0 |\n|---|---|---|---|")
for env in ENVS:
    d = load(os.path.join(D, f"{env}_transfer.json"))
    if not d:
        continue
    for a, rs in d["runs"].items():
        P(f"| {LABEL[env]} | {a} | {ms([r['in_dist'] for r in rs])} | {ms([r['transfer'] for r in rs])} |")
P()

# ---------------------------------------------------------------------------------- planners
P("## Planners with the same knowledge as OpQ (K_a known, R and P learned from n random episodes)")
P("Normalised return at the training latent / mean over 8 held-out latents; mean over seeds. "
  "MCTS-S = S simulations per decision.\n")
for env in ENVS:
    d = load(os.path.join(D, f"planners_{env}.json"))
    if not d:
        continue
    P(f"### {LABEL[env]}  (H={d['H']}, K={d['K']}; exhaustive search = {d['cost']['exhaustive_nodes']:,} nodes, "
      f"{d['cost']['exhaustive_s']:.2f} s per decision at t=0)")
    meths = sorted({r["method"] for r in d["rows"] if r["model"] == "learned"})
    P("| model data (episodes) | " + " | ".join(meths) + " |\n|---|" + "---|" * len(meths))
    for n in d["budgets"]:
        cells = []
        for m in meths:
            rs = [r for r in d["rows"] if r["method"] == m and r["n_ep"] == n and r["model"] == "learned"]
            cells.append(f"{np.mean([r['score_train'] for r in rs]):.3f} / {np.mean([r['score_heldout'] for r in rs]):.3f}"
                         if rs else "–")
        P(f"| {n} | " + " | ".join(cells) + " |")
    ex = [r for r in d["rows"] if r["model"] == "exact"]
    if ex:
        P("| exact model | " + " | ".join(
            next((f"{r['score_train']:.3f} / {r['score_heldout']:.3f}" for r in ex if r["method"] == m),
                 "1.000 / 1.000" if m == "exhaustive" else "–") for m in meths) + " |")
    P("\nCost per decision (s): " + ", ".join(f"{k.replace('_s', '')}={v:.3g}" for k, v in d["cost"].items()
                                                if k.endswith("_s")) + "\n")

# ---------------------------------------------------------------------------------- horizon
P("## Longer horizons")
P("Value = expected discounted return (Monte Carlo over mode paths, common random numbers), mean over seeds. "
  "`score` = (V − V_rand)/(V_ref − V_rand) with V_ref = exact V* when it is computed (for two-qubit preparation V* = 1 "
  "analytically) and otherwise the best value found by any method (a lower bound on V*). `cert` = operator-LSE certificate / V_ref (an upper bound on V*, so ≥ 1). "
  "ms = wall-clock per decision at t=0. LSE-spread = max β·eigenvalue spread seen by OpQ (the double-precision LSE "
  "is reliable below about 25).\n")
for env in ENVS:
    files = sorted(glob.glob(os.path.join(D, f"horizon_{env}_H*.json")),
                   key=lambda f: int(f.split("_H")[-1].split(".")[0]))
    if not files:
        continue
    P(f"### {LABEL[env]}")
    names = None
    table = []
    for f in files:
        d = load(f)
        if not d or "wall_s" not in d:
            continue
        lat = lambda v, k=None: v[0] if k is None else v[:k]
        vr = d["vrand"][0]
        vals = {m: np.mean([r["values"][0] for r in rs]) for m, rs in d["methods"].items()}
        held = {m: np.mean([np.mean(r["values"][1:]) for r in rs]) for m, rs in d["methods"].items()}
        vref = d["vstar"][0] if d["vstar"] else (d.get("v_analytic") or max(vals.values()))
        hr = np.mean(d["vrand"][1:])
        href = np.mean(d["vstar"][1:]) if d["vstar"] else None
        cert = d["certificate"]["5.0"][0] / vref
        # held-out: first two held-out latents, the set every method (incl. MCTS) was evaluated on
        h2 = {m: np.mean([np.mean(r["values"][1:3]) for r in rs]) for m, rs in d["methods"].items()}
        hr2 = np.mean(d["vrand"][1:3])
        href2 = (np.mean(d["vstar"][1:3]) if d["vstar"] else (d.get("v_analytic") or max(h2.values())))
        table.append((d, vals, held, vref, vr, cert, h2, hr2, href2))
    if not table:
        continue
    allm = sorted({m for t in table for m in t[1]})
    P("| H | exact V* | V_ref | cert/V_ref | " + " | ".join(allm) + " |\n|---|---|---|---|" + "---|" * len(allm))
    for d, vals, held, vref, vr, cert, *_ in table:
        sc = lambda v: (v - vr) / (vref - vr)
        P(f"| {d['H']} | {'yes' if d['vstar'] else 'no'} | {vref:.3f} | {cert:.2f} | " +
          " | ".join(f"{sc(vals[m]):.3f}" if m in vals else "–" for m in allm) + " |")
    P("\nZero-shot transfer (trained at the fixed latent; score on 2 held-out latents, same normalisation):")
    P("| H | " + " | ".join(allm) + " |\n|---|" + "---|" * len(allm))
    for d, vals, held, vref, vr, cert, h2, hr2, href2 in table:
        P(f"| {d['H']} | " + " | ".join(f"{(h2[m] - hr2) / (href2 - hr2):.3f}" if m in h2 else "–" for m in allm) + " |")
    P("\nDecision time (ms) and OpQ LSE-spread:")
    P("| H | " + " | ".join(allm) + " | LSE-spread |\n|---|" + "---|" * (len(allm) + 1))
    for d, *_ in table:
        cells = []
        for m in allm:
            rs = d["methods"].get(m)
            cells.append(f"{1e3 * np.mean([r['decision_s'] for r in rs]):.2g}" if rs and "decision_s" in rs[0] else "–")
        sp = [r.get("lse_beta_spread") for r in d["methods"].get("opq_soft", []) if r.get("lse_beta_spread") is not None]
        P(f"| {d['H']} | " + " | ".join(cells) + f" | {np.max(sp):.0f} |" if sp else f"| {d['H']} | " + " | ".join(cells) + " | – |")
    P(f"\nExhaustive planning needs {', '.join(f'H={t[0]['H']}: {t[0]['exhaustive_nodes']:.2e}' for t in table)} node expansions per decision.\n")

open(os.path.join(D, "summary.md"), "w", encoding="utf-8").write("\n".join(out))
print(f"\n[written to {os.path.relpath(os.path.join(D, 'summary.md'))}]")
