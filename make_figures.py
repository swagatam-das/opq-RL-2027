"""Produce every figure and LaTeX table of the paper from results/ (run after the experiments)."""
import json, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines
import matplotlib.patches
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.abspath(__file__))
RES, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "Images")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "legend.fontsize": 6.5,
                     "figure.dpi": 150, "pdf.fonttype": 42})

E18_NAMES = {"opq_soft": "OpQ-soft (learned $R$)", "opq_soft_knownR": "OpQ-soft (known $R$)",
             "opq_pi": "OpQ-PI", "scalar_state": "Scalar $Q$ on $(t,m)$",
             "scalar_history": "Scalar $Q$, history table", "dqn": "DQN on $\\rho_t$",
             "drqn": "DRQN (GRU) on history"}
E18_STYLE = {"opq_soft": ("C3", "-", "o"), "opq_soft_knownR": ("C3", ":", None),
             "opq_pi": ("C1", "-", "s"), "scalar_state": ("C7", "-", "x"),
             "scalar_history": ("C8", "--", "+"), "dqn": ("C0", "-", "^"), "drqn": ("C2", "-", "v")}
ENV_TITLES = {"random_oldp": "C1 random OLDP", "pointing": "C2 $SO(3)$ pointing",
              "qubit_prep": "C3 two-qubit preparation"}


def e18():
    envs = ["random_oldp", "pointing", "qubit_prep"]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 3.9), sharex=True)
    rows = []
    for j, env in enumerate(envs):
        for i, regime in enumerate(["fixed", "random"]):
            f = os.path.join(RES, "e18", f"{env}_{regime}.json")
            if not os.path.exists(f):
                continue
            d = json.load(open(f))
            ax = axes[i, j]
            for name, runs in d["runs"].items():
                c, ls, mk = E18_STYLE[name]
                x = [p[0] for p in runs[0]["curve"]]
                Y = np.array([[p[1][0] for p in r["curve"]] for r in runs])
                ax.plot(x, Y.mean(0), color=c, ls=ls, marker=mk, ms=2.5, lw=1, label=E18_NAMES[name])
                ax.fill_between(x, Y.mean(0) - Y.std(0), Y.mean(0) + Y.std(0), color=c, alpha=0.12, lw=0)
                rows.append((env, regime, name, Y[:, 0].mean(), Y[:, 2].mean() if Y.shape[1] > 2 else np.nan,
                             Y[:, -1].mean(), Y[:, -1].std()))
            ax.axhline(1.0, color="k", lw=0.6)
            ax.set_xscale("log")
            ax.set_ylim(-0.1, 1.05)
            ax.set_title(f"{ENV_TITLES[env]}, {'fixed' if regime == 'fixed' else 'random'} $\\rho_0$")
            if j == 0:
                ax.set_ylabel("normalised return")
            if i == 1:
                ax.set_xlabel("training episodes")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(os.path.join(FIG, "fig_e18_control.pdf"), bbox_inches="tight")
    plt.close(fig)
    return rows


def e18_table(rows):
    """LaTeX tables: main (final scores + transfer) and appendix (100 episodes / final, +- sd)."""
    envs = ["random_oldp", "pointing", "qubit_prep"]
    names = ["opq_soft", "opq_soft_knownR", "opq_pi", "scalar_state", "scalar_history", "dqn", "drqn"]
    tr = {}
    for env in envs:
        f = os.path.join(RES, "e18", f"{env}_transfer.json")
        if os.path.exists(f):
            d = json.load(open(f))
            for n, runs in d["runs"].items():
                tr[(env, n)] = (np.mean([r["transfer"] for r in runs]), np.std([r["transfer"] for r in runs]))
    get = lambda env, regime, n: ([x for x in rows if x[0] == env and x[1] == regime and x[2] == n] or [None])[0]
    main, app = [], []
    for n in names:
        c1, c2 = [], []
        for env in envs:
            for regime in ["fixed", "random"]:
                r = get(env, regime, n)
                c1.append(f"{r[5]:.2f}" if r else "--")
                c2.append(f"{r[4]:.2f} & {r[5]:.2f}$\\pm${r[6]:.2f}" if r else "-- & --")
            t = tr.get((env, n))
            c1.append(f"{t[0]:.2f}" if t else "--")
        main.append(E18_NAMES[n] + " & " + " & ".join(c1) + " \\\\")
        app.append(E18_NAMES[n] + " & " + " & ".join(c2) + " \\\\")
    open(os.path.join(RES, "table_e18.tex"), "w").write("\n".join(main) + "\n")
    open(os.path.join(RES, "table_e18_app.tex"), "w").write("\n".join(app) + "\n")


def e18_refs():
    tex = []
    for env in ["random_oldp", "pointing", "qubit_prep"]:
        for regime in ["fixed", "random"]:
            f = os.path.join(RES, "e18", f"{env}_{regime}.json")
            if os.path.exists(f):
                d = json.load(open(f)); r = d["reference"]; vs = np.mean(d["vstar"])
                cells = [f"{r[f'opq_soft_exact_beta{b}']['score'][0]:.2f} "
                         f"({r[f'opq_soft_exact_beta{b}']['certificate'] / vs:.2f})" for b in ["1", "2", "5", "10"]]
                tex.append(f"{ENV_TITLES[env].split(' ', 1)[1]}, {regime} & " + " & ".join(cells)
                           + f" & {r['opq_pi_exact']['score'][0]:.2f} \\\\")
    open(os.path.join(RES, "table_e18_ref.tex"), "w").write("\n".join(tex) + "\n")
    lines = []
    for env in ["random_oldp", "pointing", "qubit_prep"]:
        for regime in ["fixed", "random"]:
            f = os.path.join(RES, "e18", f"{env}_{regime}.json")
            if os.path.exists(f):
                d = json.load(open(f))
                r = d["reference"]
                lines.append(f"{env} {regime}: " + ", ".join(
                    f"{k}: score {v['score'][0]:.3f}" + (f" cert {v['certificate']:.2f}" if 'certificate' in v else "")
                    for k, v in r.items()) + f" | V* {np.mean(d['vstar']):.3f}")
    open(os.path.join(RES, "e18_reference.txt"), "w").write("\n".join(lines) + "\n")


def e10():
    f = os.path.join(RES, "e10.json")
    if not os.path.exists(f):
        return
    d = json.load(open(f))
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.3))
    for name, lab, c in [("so3", "so(3) (simple)", "C3"), ("n3", "n3 (nilpotent, step 2)", "C0")]:
        ax[0].plot(range(2, 11), d[name]["S"], marker="o", ms=3, color=c, label=lab)
        T = sorted(int(t) for t in d[name]["deficit"])
        ax[1].plot(T, [d[name]["deficit"][str(t)]["normalised"] for t in T], marker="o", ms=3, color=c, label=lab)
    ax[0].set_xlabel("$T$"); ax[0].set_ylabel("$S(T)$"); ax[0].set_title("(a) nested-commutator depth profile")
    ax[1].set_xlabel("$T$"); ax[1].set_ylabel("deficit / sd$(r)$"); ax[1].set_title("(b) permutation-averaged deficit")
    ax[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_e10_horizon.pdf"), bbox_inches="tight")
    plt.close(fig)


REP_TITLES = {"orth5": "E8/E13: random orthogonal, $N{=}5$", "so3": "E14: so(3), $N{=}3$",
              "se3": "E16: se(3) hinge chain, $N{=}4$", "boson": "E15: displacements, $N_F{=}6$"}


def rep():
    envs = [e for e in ["orth5", "so3", "se3", "boson"] if os.path.exists(os.path.join(RES, "rep", f"{e}_seed0.json"))]
    if not envs:
        return
    fig, axes = plt.subplots(1, len(envs), figsize=(1.9 * len(envs) + 0.4, 2.4), squeeze=False)
    tab = []
    for ax, e in zip(axes[0], envs):
        d = json.load(open(os.path.join(RES, "rep", f"{e}_seed0.json")))
        ops = [(k, v) for k, v in d["models"].items() if k.startswith("OpQ")]
        ax.plot([v["params"] for _, v in ops], [max(v["test_mse"], 1e-33) for _, v in ops], "o-", color="C3",
                ms=3, label="OpQ, $d=1,\\dots$")
        for k, v in ops:
            ax.annotate(k.split("=")[1], (v["params"], max(v["test_mse"], 1e-33)), fontsize=5.5,
                        xytext=(2, 2), textcoords="offset points")
        mk = {"Scalar": ("x", "C7"), "Windowed": ("+", "C8"), "GRU": ("v", "C2"), "LSTM": ("^", "C0"),
              "Self-attention": ("D", "C4"), "1D-CNN": ("s", "C5")}
        seen = set()
        for k, v in d["models"].items():
            if k.startswith("OpQ"):
                continue
            key = k.split(",")[0].split(" ")[0]
            m, c = mk[key]
            ax.plot(v["params"], v["test_mse"], m, color=c, ms=4, label=None if key in seen else key)
            seen.add(key)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(REP_TITLES[e], fontsize=7)
        ax.set_xlabel("parameters")
        tab.append((e, d))
    axes[0, 0].set_ylabel("test MSE")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=7, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(os.path.join(FIG, "fig_rep_benchmark.pdf"), bbox_inches="tight")
    plt.close(fig)
    # LaTeX table
    names = list(tab[0][1]["models"].keys())
    allnames = []
    for _, d in tab:
        for k in d["models"]:
            if k not in allnames:
                allnames.append(k)
    lines = []
    fmt = lambda x: f"{x:.3f}" if x >= 1e-3 else f"${x:.1e}".replace("e-", "\\times10^{-") + "}$"
    for k in allnames:
        cells = []
        for _, d in tab:
            v = d["models"].get(k)
            cells.append(f"{v['params']} & {fmt(v['test_mse'])}" if v else "-- & --")
        lines.append(k.replace("_", "\\_") + " & " + " & ".join(cells) + " \\\\")
    lines.append("\\midrule")
    lines.append("Var(test returns) & " + " & ".join(f" & {d['var_test']:.3f}" for _, d in tab) + " \\\\")
    lines.append("History table entries & " + " & ".join(f"{d['table_entries']} & $0$" for _, d in tab) + " \\\\")
    open(os.path.join(RES, "table_rep.tex"), "w").write("\n".join(lines) + "\n")


def origami():
    """Worked example (Appendix A): plans (V, D, H) and (D, V, H)."""
    A = {"V": np.array([[-1, 0], [0, 1.0]]), "H": np.array([[1, 0], [0, -1.0]]), "D": np.array([[0, 1], [1, 0.0]])}
    W = (A["V"] @ A["D"] @ A["H"]).T
    F = np.array([[0, 0], [0, 1], [0.55, 1], [0, 1], [0, 0.55], [0.4, 0.55]])   # letter-F polyline (row vectors)
    fig, axes = plt.subplots(2, 4, figsize=(5.6, 3.0))
    for i, plan in enumerate([("V", "D", "H"), ("D", "V", "H")]):
        X = np.eye(2)
        for t in range(4):
            ax = axes[i, t]
            P = F @ X
            ax.plot(P[:, 0], P[:, 1], "-", color="C3" if i == 0 else "C0", lw=2)
            ax.plot([-1.1, 1.1], [0, 0], color="0.85", lw=0.5); ax.plot([0, 0], [-1.1, 1.1], color="0.85", lw=0.5)
            ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15); ax.set_aspect("equal"); ax.axis("off")
            ttl = "start" if t == 0 else f"after {plan[t - 1]}"
            if t == 3:
                ttl += f"\n$r=\\tau(WX_3)={np.trace(W @ X) / 2:+.0f}$"
            ax.set_title(ttl, fontsize=7)
            if t < 3:
                X = X @ A[plan[t]]
        axes[i, 0].text(-1.6, 0, "(" + ",".join(plan) + ")", fontsize=8, ha="right", va="center")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "origami_opq_compare.pdf"), bbox_inches="tight")
    plt.close(fig)


def overview():
    """Figure 1 (main text): order aliasing, the compositional operator backup,
    the unrolled recursion, and the Loewner picture of the soft certificate."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    from opq.core import lse
    A = {"V": np.array([[-1, 0], [0, 1.0]]), "H": np.array([[1, 0], [0, -1.0]]), "D": np.array([[0, 1], [1, 0.0]])}
    W = (A["V"] @ A["D"] @ A["H"]).T
    F = np.array([[0, 0], [0, 1], [0.55, 1], [0, 1], [0, 0.55], [0.4, 0.55]])
    fig = plt.figure(figsize=(7.0, 3.95))
    Y = lambda y: 0.50 + 0.50 * y          # top row occupies the upper half
    S = 0.50

    def box(x, y, w, h, txt, ec, fs=7):
        fig.patches.append(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008", transform=fig.transFigure,
                                          fc="white", ec=ec, lw=1.0))
        fig.text(x + w / 2, y + h / 2, txt, fontsize=fs, ha="center", va="center", transform=fig.transFigure)

    def arrow(p, q, c="0.3", style="-|>"):
        fig.patches.append(FancyArrowPatch(p, q, transform=fig.transFigure, arrowstyle=style,
                                           mutation_scale=8, color=c, lw=1.0))
    # ---- (a)
    for i, (plan, col) in enumerate([(("V", "D", "H"), "C3"), (("D", "V", "H"), "C0")]):
        X = np.eye(2)
        for t in range(4):
            ax = fig.add_axes([0.035 + 0.075 * t, Y(0.53 - 0.44 * i), 0.068, 0.30 * S])
            P = F @ X
            ax.plot([-1.1, 1.1], [0, 0], color="0.85", lw=0.5); ax.plot([0, 0], [-1.1, 1.1], color="0.85", lw=0.5)
            ax.plot(P[:, 0], P[:, 1], "-", color=col, lw=1.8)
            ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15); ax.set_aspect("equal"); ax.axis("off")
            if t > 0:
                ax.set_title(plan[t - 1], fontsize=7, pad=1)
            if t < 3:
                X = X @ A[plan[t]]
        r = np.trace(W @ X) / 2
        fig.text(0.345, Y(0.68 - 0.44 * i), f"$r={r:+.0f}$", fontsize=8, color=col, va="center")
        fig.text(0.005, Y(0.68 - 0.44 * i), "(" + ",".join(plan) + ")", fontsize=6.5, rotation=90, va="center")
    fig.text(0.20, Y(0.97), "(a) same folds, different order", fontsize=7.5, ha="center", va="top", weight="bold")
    # ---- (b)
    fig.text(0.535, Y(0.97), "(b) scalar $Q$, order-blind", fontsize=7.5, ha="center", va="top", weight="bold")
    box(0.415, Y(0.50), 0.24, 0.30 * S, "histories $(V,D)$ and $(D,V)$\nboth observed as\n"
        "$\\sigma=\\{V{:}1,D{:}1\\}$, action $H$", "0.4")
    box(0.415, Y(0.08), 0.24, 0.30 * S, "one number $f(\\sigma,H)$ for both;\n"
        "targets $+1$ and $-1$\n$\\Rightarrow$ error $\\geq 1$ (Prop. 8)", "0.4")
    arrow((0.535, Y(0.49)), (0.535, Y(0.39)))
    # ---- (c)
    fig.text(0.845, Y(0.97), "(c) operator $Q$, compositional backup", fontsize=7.5, ha="center", va="top", weight="bold")
    box(0.705, Y(0.50), 0.285, 0.30 * S, "$Q(s,a)=\\mathcal{K}_a^{\\dagger}(R+\\gamma\\,\\mathbb{E}V)$\n"
        "unrolls to $A_{a_1}\\cdots A_{a_T}W$ (Cor. 1)\none $N{\\times}N$ operator per $(s,a)$", "C3")
    box(0.705, Y(0.08), 0.285, 0.30 * S, "value $=\\tau(X_h\\,Q(H))$, $Q(H)=A_HW$:\n"
        "$X_{VD}=A_VA_D\\Rightarrow+1$,  $X_{DV}=A_DA_V\\Rightarrow-1$\nexact for every history (Prop. 1)", "C3")
    arrow((0.8475, Y(0.49)), (0.8475, Y(0.39)), "C3")
    fig.add_artist(matplotlib.lines.Line2D([0.01, 0.99], [0.475, 0.475], transform=fig.transFigure,
                                           color="0.8", lw=0.6))
    # ---- (d) unrolled backup along the plan (V, D, H)
    fig.text(0.31, 0.455, "(d) the backup runs backwards and composes", fontsize=7.5, ha="center", va="top",
             weight="bold")
    xs = [0.435, 0.225, 0.015]
    labels = ["$Q_2(H)=\\mathcal{K}_H^{\\dagger}(W)$\n$=A_HW$",
              "$Q_1(D)=\\mathcal{K}_D^{\\dagger}(Q_2)$\n$=A_DA_HW$",
              "$Q_0(V)=\\mathcal{K}_V^{\\dagger}(Q_1)$\n$=A_VA_DA_HW$"]
    for x, lab in zip(xs, labels):
        box(x, 0.235, 0.175, 0.13, lab, "C3", fs=6.8)
    arrow((0.435, 0.30), (0.395, 0.30), "C3"); arrow((0.225, 0.30), (0.185, 0.30), "C3")
    fig.text(0.31, 0.19, "readout at $X_0=I$: $\\tau(Q_0(V))=\\frac{1}{N}\\mathrm{Tr}(WA_VA_DA_H)=+1$;  "
             "after history $h$: $\\tau(X_h\\,Q_{|h|})$", fontsize=6.8, ha="center", va="center")
    fig.text(0.31, 0.105, "memory: $|\\mathcal{S}|K$ operators ($N^2$ numbers each), for every horizon\n"
             "vs. $K^{T}$ entries for an exact history table (Prop. 9)", fontsize=6.8, ha="center", va="center",
             color="0.25")
    # ---- (e) Loewner picture of the soft certificate
    fig.text(0.815, 0.455, "(e) control: no operator max; LSE is an upper bound", fontsize=7.5, ha="center",
             va="top", weight="bold")
    ax = fig.add_axes([0.585, 0.03, 0.25, 0.37])
    rot = lambda th: np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    x1 = rot(0.35) @ np.diag([1.6, 0.35]) @ rot(0.35).T
    x2 = rot(1.25) @ np.diag([1.3, 0.45]) @ rot(1.25).T
    th = np.linspace(0, 2 * np.pi, 200)
    circ = np.stack([np.cos(th), np.sin(th)])

    def ell(M, **kw):
        w, V = np.linalg.eigh(M)
        P = V @ np.diag(np.sqrt(np.maximum(w, 1e-12))) @ V.T @ circ
        ax.plot(P[0], P[1], **kw)
    ell(x1, color="C0", lw=1.3, label="$x_1=Q(s',a_1)$")
    ell(x2, color="C2", lw=1.3, label="$x_2=Q(s',a_2)$")
    for b, ls in [(1.0, "--"), (6.0, "-")]:
        ell(lse([x1, x2], b), color="C3", lw=1.1, ls=ls, label=f"LSE$_\\beta$, $\\beta={b:g}$")
    ax.set_aspect("equal"); ax.axis("off")
    ax.legend(fontsize=6.2, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.55), handlelength=1.6)
    ax.text(0.45, -1.5, "ellipse $\\{x^{1/2}u:|u|\\leq1\\}$; containment = Loewner order", fontsize=6,
            ha="center", color="0.3")
    fig.savefig(os.path.join(FIG, "fig1_overview.pdf"), bbox_inches="tight")
    plt.close(fig)


def e24():
    """Fig. (E24): sup-norm distance of sampled OpQ-soft to the exact finite-horizon Q^beta."""
    path = os.path.join(ROOT, "results", "e24", "e24_results.json")
    if not os.path.exists(path):
        return
    d = json.load(open(path))
    names = [("random_oldp", "C1 random OLDP"), ("pointing", "C2 $SO(3)$ pointing"), ("qubit_prep", "C3 two-qubit prep.")]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.1), sharey=True)
    for ax, (env, title) in zip(axes, names):
        for rule, ls, lab in [("rm", "-", r"$\alpha_n=n^{-0.6}$ (Thm. 3)"), ("floor", "--", r"$\alpha_n=\max(0.02,n^{-0.6})$")]:
            rs = [r for r in d["runs"] if r["env"] == env and r["rule"] == rule and not r["known_R"]]
            ep = [c["episode"] for c in rs[0]["checkpoints"]]
            em = np.array([[max(c["err_max"], 1e-10) for c in r["checkpoints"]] for r in rs])
            ax.plot(ep, np.exp(np.log(em).mean(0)), ls, color="C0" if rule == "rm" else "C3", lw=1.3, label=lab)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(title, fontsize=8)
        ax.set_xlabel("episodes", fontsize=7); ax.tick_params(labelsize=6); ax.grid(alpha=0.3, lw=0.4)
    axes[0].set_ylabel(r"$\max\|Q_t-Q^\beta_t\|_{\mathrm{op}}/\max\|Q^\beta_t\|_{\mathrm{op}}$", fontsize=7)
    axes[0].legend(fontsize=6, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_e24_convergence.pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    overview()
    origami()
    rows = e18()
    e18_table(rows)
    e18_refs()
    e10()
    rep()
    e24()
    print("figures written to", FIG)
