"""Representation experiments (class B): E8/E13 (random orthogonal, N=5), E14 (so(3)),
E15 (bosonic displacements), E16 (se(3) hinge chain).

Every environment: 3000 training / 1000 test words of uniformly random actions;
the exact return (1/N) Re Tr(W A_{a1} ... A_{aT}) is the regression target.
Before any fitting, a planted-solution check confirms that the operator model
with the TRUE generators reproduces the targets (MSE ~ 1e-30).

Usage: python experiments/rep_benchmark.py --env orth5 [--quick]
"""
import argparse, json, os, sys, time
import numpy as np
from scipy.linalg import expm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from opq import rep_models as M


def realify(A):
    """Real 2n x 2n block encoding of a complex n x n matrix (multiplicative)."""
    return np.block([[A.real, -A.imag], [A.imag, A.real]])


def make_env(name, rng):
    if name == "orth5":                               # E8 / E13
        N, K, T = 5, 4, 6
        gens = []
        for _ in range(K):
            G = rng.normal(scale=0.6, size=(N, N))
            gens.append(G - G.T)
        A = [expm(g) for g in gens]
        W = rng.normal(size=(N, N)); W = (W + W.T) / 2
        return dict(N=N, K=K, T=T, A=A, W=W, gens=gens, ds=[1, 2, 3, 4, 5])
    if name == "so3":                                 # E14
        L = [np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]], float),
             np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]], float),
             np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]], float)]
        gens = [s * 0.6 * l for l in L for s in (1, -1)]
        A = [expm(g) for g in gens]
        return dict(N=3, K=6, T=8, A=A, W=rng.normal(size=(3, 3)), gens=gens, ds=[1, 2, 3])
    if name == "se3":                                 # E16
        gens = []
        for _ in range(6):
            n = rng.normal(size=3); n /= np.linalg.norm(n)
            p = rng.normal(size=3); th = rng.uniform(0.5, 1.5)
            nx = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
            O = np.zeros((4, 4)); O[:3, :3] = th * nx; O[:3, 3] = -th * nx @ p
            gens.append(O)
        A = [expm(g) for g in gens]
        return dict(N=4, K=6, T=8, A=A, W=rng.normal(size=(4, 4)), gens=gens, ds=[1, 2, 3, 4])
    if name == "boson":                               # E15
        NF = 6
        a = np.diag(np.sqrt(np.arange(1, NF)), 1)
        alphas = 0.5 * np.exp(1j * rng.uniform(0, 2 * np.pi, 6)) * rng.uniform(0.6, 1.4, 6)
        gens_c = [al * a.conj().T - np.conj(al) * a for al in alphas]
        Wc = rng.normal(size=(NF, NF)) + 1j * rng.normal(size=(NF, NF)); Wc = (Wc + Wc.conj().T) / 2
        A = [realify(expm(g)) for g in gens_c]
        W = realify(Wc) / 2                          # Re Tr(Wc X) = Tr(realify(Wc) realify(X)) / 2
        return dict(N=2 * NF, NF=NF, K=6, T=6, A=A, W=W, gens=[realify(g) for g in gens_c],
                    gens_c=gens_c, alphas=alphas, a=a, ds=[2, 4, 6, 8, 10, 12], Nread=NF)
    raise ValueError(name)


def returns(env, seqs):
    Nread = env.get("Nread", env["N"])
    out = np.empty(len(seqs))
    for i, s in enumerate(seqs):
        X = np.linalg.multi_dot([np.eye(env["N"])] + [env["A"][a] for a in s])
        out[i] = np.trace(env["W"] @ X) / Nread
    return out


def depth_profile(gens, W, N, Tmax=10):
    S, terms = [], []
    for depth in range(1, Tmax):
        best = 0.0
        for a in range(len(gens)):
            for b in range(len(gens)):
                if a == b:
                    continue
                Mx = gens[b]
                for _ in range(depth):
                    Mx = gens[a] @ Mx - Mx @ gens[a]
                best = max(best, abs(np.trace(W @ Mx)) / N)
        terms.append(best)
    return np.cumsum(terms).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="orth5")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng({"orth5": 8, "so3": 14, "boson": 15, "se3": 16}[args.env] + 100 * args.seed)
    env = make_env(args.env, rng)
    K, T, N = env["K"], env["T"], env["N"]
    Xtr = rng.integers(K, size=(3000, T)); Xte = rng.integers(K, size=(1000, T))
    ytr, yte = returns(env, Xtr), returns(env, Xte)
    tr, te = (Xtr, ytr), (Xte, yte)
    res = {"env": args.env, "N": N, "K": K, "T": T, "var_test": float(np.var(yte)),
           "table_entries": K ** T, "models": {}}
    print(f"[{args.env}] N={N} K={K} T={T}  Var(test returns) = {np.var(yte):.4f}", flush=True)

    # planted-solution sanity check (true generators, no training)
    Nread = env.get("Nread", N)
    p_true = {"Q": np.stack(env["A"]), "W": env["W"] / Nread}
    res["planted_mse"] = float(np.mean((np.asarray(M.opq_apply(p_true, Xte)) - yte) ** 2))
    print(f"   planted-solution check: test MSE {res['planted_mse']:.2e}", flush=True)

    # diagnostics
    if args.env == "boson":
        prof = {}
        for NF in [6, 12, 24]:
            a = np.diag(np.sqrt(np.arange(1, NF)), 1)
            g = [al * a.conj().T - np.conj(al) * a for al in env["alphas"]]
            Wc = np.random.default_rng(1).normal(size=(NF, NF)); Wc = (Wc + Wc.T) / 2
            prof[NF] = depth_profile(g, Wc, NF)[-1]
        res["S10_by_truncation"] = prof
        al, be = env["alphas"][0], env["alphas"][1]
        a = env["a"]
        Da = lambda z: expm(z * a.conj().T - np.conj(z) * a)
        ab, ba = (Da(al) @ Da(be))[0, 0], (Da(be) @ Da(al))[0, 0]
        res["swap_phase_truncated"] = float(np.angle(ab / ba))
        res["swap_phase_exact"] = float(2 * np.imag(al * np.conj(be)))
        print(f"   S(10) by Fock truncation: {prof}; swap phase {res['swap_phase_truncated']:.4f} "
              f"(untruncated 2 Im(a b*) = {res['swap_phase_exact']:.4f})", flush=True)
    else:
        res["S"] = depth_profile(env["gens"], env["W"], N)
        print(f"   S(T), T=2..10: " + ", ".join(f"{s:.3g}" for s in res["S"]), flush=True)
    if args.env == "se3":
        x0 = np.array([1.0, 1.0, 0.0, 1.0])
        g = [(np.linalg.norm((env["A"][a] @ env["A"][b] @ x0 - env["A"][b] @ env["A"][a] @ x0)[:3]), a, b)
             for a in range(K) for b in range(a + 1, K)]
        res["marked_point_gap"] = float(max(g)[0])
        print(f"   marked-point swap gap (largest pair): {res['marked_point_gap']:.4f}", flush=True)

    q = args.quick
    t0 = time.time()
    mse, rank = M.fit_linear(tr, te, K, 1)
    res["models"]["Scalar order-blind (t, a_t)"] = {"params": rank, "test_mse": mse}
    mse, rank = M.fit_linear(tr, te, K, 2)
    res["models"]["Windowed table (t, a_{t-1}, a_t)"] = {"params": rank, "test_mse": mse}
    ep_nn, rs_nn = (300, 1) if q else (1000, 1)
    for h in [4, 16, 64]:
        res["models"][f"GRU, hidden {h}"] = M.fit(lambda k, h=h: M.rnn_init(k, K, h, "gru"),
                                                 lambda p, X: M.rnn_apply(p, X, "gru"),
                                                 tr, te, ep_nn, 0.01, rs_nn)
    res["models"]["LSTM, hidden 64"] = M.fit(lambda k: M.rnn_init(k, K, 64, "lstm"),
                                             lambda p, X: M.rnn_apply(p, X, "lstm"), tr, te, ep_nn, 0.01, rs_nn)
    res["models"]["Self-attention, width 32"] = M.fit(lambda k: M.attn_init(k, K, T, 32), M.attn_apply,
                                                      tr, te, ep_nn, 0.005, rs_nn)
    res["models"]["1D-CNN on action embeddings"] = M.fit(lambda k: M.cnn_init(k, K), M.cnn_apply,
                                                         tr, te, ep_nn, 0.01, rs_nn)
    ep_op, rs_op, pol = (300, 1, 100) if q else (1500, 4, 2000)
    for d in env["ds"]:
        res["models"][f"OpQ, d={d}"] = M.fit(lambda k, d=d: M.opq_init(k, K, d), M.opq_apply,
                                             tr, te, ep_op, 0.02, rs_op, polish=pol)
    for k, v in res["models"].items():
        print(f"   {k:34s} params {v['params']:6d}  test MSE {v['test_mse']:.3e}", flush=True)
    print(f"   (history table: {K ** T} entries; fit time {time.time() - t0:.0f}s)", flush=True)
    out = os.path.join(os.path.dirname(__file__), "..", "results", "rep")
    os.makedirs(out, exist_ok=True)
    json.dump(res, open(os.path.join(out, f"{args.env}_seed{args.seed}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
