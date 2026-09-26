"""E10: nilpotent (Heisenberg n_3) versus non-nilpotent (so(3)) control algebras (Thm. 5).

(a) nested-commutator depth profile
      S(T) = sum_{t=2}^{T} max_{a != b} |Tr(W ad_{Omega_a}^{t-1} Omega_b)| / N,
    computed exactly by iterated bracketing (no sampling);
(b) permutation-averaged deficit |r - r_hat| where r_hat averages the return over all
    T! orderings of the same multiset of actions (the best order-blind predictor
    given action counts), normalised by the standard deviation of returns.
Both algebras are represented in dimension 3 with every generator at Frobenius
norm sqrt(2); W is a general (non-symmetric) random matrix.
"""
import os, sys, json, itertools
import numpy as np
from scipy.linalg import expm

rng = np.random.default_rng(0)
N = 3
Lx = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]], float)
Ly = np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]], float)
Lz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]], float)
E = lambda i, j: np.eye(3)[:, [i]] @ np.eye(3)[[j], :]
X, Y, Z = E(0, 1), E(1, 2), E(0, 2)
norm = lambda A: np.sqrt(2) * A / np.linalg.norm(A)
algebras = {"so3": [norm(A) for A in (Lx, Ly, Lz)], "n3": [norm(A) for A in (X, Y, Z)]}
W = rng.normal(size=(N, N))
out = {}
for name, Om in algebras.items():
    terms = []
    for depth in range(1, 10):                    # t-1 = depth, t = 2..10
        best = 0.0
        for a in range(3):
            for b in range(3):
                if a == b:
                    continue
                M = Om[b]
                for _ in range(depth):
                    M = Om[a] @ M - M @ Om[a]
                best = max(best, abs(np.trace(W @ M)) / N)
        terms.append(best)
    S = np.cumsum(terms)
    # permutation-averaged deficit
    A = [expm(o) for o in Om]
    defs = {}
    for T in range(2, 9):
        nseq = 400 if T <= 6 else 100
        dev, rets = [], []
        perms = np.array(list(itertools.permutations(range(T))))
        for _ in range(nseq):
            seq = rng.integers(3, size=T)
            mats = np.stack([A[a] for a in seq])                     # (T, N, N)
            P = mats[perms[:, 0]]
            for k in range(1, T):
                P = P @ mats[perms[:, k]]
            r_all = np.einsum('ij,pji->p', W, P) / N
            r = r_all[0]                                             # identity permutation
            dev.append(abs(r - r_all.mean()))
            rets.append(r)
        defs[T] = {"mean_deficit": float(np.mean(dev)), "std_return": float(np.std(rets)),
                   "normalised": float(np.mean(dev) / np.std(rets))}
    out[name] = {"depth_terms": terms, "S": S.tolist(), "deficit": defs}
    print(f"{name}: per-depth terms " + ", ".join(f"{t:.3g}" for t in terms))
    print(f"{name}: S(T), T=2..10: " + ", ".join(f"{s:.4f}" for s in S))
    print(f"{name}: normalised deficit T=2..8: " +
          ", ".join(f"{defs[T]['normalised']:.3f}" for T in range(2, 9)))
os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results"), exist_ok=True)
json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results", "e10.json"), "w"), indent=1)
