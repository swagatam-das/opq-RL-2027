"""E7: Lipschitz behaviour of the operator log-sum-exp (Prop. 4 of the paper).

(i)   commuting sector: 20,000 random configurations (shared random eigenbasis);
      largest ratio ||LSE(x)-LSE(y)||_op / max_i ||x_i - y_i||_op (Prop. 4a: <= 1)
(ii)  the K = N = 25, beta = 5 construction of Prop. 4c: max-aggregated HS ratio
      equals p_beta sqrt(N) = 4.304 > 1 + beta/4 = 2.25 (the bound claimed in the
      original submission); l2-aggregated ratio <= 1
(iii) non-commuting inputs: local operator-norm ratios in 80-digit arithmetic
      (double precision gives spurious ratios once e^{2 beta M} ~ 1e16)
"""
import os, sys
import numpy as np
import mpmath as mp
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from opq.core import lse

rng = np.random.default_rng(0)
op = lambda A: np.linalg.norm(A, 2)

# (i) commuting sector ---------------------------------------------------------------
worst, n_trials = 0.0, 20000
betas = [0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10]
for k in range(n_trials):
    N, K, b = rng.integers(2, 6), rng.integers(2, 5), betas[k % len(betas)]
    V = np.linalg.qr(rng.normal(size=(N, N)))[0]
    lx = rng.uniform(-1, 1, size=(K, N))
    if k % 2:                                           # local perturbation
        ly = lx + rng.uniform(1e-4, 1e-2) * rng.normal(size=(K, N))
    else:                                               # independent draw
        ly = rng.uniform(-1, 1, size=(K, N))
    xs = [V @ np.diag(l) @ V.T for l in lx]
    ys = [V @ np.diag(l) @ V.T for l in ly]
    ratio = op(lse(xs, b) - lse(ys, b)) / max(op(x - y) for x, y in zip(xs, ys))
    worst = max(worst, ratio)
print(f"(i)   commuting sector, {n_trials} configurations: largest op-norm ratio = {worst:.6f}")

# (ii) K = N = 25 construction -------------------------------------------------------
def lse_plain(xs, b):
    f = lambda A, g: (lambda w, V: (V * g(w)) @ V.T)(*np.linalg.eigh(A))
    return f(sum(f(x, lambda w: np.exp(b * w)) for x in xs), np.log) / b
hs = lambda A: np.linalg.norm(A) / np.sqrt(A.shape[0])
N = K = 25; b = 5.0; eps = 1e-6
xs = [np.diag(np.eye(N)[i]) for i in range(K)]
ys = [x + eps * np.diag(np.eye(N)[i]) for i, x in enumerate(xs)]
d = lse_plain(xs, b) - lse_plain(ys, b)
p = np.exp(b) / (np.exp(b) + K - 1)
print(f"(ii)  K=N=25, beta=5: max-aggregated HS ratio {hs(d) / max(hs(x - y) for x, y in zip(xs, ys)):.4f} "
      f"(predicted p*sqrt(N) = {p * np.sqrt(N):.4f}; bound 1+beta/4 = {1 + b / 4:.2f}); "
      f"l2-aggregated ratio {hs(d) / np.sqrt(sum(hs(x - y) ** 2 for x, y in zip(xs, ys))):.4f}")

# (iii) non-commuting, 80-digit arithmetic --------------------------------------------
mp.mp.dps = 80
rng = np.random.default_rng(2)
def rsym(n, s=1.0):
    A = rng.normal(size=(n, n)); A = (A + A.T) / 2
    return s * A / np.linalg.norm(A, 2)
def LSE_mp(xs, b):
    S = None
    for x in xs:
        E = mp.expm(mp.matrix(x.tolist()) * b)
        S = E if S is None else S + E
    return mp.logm(S) / b
tonp = lambda M: np.array([[float(mp.re(v)) for v in row] for row in M.tolist()])
for b in [1, 5, 10]:
    wop = 0
    for t in range(80):
        n = 2 if t % 2 else 3
        k = 2 if t % 3 else 3
        xs = [rsym(n) for _ in range(k)]
        h = [rsym(n, 1e-12) * rng.uniform(0.1, 1) for _ in range(k)]
        dd = tonp(LSE_mp([x + g for x, g in zip(xs, h)], b) - LSE_mp(xs, b))
        wop = max(wop, op(dd) / max(op(g) for g in h))
    print(f"(iii) non-commuting, beta={b:2d}: largest local op-norm ratio over 80 draws = {wop:.3f}")
