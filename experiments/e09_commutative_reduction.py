"""E9: exact commutative reduction (Prop. 2).

When all U_a and R(s,a) are diagonal in one basis, the soft operator recursion
(Eq. 4) stays diagonal and each diagonal coordinate follows classical soft
Q-iteration; the sample-based OpQ update then coincides, step by step, with N
decoupled copies of scalar soft Q-learning fed the same samples.  For N = 1 the
operator update *is* soft Q-learning.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from opq.core import lse

def slse(v, b):
    c = v.max(); return c + np.log(np.exp(b * (v - c)).sum()) / b

rng = np.random.default_rng(0)
for N in [1, 4]:
    nS, K, gam, b = 3, 3, 0.9, 3.0
    P = rng.dirichlet(np.ones(nS), size=(nS, K))
    phases = rng.uniform(0, 2 * np.pi, size=(K, N))
    U = [np.diag(np.exp(1j * ph)) for ph in phases]            # commuting (diagonal) unitaries
    r = rng.normal(size=(nS, K, N))
    R = [[np.diag(r[s, a]).astype(complex) for a in range(K)] for s in range(nS)]
    dual = lambda a, Y: U[a].conj().T @ Y @ U[a]
    # (a) exact recursion: 200 iterations of T_beta vs N scalar soft Bellman operators
    Q = [[np.zeros((N, N), complex) for _ in range(K)] for _ in range(nS)]
    q = np.zeros((nS, K, N))
    for _ in range(200):
        V = [lse(Q[s2], b) for s2 in range(nS)]
        Q = [[dual(a, R[s][a] + gam * sum(P[s, a, s2] * V[s2] for s2 in range(nS)))
              for a in range(K)] for s in range(nS)]
        v = np.array([[slse(q[s2, :, j], b) for j in range(N)] for s2 in range(nS)])
        q = r + gam * np.einsum('sat,tj->saj', P, v)
    offd = max(np.abs(Q[s][a] - np.diag(np.diag(Q[s][a]))).max() for s in range(nS) for a in range(K))
    err = max(np.abs(np.diag(Q[s][a]).real - q[s, a]).max() for s in range(nS) for a in range(K))
    # (b) sample-based: same samples fed to operator TD and to N scalar soft Q-learners
    Qs = [[np.zeros((N, N), complex) for _ in range(K)] for _ in range(nS)]
    qs = np.zeros((nS, K, N))
    dev = 0.0
    for k in range(1, 20001):
        s, a = rng.integers(nS), rng.integers(K)
        s2 = rng.choice(nS, p=P[s, a]); al = 1 / (1 + k / 100) ** 0.7
        Qs[s][a] = Qs[s][a] + al * (dual(a, R[s][a] + gam * lse(Qs[s2], b)) - Qs[s][a])
        qs[s, a] = qs[s, a] + al * (r[s, a] + gam * np.array([slse(qs[s2, :, j], b) for j in range(N)]) - qs[s, a])
        dev = max(dev, np.abs(np.diag(Qs[s][a]).real - qs[s, a]).max())
    print(f"N={N}: exact recursion  max off-diagonal {offd:.1e}, max |diag - scalar soft Q| {err:.1e}; "
          f"sample-based (20k updates) max deviation from scalar soft Q-learning {dev:.1e}")
