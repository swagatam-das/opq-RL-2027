"""E17b: finite-instance checks of the extensions in Appendix D (Props. 6-8, Cor. 2).

(1) Prop. 7 (noisy / random latent dynamics): the exact representation (Thm. 1) and the
    Löwner certificate (Thm. 3) with non-unitary positive maps -- amplitude-damping-type
    Kraus channels, and random choice between two unitaries.
(2) Cor. 2 (certificate without contraction): infinite-horizon values versus the
    truncated certificate tau(rho Q^{beta,(n)}_0) + tail.
(3) Prop. 6 (robust greedy improvement): greedy on a perturbed operator Q_hat.
(4) Prop. 8 (structure-refined dimension): Hankel rank equals dim of the generated
    algebra for a reducible unitary representation.
"""
import itertools
from functools import reduce
import numpy as np
from scipy.linalg import expm, logm

rng = np.random.default_rng(0)
N, K, nS, g = 3, 3, 2, 0.9
tau = lambda X: np.real(np.trace(X)) / X.shape[0]


def herm(n):
    A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)); return (A + A.conj().T) / 2


def unitary(n, s=0.8):
    return expm(-1j * s * herm(n))


def lse(xs, b):
    c = max(np.linalg.eigvalsh(x)[-1] for x in xs)
    S = sum(expm(b * (x - c * np.eye(len(x)))) for x in xs)
    w, V = np.linalg.eigh((S + S.conj().T) / 2)
    return c * np.eye(len(S)) + (V * (np.log(w) / b)) @ V.conj().T


# ---------------------------------------------------------------- channels
def kraus_channel(U, p):
    """U followed by amplitude damping of strength p towards |0> (trace preserving)."""
    E = [np.zeros((N, N), complex) for _ in range(2)]
    E[0] = np.diag([1] + [np.sqrt(1 - p)] * (N - 1)).astype(complex)
    E[1][0, 1:] = np.sqrt(p)
    E = E[:1] + [np.outer(np.eye(N)[0], np.eye(N)[j]) * np.sqrt(p) for j in range(1, N)]
    Ks = [e @ U for e in E]
    assert np.allclose(sum(k.conj().T @ k for k in Ks), np.eye(N))
    return Ks

P = rng.dirichlet(np.ones(nS), size=(nS, K))
R = [[herm(N) for _ in range(K)] for _ in range(nS)]
# action a: with prob 1/2 channel C_a^1, with prob 1/2 channel C_a^2 (random latent dynamics)
KR = [[kraus_channel(unitary(N), 0.2), kraus_channel(unitary(N), 0.1)] for _ in range(K)]
fwd = lambda a, rho, w: sum(k @ rho @ k.conj().T for k in KR[a][w])
dual = lambda a, Y: 0.5 * sum(sum(k.conj().T @ Y @ k for k in KR[a][w]) for w in range(2))


def backward(H, agg):
    Z = np.zeros((N, N), complex)
    Q = [[[Z] * K for _ in range(nS)] for _ in range(H + 1)]
    for t in range(H - 1, -1, -1):
        V = [agg(Q[t + 1][s2]) if t + 1 < H else Z for s2 in range(nS)]
        Q[t] = [[dual(a, R[s][a] + g * sum(P[s, a, s2] * V[s2] for s2 in range(nS))) for a in range(K)]
                for s in range(nS)]
    return Q


def q_star(t, H, s, rho, a):
    """Optimal latent-aware value (latent observed after the random channel choice)."""
    tot = 0.0
    for w in range(2):
        rho2 = fwd(a, rho, w)
        v = tau(R[s][a] @ rho2)
        if t + 1 < H:
            v += g * sum(P[s, a, s2] * max(q_star(t + 1, H, s2, rho2, a2) for a2 in range(K)) for s2 in range(nS))
        tot += 0.5 * v
    return tot


def q_pi(t, H, s, rho, a, pi):
    tot = 0.0
    for w in range(2):
        rho2 = fwd(a, rho, w)
        v = tau(R[s][a] @ rho2)
        if t + 1 < H:
            v += g * sum(P[s, a, s2] * sum(pi[s2, a2] * q_pi(t + 1, H, s2, rho2, a2, pi) for a2 in range(K))
                         for s2 in range(nS))
        tot += 0.5 * v
    return tot


def rand_rho():
    G = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N)); r = G @ G.conj().T
    return N * r / np.real(np.trace(r))

H = 4
pi = rng.dirichlet(np.ones(K), size=nS)
Qpi = [[[None] * K for _ in range(nS)] for _ in range(H + 1)]
Z = np.zeros((N, N), complex)
for t in range(H - 1, -1, -1):
    V = [sum(pi[s2, a2] * Qpi[t + 1][s2][a2] for a2 in range(K)) if t + 1 < H else Z for s2 in range(nS)]
    for s in range(nS):
        for a in range(K):
            Qpi[t][s][a] = dual(a, R[s][a] + g * sum(P[s, a, s2] * V[s2] for s2 in range(nS)))
err = max(abs(q_pi(0, H, s, r, a, pi) - tau(r @ Qpi[0][s][a]))
          for s in range(nS) for a in range(K) for r in [rand_rho() for _ in range(3)])
print(f"(1) noisy random channels, Thm 1: max |q_pi - tau(rho Q_pi)| = {err:.2e}")
for b in [2.0, 10.0]:
    Qb = backward(H, lambda qs: lse(qs, b))
    viol, slack = 0, []
    for _ in range(20):
        s, a, r = rng.integers(nS), rng.integers(K), rand_rho()
        d = tau(r @ Qb[0][s][a]) - q_star(0, H, s, r, a)
        viol += d < -1e-10; slack.append(d)
    print(f"(1) noisy random channels, Thm 3, beta={b:g}: {viol}/20 violations, mean slack {np.mean(slack):.3f}")

# ---------------------------------------------------------------- Cor. 2 (unitary CP, infinite horizon)
U = [unitary(N) for _ in range(K)]
fwdU = lambda a, rho: U[a] @ rho @ U[a].conj().T
Rmax = max(np.linalg.norm(R[s][a], 2) for s in range(nS) for a in range(K))
pi = np.ones((nS, K)) / K
# infinite-horizon value of the uniform policy via Thm 1 (fixed point) and certificate by truncation
Qinf = [[np.zeros((N, N), complex)] * K for _ in range(nS)]
for _ in range(600):
    V = [sum(Qinf[s2][a2] for a2 in range(K)) / K for s2 in range(nS)]
    Qinf = [[U[a].conj().T @ (R[s][a] + g * sum(P[s, a, s2] * V[s2] for s2 in range(nS))) @ U[a]
             for a in range(K)] for s in range(nS)]
b = 5.0
Qn = [[np.zeros((N, N), complex)] * K for _ in range(nS)]
res = []
for n in range(1, 61):
    V = [lse(Qn[s2], b) if n > 1 else np.zeros((N, N)) for s2 in range(nS)]
    Qn = [[U[a].conj().T @ (R[s][a] + g * sum(P[s, a, s2] * V[s2] for s2 in range(nS))) @ U[a]
           for a in range(K)] for s in range(nS)]
    if n in (5, 10, 20, 40, 60):
        tail = Rmax * g ** n / (1 - g)
        worst = min(tau(r @ Qn[s][a]) + tail - tau(r @ Qinf[s][a])
                    for s in range(nS) for a in range(K) for r in [rand_rho() for _ in range(5)])
        res.append((n, worst))
print("(2) Cor 2, min over (s,a,rho) of [truncated certificate + tail - q_pi] by n: "
      + ", ".join(f"n={n}: {w:+.3f}" for n, w in res))

# ---------------------------------------------------------------- Prop. 6 (robust greedy improvement)
eps = 0.05
Qhat = [[Qinf[s][a] + eps * (lambda E: E / np.linalg.norm(E, 2))(herm(N)) for a in range(K)] for s in range(nS)]
def value_greedy(Qg, s, rho, depth):
    if depth == 0:
        return 0.0
    a = int(np.argmax([tau(rho @ q) for q in Qg[s]]))
    rho2 = fwdU(a, rho)
    return tau(R[s][a] @ rho2) + g * sum(P[s, a, s2] * value_greedy(Qg, s2, rho2, depth - 1) for s2 in range(nS))
D = 14                              # truncation; remaining tail <= Rmax g^D/(1-g) is subtracted
worst = np.inf
for _ in range(10):
    s, r = rng.integers(nS), rand_rho()
    Vpi = np.mean([tau(r @ Qinf[s][a]) for a in range(K)])
    worst = min(worst, value_greedy(Qhat, s, r, D) - Rmax * g ** D / (1 - g) - (Vpi - 2 * eps / (1 - g)))
print(f"(3) Prop 6, eps={eps}: min [V_greedy(Q_hat) - (V_pi - 2 eps/(1-gamma))] = {worst:+.3f} (>= 0 required)")

# ---------------------------------------------------------------- Prop. 8 (structure-refined dimension)
def hankel_rank(A, W, L=3):
    words = [w for l in range(L + 1) for w in itertools.product(range(len(A)), repeat=l)]
    Aw = {w: reduce(np.matmul, [A[i] for i in w], np.eye(len(W))) for w in words}
    Hm = np.array([[np.trace(W @ Aw[u] @ Aw[v]) for v in words] for u in words])
    s = np.linalg.svd(Hm, compute_uv=False)
    return int(np.sum(s > 1e-9 * s[0]))
# A = diag(B, C, C): blocks n = (2, 1) with multiplicities m = (1, 2); N = 4, dim alg = 4 + 1 = 5
A = []
for _ in range(3):
    B, c = unitary(2), np.exp(1j * rng.uniform(0, 2 * np.pi))
    A.append(np.block([[B, np.zeros((2, 2))], [np.zeros((2, 2)), c * np.eye(2)]]))
W = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
print(f"(4) Prop 8: reducible unitary rep, N=4, blocks n=(2,1), m=(1,2): Hankel rank {hankel_rank(A, W)} "
      f"(dim of algebra = 5); bounds sqrt(5)=2.24 <= d_min <= 3 < N = 4")
A2 = [unitary(3) for _ in range(3)]
print(f"(4) Prop 8: irreducible (generic) unitary rep, N=3: Hankel rank {hankel_rank(A2, rng.normal(size=(3, 3)))} (= N^2)")
# explicit d=3 model for the reducible case: Q_a = B_a (+) c_a, readout = compressed W
Wt = np.zeros((3, 3), complex); Wt[:2, :2] = W[:2, :2]; Wt[2, 2] = W[2, 2] + W[3, 3]
Q3 = [np.block([[a_[:2, :2], np.zeros((2, 1))], [np.zeros((1, 2)), a_[2:3, 2:3]]]) for a_ in A]
errs = []
for w in itertools.product(range(3), repeat=5):
    X4 = reduce(np.matmul, [A[i] for i in w]); X3 = reduce(np.matmul, [Q3[i] for i in w])
    errs.append(abs(np.trace(W @ X4) - np.trace(Wt @ X3)))
print(f"(4) Prop 8: explicit d=3 model reproduces all 243 words of length 5 to {max(errs):.1e}")
