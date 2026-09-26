"""Sanity checks for the compositional operator Bellman recursion (revised Sec. 3).
All numbers printed here are the ones quoted in the revised paper (Sec. 6, 'E17')."""
import numpy as np, itertools
from scipy.linalg import expm, eigh
rng = np.random.default_rng(0)
tau = lambda X: np.trace(X).real / X.shape[0]

def fn(A, f):
    w, V = eigh((A + A.conj().T) / 2); return (V * f(w)) @ V.conj().T
import mpmath as mp
mp.mp.dps = 120
def LSE(xs, b):
    """operator log-sum-exp in extended precision (float64 underflows for large beta)"""
    S = None
    for x in xs:
        w, V = eigh((x + x.T)/2)
        E = mp.matrix(V.tolist()) * mp.diag([mp.exp(b*mp.mpf(float(v))) for v in w]) * mp.matrix(V.T.tolist())
        S = E if S is None else S + E
    ev, EV = mp.eigsy(S)
    L = EV * mp.diag([mp.log(e)/b for e in ev]) * EV.T
    return np.array([[float(L[i, j]) for j in range(L.cols)] for i in range(L.rows)])
def rand_skew(n, s):
    A = rng.normal(size=(n, n)); return s*(A - A.T)
def rand_sym(n):
    A = rng.normal(size=(n, n)); return (A + A.T)/2
def rand_density(n):
    G = rng.normal(size=(n, n)); r = G@G.T; return n*r/np.trace(r)   # tau(rho)=1

# ---------------- environment: observable finite MDP x latent operator ---------------
nS, K, N, H, gam = 2, 3, 3, 4, 0.9
P = rng.dirichlet(np.ones(nS), size=(nS, K))                 # P[s,a,s']
U = [expm(rand_skew(N, 0.8)) for _ in range(K)]              # real orthogonal actions
R = [[rand_sym(N) for _ in range(K)] for _ in range(nS)]     # reward observables

# conjugation picture: rho' = U rho U^T, reward tau(R(s,a) rho'); dual map K^dag(Y)=U^T Y U
Kc  = lambda a, rho: U[a] @ rho @ U[a].T
Kcd = lambda a, Y:   U[a].T @ Y @ U[a]
# multiplicative picture: X' = X A, reward tau(R X'); dual K^dag(Y) = A Y
Km  = lambda a, X: X @ U[a]
Kmd = lambda a, Y: U[a] @ Y

def op_eval(pi, Kd, H):
    """finite-horizon operator policy evaluation, Q[t][s][a]; pi[s] a prob. vector"""
    Q = [[[np.zeros((N, N)) for a in range(K)] for s in range(nS)] for t in range(H+1)]
    for t in range(H-1, -1, -1):
        for s in range(nS):
            for a in range(K):
                cont = sum(P[s, a, s2]*sum(pi[s2][a2]*Q[t+1][s2][a2] for a2 in range(K)) for s2 in range(nS))
                Q[t][s][a] = Kd(a, R[s][a] + gam*cont)
    return Q

def brute_eval(pi, Kfwd, s, rho, a, t, H):
    """exact expectation by enumerating all branches"""
    rho2 = Kfwd(a, rho); r = tau(R[s][a] @ rho2)
    if t == H-1: return r
    return r + gam*sum(P[s, a, s2]*pi[s2][a2]*brute_eval(pi, Kfwd, s2, rho2, a2, t+1, H)
                       for s2 in range(nS) for a2 in range(K))

pi = [rng.dirichlet(np.ones(K)) for _ in range(nS)]
err = 0
for (Kf, Kd, name) in [(Kc, Kcd, "conjugation"), (Km, Kmd, "multiplicative")]:
    Q = op_eval(pi, Kd, H)
    e = 0
    for _ in range(20):
        s, a = rng.integers(nS), rng.integers(K); rho = rand_density(N) if name=="conjugation" else expm(rand_skew(N, .5))
        e = max(e, abs(tau(rho @ Q[0][s][a]) - brute_eval(pi, Kf, s, rho, a, 0, H)))
    print(f"[Thm 1] {name}: max |tau(rho Q^pi) - exact expected return| = {e:.2e}")

# ---------------- Loewner certificate (conjugation picture) ---------------------------
def soft_op(Kd, H, b):
    Q = [[[np.zeros((N, N)) for a in range(K)] for s in range(nS)] for t in range(H+1)]
    for t in range(H-1, -1, -1):
        V = [LSE(Q[t+1][s2], b) if t+1 < H else np.zeros((N, N)) for s2 in range(nS)]
        for s in range(nS):
            for a in range(K):
                Q[t][s][a] = Kd(a, R[s][a] + gam*sum(P[s, a, s2]*V[s2] for s2 in range(nS)))
    return Q
def opt_latent_aware(s, rho, a, t, H):
    rho2 = Kc(a, rho); r = tau(R[s][a] @ rho2)
    if t == H-1: return r
    return r + gam*sum(P[s, a, s2]*max(opt_latent_aware(s2, rho2, a2, t+1, H) for a2 in range(K)) for s2 in range(nS))
for b in [1.0, 5.0, 20.0, 50.0]:
    Qb = soft_op(Kcd, H, b); viol = 0; gaps = []
    for _ in range(30):
        s, a = rng.integers(nS), rng.integers(K); rho = rand_density(N)
        ub, qs = tau(rho @ Qb[0][s][a]), opt_latent_aware(s, rho, a, 0, H)
        viol += ub < qs - 1e-9; gaps.append(ub - qs)
    # Loewner check against Q^pi for random latent-blind pi
    lo = min(np.linalg.eigvalsh(Qb[0][s][a] - op_eval([rng.dirichlet(np.ones(K)) for _ in range(nS)], Kcd, H)[0][s][a]).min()
             for s in range(nS) for a in range(K) for _ in range(5))
    print(f"[Thm 3] beta={b:6.1f}: certificate violations {viol}/30, mean slack {np.mean(gaps):.3f}, "
          f"min eig(Q_beta - Q^pi) = {lo:.2e}")

# commuting case: diagonal actions -> slack at pure eigenstates vanishes as beta->inf
Ud = [np.diag(np.sign(rng.normal(size=N))) for _ in range(K)]
Rd = [[np.diag(rng.normal(size=N)) for _ in range(K)] for _ in range(nS)]
U_save, R_save = U, R; U, R = Ud, Rd
Qb = soft_op(Kcd, H, 200.0)
e = max(abs(tau(np.diag(np.eye(N)[j])*N @ Qb[0][s][a]) - opt_latent_aware(s, np.diag(np.eye(N)[j])*N, a, 0, H))
        for s in range(nS) for a in range(K) for j in range(N))
print(f"[Prop 2] commuting, pure eigenstates, beta=200: max slack {e:.2e}  (<= gamma*log K/beta*H scale {H*np.log(K)/200:.2e})")
U, R = U_save, R_save

# ---------------- aliasing lower bound (Thm 4 revised), T=3 --------------------------
A = [expm(rand_skew(N, 0.6)) for _ in range(3)]; W = rng.normal(size=(N, N))
f = lambda w: tau(W @ np.linalg.multi_dot([A[i] for i in w]))
print(f"[Thm 4] r(a,b,c)={f((0,1,2)):.4f}, r(b,a,c)={f((1,0,2)):.4f}, "
      f"lower bound on any order-blind scalar Q: {abs(f((0,1,2))-f((1,0,2)))/2:.4f}")

# ---------------- Hankel rank (Thm 6 revised) ----------------------------------------
Kw, Nw = 3, 3
A = [expm(rand_skew(Nw, 0.6)) for _ in range(Kw)]; W = rng.normal(size=(Nw, Nw))
words = [w for L in range(0, 4) for w in itertools.product(range(Kw), repeat=L)]
mat = lambda w: np.linalg.multi_dot([np.eye(Nw)] + [A[i] for i in w] + [np.eye(Nw)])
Hk = np.array([[tau(W @ mat(u) @ mat(v)) for v in words] for u in words])
sv = np.linalg.svd(Hk, compute_uv=False)
print(f"[Thm 6] Hankel block {Hk.shape}, numerical rank {np.sum(sv > 1e-9*sv[0])} (N^2 = {Nw*Nw})")

# ---------------- operator TD(0) convergence (Thm 2 revised) -------------------------
Qstar = None
def op_eval_inf(pi, Kd, iters=400):
    Q = [[np.zeros((N, N)) for a in range(K)] for s in range(nS)]
    for _ in range(iters):
        Q = [[Kd(a, R[s][a] + gam*sum(P[s, a, s2]*sum(pi[s2][a2]*Q[s2][a2] for a2 in range(K)) for s2 in range(nS)))
              for a in range(K)] for s in range(nS)]
    return Q
Qstar = op_eval_inf(pi, Kcd)
Q = [[np.zeros((N, N)) for a in range(K)] for s in range(nS)]; cnt = np.zeros((nS, K))
for k in range(1, 400001):
    s, a = rng.integers(nS), rng.integers(K)                      # i.i.d. (s,a) sampling
    s2 = rng.choice(nS, p=P[s, a]); a2 = rng.choice(K, p=pi[s2])  # sampled transition
    al = 1.0/(1.0 + k/200.0)**0.7            # global Robbins-Monro step (as in Thm 2)
    Q[s][a] = Q[s][a] + al*(Kcd(a, R[s][a] + gam*Q[s2][a2]) - Q[s][a])
    if k in (1000, 10000, 100000, 400000):
        print(f"[Thm 2] operator TD, step {k:6d}: max_(s,a) ||Q-Q^pi||_op = "
              f"{max(np.linalg.norm(Q[s][a]-Qstar[s][a], 2) for s in range(nS) for a in range(K)):.4f}")

# scale of optimal values in the certificate instance (context for the slack)
rng2 = np.random.default_rng(5)
vals = [abs(opt_latent_aware(rng2.integers(nS), rand_density(N), rng2.integers(K), 0, H)) for _ in range(30)]
print(f"[Thm 3] context: mean |q*| over 30 random (s,rho,a) = {np.mean(vals):.3f}")
