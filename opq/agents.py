"""Agents for end-to-end control on OLDPs (experiment E18).

Operator agents (tabular over the observable state s = (t, m)):
  OpQSoft  : sample-based soft operator Q-learning, the stochastic-approximation
             form of the soft recursion (Eq. 4); acts greedily on tau(rho Q(s, a)).
  OpQPI    : operator TD evaluation (Eq. 5) of the uniform policy, then latent-aware
             greedy improvement (Prop. 1).
Both use the known dual maps K_a^dag.  The reward observable R(m, a) is either
known or estimated online by least squares from the scalar rewards
(r = tau(R rho') is linear in rho').

Scalar baselines:
  TabularQ     : Q-learning on the order-blind observable state (t, m)
  HistoryQ     : Q-learning on the full history (t, m, a_1 m_1 ... a_t m_t) -- the K^T table
  DQN          : MLP Q-network on (t, m, rho_t)      (latent observed, JAX)
  DRQN         : GRU Q-network on the history       (JAX)
"""
import numpy as np
from .core import tau, lse, hermitian_basis, herm_features


# ------------------------------------------------------------------------- exact operator DP
def op_policy_eval_exact(env, pi):
    """Finite-horizon operator policy evaluation, Eq. (3); pi(t, m) -> prob vector."""
    H, K, nM, N = env.H, env.K, env.nM, env.N
    Z = np.zeros((N, N), dtype=complex if env.is_complex else float)
    Q = [[[Z.copy() for _ in range(K)] for _ in range(nM)] for _ in range(H + 1)]
    for t in range(H - 1, -1, -1):
        V = [sum(pi(t + 1, m2)[a2] * Q[t + 1][m2][a2] for a2 in range(K)) if t + 1 < H else Z
             for m2 in range(nM)]
        for m in range(nM):
            for a in range(K):
                cont = sum(env.P[m, a, m2] * V[m2] for m2 in range(nM))
                Q[t][m][a] = env.dual(a, env.R[m][a] + env.gamma * cont)
    return Q


def op_soft_exact(env, beta):
    """Finite-horizon soft operator recursion, Eq. (4) / Thm. 3."""
    H, K, nM, N = env.H, env.K, env.nM, env.N
    Z = np.zeros((N, N), dtype=complex if env.is_complex else float)
    Q = [[[Z.copy() for _ in range(K)] for _ in range(nM)] for _ in range(H + 1)]
    for t in range(H - 1, -1, -1):
        V = [lse(Q[t + 1][m2], beta) if t + 1 < H else Z for m2 in range(nM)]
        for m in range(nM):
            for a in range(K):
                cont = sum(env.P[m, a, m2] * V[m2] for m2 in range(nM))
                Q[t][m][a] = env.dual(a, env.R[m][a] + env.gamma * cont)
    return Q


def greedy_from_ops(Q):
    def pol(t, m, rho, hist):
        return int(np.argmax([tau(rho @ q) for q in Q[t][m]]))
    return pol


# ------------------------------------------------------------------------- reward model
class RewardModel:
    """Least-squares estimate of R(m, a) from (rho', r) pairs (ridge 1e-6)."""

    def __init__(self, env, known=False):
        self.env, self.known = env, known
        self.basis = hermitian_basis(env.N, env.is_complex)
        d = len(self.basis)
        self.A = np.tile(1e-6 * np.eye(d), (env.nM, env.K, 1, 1))
        self.b = np.zeros((env.nM, env.K, d))
        self.cache = {}

    def update(self, m, a, rho2, r):
        if self.known:
            return
        phi = herm_features(rho2, self.basis) / self.env.N        # r = tau(R rho') = <coef, phi>
        self.A[m, a] += np.outer(phi, phi)
        self.b[m, a] += r * phi
        self.cache.pop((m, a), None)

    def __call__(self, m, a):
        if self.known:
            return self.env.R[m][a]
        if (m, a) not in self.cache:
            c = np.linalg.solve(self.A[m, a], self.b[m, a])
            self.cache[(m, a)] = sum(ci * B for ci, B in zip(c, self.basis))
        return self.cache[(m, a)]


# ------------------------------------------------------------------------- operator agents
class OpQSoft:
    name = "OpQ-soft"

    def __init__(self, env, beta=5.0, known_R=False, lr_pow=0.6, lr_min=0.02, seed=0):
        self.env, self.beta = env, beta
        self.Rm = RewardModel(env, known_R)
        dt = complex if env.is_complex else float
        self.Q = [[[np.zeros((env.N, env.N), dtype=dt) for _ in range(env.K)]
                   for _ in range(env.nM)] for _ in range(env.H + 1)]
        self.n = np.zeros((env.H, env.nM, env.K))
        self.lr_pow, self.lr_min = lr_pow, lr_min
        if known_R:
            self.name = "OpQ-soft (known R)"

    def act(self, t, m, rho, hist, eps, rng):
        if rng.random() < eps:
            return int(rng.integers(self.env.K))
        return int(np.argmax([tau(rho @ q) for q in self.Q[t][m]]))

    def observe(self, t, m, rho, a, r, t2, m2, rho2, done, hist):
        env = self.env
        self.Rm.update(m, a, rho2, r)
        V = lse(self.Q[t2][m2], self.beta) if not done else 0.0
        target = env.dual(a, self.Rm(m, a) + env.gamma * V)
        self.n[t, m, a] += 1
        al = max(self.lr_min, self.n[t, m, a] ** -self.lr_pow)
        self.Q[t][m][a] = self.Q[t][m][a] + al * (target - self.Q[t][m][a])

    def end_episode(self):
        pass

    def policy(self):
        return greedy_from_ops(self.Q)


class OpQPI(OpQSoft):
    """Operator TD evaluation of the uniform policy (behaviour = uniform), then greedy."""
    name = "OpQ-PI"

    def __init__(self, env, known_R=False, **kw):
        super().__init__(env, known_R=known_R, **kw)
        self.name = "OpQ-PI" + (" (known R)" if known_R else "")

    def act(self, t, m, rho, hist, eps, rng):
        return int(rng.integers(self.env.K))                      # on-policy for pi = uniform

    def observe(self, t, m, rho, a, r, t2, m2, rho2, done, hist):
        env = self.env
        self.Rm.update(m, a, rho2, r)
        V = np.mean(self.Q[t2][m2], axis=0) if not done else 0.0  # E_{a'~uniform} Q(s', a')
        target = env.dual(a, self.Rm(m, a) + env.gamma * V)
        self.n[t, m, a] += 1
        al = max(self.lr_min, self.n[t, m, a] ** -self.lr_pow)
        self.Q[t][m][a] = self.Q[t][m][a] + al * (target - self.Q[t][m][a])


# ------------------------------------------------------------------------- scalar tabular
class TabularQ:
    """Scalar Q-learning on a discrete key; key_fn(t, m, hist) -> hashable."""

    def __init__(self, env, key="state", lr_pow=0.6, lr_min=0.02, seed=0):
        self.env, self.keyname = env, key
        self.Q, self.n = {}, {}
        self.lr_pow, self.lr_min = lr_pow, lr_min
        self.name = {"state": "Scalar Q (t, m)", "history": "Scalar Q (history table)"}[key]

    def key(self, t, m, hist):
        if self.keyname == "state":
            return (t, m)
        return (t, m) + tuple((a, mm) for a, mm, _ in hist)

    def q(self, k):
        return self.Q.setdefault(k, np.zeros(self.env.K))

    def act(self, t, m, rho, hist, eps, rng):
        if rng.random() < eps:
            return int(rng.integers(self.env.K))
        q = self.q(self.key(t, m, hist))
        return int(rng.choice(np.flatnonzero(q == q.max())))

    def observe(self, t, m, rho, a, r, t2, m2, rho2, done, hist):
        k = self.key(t, m, hist)
        k2 = self.key(t2, m2, hist + ((a, m2, r),))
        target = r + (0.0 if done else self.env.gamma * self.q(k2).max())
        c = self.n.setdefault(k, np.zeros(self.env.K))
        c[a] += 1
        al = max(self.lr_min, c[a] ** -self.lr_pow)
        self.q(k)[a] += al * (target - self.q(k)[a])

    def end_episode(self):
        pass

    def policy(self):
        def pol(t, m, rho, hist):
            q = self.Q.get(self.key(t, m, hist))
            return 0 if q is None else int(np.argmax(q))
        return pol
