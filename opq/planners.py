"""Planners that use exactly the knowledge OpQ uses (E19).

OpQ knows the action operators K_a (their duals K_a^dag) and learns the reward observable
R(m, a) from scalar rewards and the mode kernel P(m'|m, a) from sampled transitions.
`LearnedModel` collects the same statistics.  The planners below use that model:

  expectimax_q / ExhaustivePolicy : exact expectimax over all K^(H-t) action sequences,
        vectorised.  With the true (R, P) it attains the optimum V*; with the learned
        (R_hat, P_hat) it is the best a model-based agent can do from the same data.
  MCTS : UCT with random rollouts and a fixed simulation budget per decision.
  evaluate_mc : Monte-Carlo policy evaluation with common random numbers over mode paths.
"""
import time
import numpy as np
from .agents import RewardModel


# ----------------------------------------------------------------------------- learned model
class LearnedModel:
    """R_hat(m, a) by least squares from scalar rewards, P_hat(.|m, a) from counts.
    K_a is known (the environment's operators)."""

    def __init__(self, env, known_R=False, known_P=False):
        self.env, self.known_P = env, known_P
        self.Rm = RewardModel(env, known_R)
        self.cnt = np.zeros((env.nM, env.K, env.nM))

    def update(self, m, a, rho2, r, m2):
        self.Rm.update(m, a, rho2, r)
        self.cnt[m, a, m2] += 1

    def P(self):
        if self.known_P:
            return np.asarray(self.env.P)
        c = self.cnt + 1e-3                      # unseen (m, a) -> uniform row
        return c / c.sum(axis=2, keepdims=True)

    def R(self):
        """array (nM, K, N, N) of reward observables."""
        e = self.env
        return np.array([[self.Rm(m, a) for a in range(e.K)] for m in range(e.nM)])


# ----------------------------------------------------------------------------- exhaustive
def _latents(U, rho, depth):
    N = rho.shape[0]
    out = [rho[None]]
    Uc = np.stack(U)
    for _ in range(depth):
        out.append(np.einsum('aij,hjk,alk->hail', Uc, out[-1], Uc.conj()).reshape(-1, N, N))
    return out


def expectimax_q(env, t, rho, Rst, P):
    """Exact expectimax from (t, rho) under the model (K_a = env.U, reward observables
    Rst[m][a], mode kernel P).  Returns Q_root of shape (K, nM): the value of taking action a
    now at mode m and acting optimally afterwards.  Cost O(K^(H-t))."""
    K, nM, N, g = env.K, env.nM, env.N, env.gamma
    depth = env.H - t
    lat = _latents(env.U, rho, depth)
    V = np.zeros((K ** depth, nM))
    Q = None
    for d in range(depth - 1, -1, -1):
        rho_next = lat[d + 1]
        Q = np.zeros((K ** d, K, nM))
        for m in range(nM):
            for a in range(K):
                r = np.real(np.einsum('ij,hji->h', Rst[m][a], rho_next[a::K])) / N
                Q[:, a, m] = r + g * (V[a::K] @ P[m, a])
        V = Q.max(axis=1)
    return Q[0]


class ExhaustivePolicy:
    """Greedy w.r.t. exhaustive expectimax under a model.  Results are cached per (t, rho)."""

    def __init__(self, env, Rst=None, P=None):
        self.env = env
        self.Rst = env.R if Rst is None else Rst
        self.P = np.asarray(env.P) if P is None else P
        self.cache = {}
        self.nodes = 0

    def q(self, t, rho):
        key = (t, rho.tobytes())
        if key not in self.cache:
            self.cache[key] = expectimax_q(self.env, t, rho, self.Rst, self.P)
            self.nodes += sum(self.env.K ** d for d in range(1, self.env.H - t + 1))
        return self.cache[key]

    def __call__(self, t, m, rho, hist):
        return int(np.argmax(self.q(t, rho)[:, m]))


# ----------------------------------------------------------------------------- MCTS
class _Node:
    __slots__ = ("t", "m", "rho", "n", "na", "wa", "ch")

    def __init__(self, t, m, rho, K):
        self.t, self.m, self.rho = t, m, rho
        self.n = 0
        self.na = np.zeros(K)
        self.wa = np.zeros(K)
        self.ch = {}


class MCTS:
    """UCT over (action, next-mode) chance nodes with uniform-random rollouts.
    Uses the same model as OpQ: known K_a, reward observables Rst, mode kernel P."""

    def __init__(self, env, Rst, P, n_sims=200, c=1.0, seed=0):
        self.env, self.Rst, self.P = env, Rst, np.asarray(P)
        self.cdf = np.cumsum(self.P, axis=2)
        self.Ud = [u.conj().T for u in env.U]
        self.N = env.N
        self.S, self.c = n_sims, c
        self.rng = np.random.default_rng(seed)
        self.sims = 0

    def _step(self, m, rho, a):
        rho2 = self.env.U[a] @ rho @ self.Ud[a]
        r = float(np.real(np.sum(self.Rst[m][a].T * rho2))) / self.N   # Re Tr(R rho') / N
        return r, rho2

    def _next_mode(self, m, a):
        return int(min(np.searchsorted(self.cdf[m, a], self.rng.random(), side='right'),
                       self.env.nM - 1))

    def _rollout(self, t, m, rho):
        g, tot, disc = self.env.gamma, 0.0, 1.0
        acts = self.rng.integers(self.env.K, size=self.env.H - t)
        for a in acts:
            r, rho = self._step(m, rho, int(a))
            tot += disc * r
            disc *= g
            m = self._next_mode(m, int(a))
        return tot

    def search(self, t, m, rho):
        env, K, g = self.env, self.env.K, self.env.gamma
        root = _Node(t, m, rho, K)
        lo, hi = np.inf, -np.inf
        for _ in range(self.S):
            node, path = root, []
            while True:
                if node.t == env.H:
                    G = 0.0
                    break
                untried = np.flatnonzero(node.na == 0)
                if len(untried):
                    a = int(self.rng.choice(untried))
                else:
                    q = node.wa / node.na
                    span = (hi - lo) if hi > lo else 1.0
                    ucb = (q - lo) / span + self.c * np.sqrt(np.log(node.n) / node.na)
                    a = int(np.argmax(ucb))
                r, rho2 = self._step(node.m, node.rho, a)
                m2 = self._next_mode(node.m, a)
                path.append((node, a, r))
                child = node.ch.get((a, m2))
                if child is None:
                    child = _Node(node.t + 1, m2, rho2, K)
                    node.ch[(a, m2)] = child
                    G = self._rollout(child.t, m2, rho2) if child.t < env.H else 0.0
                    break
                node = child
            for nd, a, r in reversed(path):
                G = r + g * G
                nd.n += 1
                nd.na[a] += 1
                nd.wa[a] += G
                lo, hi = min(lo, G), max(hi, G)
            self.sims += 1
        return int(np.argmax(root.na + 1e-9 * root.wa / np.maximum(root.na, 1)))


class MCTSPolicy:
    def __init__(self, env, Rst, P, n_sims, seed=0, c=1.0):
        self.mcts = MCTS(env, Rst, P, n_sims, c, seed)

    def __call__(self, t, m, rho, hist):
        return self.mcts.search(t, m, rho)


# ----------------------------------------------------------------------------- evaluation
def is_deterministic_modes(env):
    P = np.asarray(env.P)
    return bool(np.all(P.max(axis=2) > 1 - 1e-12) and env.p0.max() > 1 - 1e-12)


def evaluate_mc(env, policy, rho0, n_paths=64, seed=0):
    """Expected return of `policy` from rho0 by Monte Carlo over mode paths (mean rewards are
    used, so the only randomness is the observable mode chain).  Mode uniforms depend only on
    (seed, path index), so different policies see the same mode noise (common random numbers).
    Returns the per-path returns, shape (n_paths,)."""
    H, g, nM = env.H, env.gamma, env.nM
    cdf0, cdf = np.cumsum(env.p0), np.cumsum(env.P, axis=2)
    if is_deterministic_modes(env):
        n_paths = 1
    out = np.zeros(n_paths)
    for i in range(n_paths):
        u = np.random.default_rng(seed * 100003 + i).random(H + 1)
        m = int(min(np.searchsorted(cdf0, u[0], side='right'), nM - 1))
        rho, hist, tot, disc = rho0, (), 0.0, 1.0
        for t in range(H):
            a = int(policy(t, m, rho, hist))
            rho2 = env.fwd(a, rho)
            r = env.mean_reward(m, a, rho2)
            tot += disc * r
            disc *= g
            m2 = int(min(np.searchsorted(cdf[m, a], u[t + 1], side='right'), nM - 1))
            hist = hist + ((a, m2, r),)
            rho, m = rho2, m2
        out[i] = tot
    return out


def decision_time(policy, env, rho, m=0, reps=5):
    """Wall-clock seconds of one decision at t = 0 (median of `reps`)."""
    ts = []
    for _ in range(reps):
        if hasattr(policy, "cache"):
            policy.cache.clear()                  # time a cold decision
        t0 = time.perf_counter()
        policy(0, m, rho, ())
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))
