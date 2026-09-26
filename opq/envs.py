"""Operator-latent decision processes (OLDPs) in the conjugation picture (CP).

Observable state s_t = (t, m_t): the step counter and an observable mode m_t
whose transitions P(m'|m,a) do not depend on the latent.  The latent is a
density operator rho_t with rho_{t+1} = U_a rho_t U_a^*, and the reward is
r_t = tau(R(m_t, a_t) rho_{t+1}) (+ optional Gaussian noise).

Three environments are provided (experiment E18 of the paper):
  random_oldp  : random orthogonal actions, two stochastic modes (C1)
  pointing     : SO(3) attitude pointing of a body axis, rho = 3 n n^T (C2)
  qubit_prep   : two-qubit state preparation with {H1, H2, T1, T2, CNOT} (C3)
"""
import numpy as np
from scipy.linalg import expm
from .core import tau, rand_orth, rand_sym, rand_density


class OLDP:
    def __init__(self, U, R, P, rho0, p0, H, gamma, noise=0.0, name="oldp"):
        self.U = [np.asarray(u) for u in U]
        self.R = R                      # R[m][a], Hermitian
        self.P = np.asarray(P)          # P[m, a, m']
        self.rho0 = rho0
        self.p0 = np.asarray(p0)
        self.H, self.gamma, self.noise, self.name = H, gamma, noise, name
        self.K, self.N, self.nM = len(U), U[0].shape[0], self.P.shape[0]
        self.is_complex = any(np.iscomplexobj(u) for u in U) or np.iscomplexobj(rho0)
        self.random_init = False          # if True, rho0 ~ random pure state at every reset

    def sample_rho0(self, rng):
        v = rng.normal(size=self.N) + (1j * rng.normal(size=self.N) if self.is_complex else 0)
        v = v / np.linalg.norm(v)
        return self.N * np.outer(v, v.conj())

    # dynamics -------------------------------------------------------------
    def fwd(self, a, rho):
        return self.U[a] @ rho @ self.U[a].conj().T

    def dual(self, a, Y):
        return self.U[a].conj().T @ Y @ self.U[a]

    def mean_reward(self, m, a, rho_next):
        return tau(self.R[m][a] @ rho_next)

    def reset(self, rng):
        rho = self.sample_rho0(rng) if self.random_init else self.rho0.copy()
        return 0, rng.choice(self.nM, p=self.p0), rho

    def step(self, t, m, rho, a, rng):
        rho2 = self.fwd(a, rho)
        r = self.mean_reward(m, a, rho2) + (self.noise * rng.normal() if self.noise else 0.0)
        m2 = rng.choice(self.nM, p=self.P[m, a])
        return r, t + 1, m2, rho2, (t + 1 == self.H)

    # exact planning ---------------------------------------------------------
    def all_latents(self, rho0=None):
        """rho for every action history, by depth: list of arrays (K^t, N, N); index h*K+a."""
        out = [(self.rho0 if rho0 is None else rho0)[None]]
        Uc = np.stack(self.U)
        for t in range(self.H):
            r = out[-1]
            nxt = np.einsum('aij,hjk,alk->hail', Uc, r, Uc.conj())
            out.append(nxt.reshape(-1, self.N, self.N))
        return out

    def optimal_values(self, rho0=None):
        """Exact latent-aware optimal values by backward DP over action histories.
        Returns V0[m] (value at t=0 for each initial mode) and the expected optimum."""
        lat = self.all_latents(rho0)
        H, K, nM = self.H, self.K, self.nM
        V = np.zeros((K ** H, nM))
        for t in range(H - 1, -1, -1):
            rho_next = lat[t + 1]                               # (K^t*K, N, N)
            nh = K ** t
            Q = np.zeros((nh, K, nM))
            for m in range(nM):
                for a in range(K):
                    r = np.real(np.einsum('ij,hji->h', self.R[m][a], rho_next[a::K])) / self.N
                    cont = V[a::K] @ self.P[m, a]                # (nh,)
                    Q[:, a, m] = r + self.gamma * cont
            V = Q.max(axis=1)
        return V[0], float(self.p0 @ V[0])

    def evaluate(self, policy, rho0=None):
        """Exact expected return of a deterministic policy
        policy(t, m, rho, hist) -> a, where hist is a tuple of (a, m', r) triples.
        Enumerates all mode branches (rewards enter with their mean)."""
        H, g = self.H, self.gamma

        def rec(t, m, rho, hist):
            if t == H:
                return 0.0
            a = policy(t, m, rho, hist)
            rho2 = self.fwd(a, rho)
            r = self.mean_reward(m, a, rho2)
            tot = r
            for m2 in range(self.nM):
                p = self.P[m, a, m2]
                if p > 0:
                    tot += g * p * rec(t + 1, m2, rho2, hist + ((a, m2, r),))
            return tot
        r0 = self.rho0 if rho0 is None else rho0
        return float(sum(self.p0[m] * rec(0, m, r0, ()) for m in range(self.nM) if self.p0[m] > 0))

    def random_value(self, rho0=None):
        """Expected return of the uniformly random policy (exact, via Thm. 1)."""
        from .agents import op_policy_eval_exact
        r0 = self.rho0 if rho0 is None else rho0
        Q = op_policy_eval_exact(self, lambda t, m: np.ones(self.K) / self.K)
        return float(sum(self.p0[m] * np.mean([tau(r0 @ Q[0][m][a]) for a in range(self.K)])
                         for m in range(self.nM)))


# -------------------------------------------------------------------------------------------
def random_oldp(seed=0, N=3, K=4, nM=2, H=6, gamma=0.95, scale=0.8, noise=0.1):
    rng = np.random.default_rng(seed)
    U = [rand_orth(rng, N, scale) for _ in range(K)]
    R = [[rand_sym(rng, N) for _ in range(K)] for _ in range(nM)]
    P = rng.dirichlet(np.ones(nM), size=(nM, K))
    rho0 = rand_density(rng, N)
    return OLDP(U, R, P, rho0, np.ones(nM) / nM, H, gamma, noise, "random_oldp")


def _rot(axis, th):
    L = np.zeros((3, 3))
    i, j = [(1, 2), (2, 0), (0, 1)][axis]
    L[i, j], L[j, i] = -1.0, 1.0
    return expm(th * L)


def pointing(seed=0, theta=0.6, H=8, gamma=0.95, noise=0.0):
    """SO(3) pointing: body axis n (rho = 3 n n^T), K = 6 rotations +-theta about x, y, z.
    Mode 0 = 'cruise' (reward alignment with target 1), mode 1 = 'slew' (target 2);
    the mode switches stochastically, independently of the latent."""
    rng = np.random.default_rng(seed)
    U = [_rot(ax, s * theta) for ax in range(3) for s in (+1, -1)]
    t1 = rng.normal(size=3); t1 /= np.linalg.norm(t1)
    t2 = rng.normal(size=3); t2 /= np.linalg.norm(t2)
    R = [[np.outer(t1, t1) for _ in range(6)], [np.outer(t2, t2) for _ in range(6)]]   # r = (t.n)^2
    P = np.zeros((2, 6, 2))
    P[0, :, :] = [0.8, 0.2]
    P[1, :, :] = [0.3, 0.7]
    n0 = np.array([0.0, 0.0, 1.0])
    return OLDP(U, R, P, 3 * np.outer(n0, n0), np.array([1.0, 0.0]), H, gamma, noise, "pointing")


def qubit_prep(seed=0, H=8, gamma=1.0, depth=8):
    """Two-qubit state preparation, gates {H1, H2, T1, T2, CNOT}; terminal reward is
    the fidelity with a target state reachable in `depth` gates; single mode.
    Reward observable: R = 0 before the last step and |psi><psi| at the last step.
    Time-dependent rewards are encoded with one mode per time step (deterministic P)."""
    rng = np.random.default_rng(seed)
    Hd = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    T = np.diag([1, np.exp(1j * np.pi / 4)])
    I2 = np.eye(2)
    CX = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
    U = [np.kron(Hd, I2).astype(complex), np.kron(I2, Hd).astype(complex),
         np.kron(T, I2), np.kron(I2, T), CX]
    K = len(U)
    psi0 = np.zeros(4, dtype=complex); psi0[0] = 1
    # target: a random gate word of the given depth, rejecting low-entanglement targets
    while True:
        w = rng.integers(K, size=depth)
        psi = psi0.copy()
        for a in w:
            psi = U[a] @ psi
        red = np.einsum('ij,kj->ik', psi.reshape(2, 2), psi.reshape(2, 2).conj())
        purity = np.real(np.trace(red @ red))
        if purity < 0.9 and abs(psi[0]) ** 2 < 0.5:
            break
    Rt = np.outer(psi, psi.conj())               # r = fidelity |<psi|phi>|^2
    nM = H + 1                                   # mode = step counter (deterministic)
    R = [[(Rt if m == H - 1 else np.zeros((4, 4), dtype=complex)) for _ in range(K)] for m in range(nM)]
    P = np.zeros((nM, K, nM))
    for m in range(nM):
        P[m, :, min(m + 1, nM - 1)] = 1.0
    p0 = np.zeros(nM); p0[0] = 1
    env = OLDP(U, R, P, 4 * np.outer(psi0, psi0.conj()), p0, H, gamma, 0.0, "qubit_prep")
    env.target_word = tuple(int(a) for a in w)
    return env


ENVS = {"random_oldp": random_oldp, "pointing": pointing, "qubit_prep": qubit_prep}
