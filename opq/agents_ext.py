"""Extra agents for E19: the 2x2 comparison and model-based planners.

                       operators K_a unknown          operators K_a known
  nonlinear in rho     DQN on rho_t (E18)             DQNAug  (DQN + transitions generated from K_a)
  linear in rho        LinearQ (below)                OpQ-soft (E18)

LinearQ    : Q(t, m, a; rho) = <theta_{t,m,a}, phi(rho)>, learned by TD from observed
             (rho_t, a, r, rho_{t+1}) only.  It has the same function class as OpQ (a linear
             readout of rho per (t, m, a)) but never uses K_a.
ModelExh   : learns (R_hat, P_hat) from uniformly random episodes, knows K_a, and plans by
             exhaustive expectimax.  Exactly what OpQ knows, plus full search at decision time.
"""
import numpy as np
from .core import hermitian_basis, herm_features
from .planners import LearnedModel, ExhaustivePolicy


class LinearQ:
    """Linear-in-rho Q-learning without the operators (normalised-LMS TD, step max(0.02, n^-0.6))."""

    def __init__(self, env, soft=True, beta=5.0, lr_pow=0.6, lr_min=0.02, seed=0):
        self.env, self.soft, self.beta = env, soft, beta
        self.basis = hermitian_basis(env.N, env.is_complex)
        self.d = len(self.basis)
        self.th = np.zeros((env.H + 1, env.nM, env.K, self.d))
        self.n = np.zeros((env.H + 1, env.nM, env.K))
        self.lr_pow, self.lr_min = lr_pow, lr_min
        self.name = "Linear-Q (soft)" if soft else "Linear-Q (max)"

    def phi(self, rho):
        return herm_features(rho, self.basis) / self.env.N          # ||phi|| = 1 for pure rho

    def qvals(self, t, m, phi):
        return self.th[t, m] @ phi

    def act(self, t, m, rho, hist, eps, rng):
        if rng.random() < eps:
            return int(rng.integers(self.env.K))
        return int(np.argmax(self.qvals(t, m, self.phi(rho))))

    def observe(self, t, m, rho, a, r, t2, m2, rho2, done, hist):
        phi = self.phi(rho)
        if done:
            v = 0.0
        else:
            q2 = self.qvals(t2, m2, self.phi(rho2))
            if self.soft:
                c = q2.max()
                v = c + np.log(np.exp(self.beta * (q2 - c)).sum()) / self.beta
            else:
                v = q2.max()
        y = r + self.env.gamma * v
        self.n[t, m, a] += 1
        al = max(self.lr_min, self.n[t, m, a] ** -self.lr_pow)
        err = y - self.th[t, m, a] @ phi
        self.th[t, m, a] += al * err * phi / (phi @ phi + 1e-8)

    def end_episode(self):
        pass

    def policy(self):
        th = self.th.copy()

        def pol(t, m, rho, hist):
            return int(np.argmax(th[t, m] @ self.phi(rho)))
        return pol


class ModelExh:
    """Model-based planner: K_a known, R_hat and P_hat learned, exhaustive expectimax.
    Behaviour is uniformly random (model learning does not need a policy)."""
    name = "Model + exhaustive"

    def __init__(self, env, known_R=False, known_P=False, seed=0):
        self.env = env
        self.model = LearnedModel(env, known_R, known_P)
        if known_R and known_P:
            self.name = "Exact model + exhaustive"

    def act(self, t, m, rho, hist, eps, rng):
        return int(rng.integers(self.env.K))

    def observe(self, t, m, rho, a, r, t2, m2, rho2, done, hist):
        self.model.update(m, a, rho2, r, m2)

    def end_episode(self):
        pass

    def policy(self):
        return ExhaustivePolicy(self.env, self.model.R(), self.model.P())
