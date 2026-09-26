"""Core operator utilities for operator-valued Q-learning (OpQ).

Conventions follow the paper: tau = (1/N) Tr is the normalised trace, the
latent evolves as rho' = K_a(rho), and the dual (Heisenberg) map K_a^dag is
defined by tau(Y K_a(rho)) = tau(K_a^dag(Y) rho).
"""
import numpy as np
from scipy.linalg import expm


def tau(X):
    """Normalised trace (real part)."""
    return np.real(np.trace(X)) / X.shape[0]


def herm(X):
    return (X + X.conj().T) / 2


def fun_h(X, f):
    """Functional calculus f(X) for a Hermitian / real symmetric matrix X."""
    w, V = np.linalg.eigh(herm(X))
    return (V * f(w)) @ V.conj().T


def lse(xs, beta):
    """Operator log-sum-exp  (1/beta) log sum_i exp(beta x_i)  in double precision.

    Shifts by the largest eigenvalue for stability.  Reliable while
    beta * (eigenvalue spread) is below about 25; use lse_mp beyond that.
    """
    xs = [herm(x) for x in xs]
    c = max(np.linalg.eigvalsh(x)[-1] for x in xs)
    S = sum(fun_h(x - c * np.eye(x.shape[0]), lambda w: np.exp(beta * w)) for x in xs)
    return c * np.eye(S.shape[0]) + fun_h(S, lambda w: np.log(np.maximum(w, 1e-300)) / beta)


def lse_mp(xs, beta, dps=80):
    """Operator log-sum-exp in extended precision (mpmath), real symmetric input."""
    import mpmath as mp
    mp.mp.dps = dps
    S = None
    for x in xs:
        w, V = np.linalg.eigh(herm(np.real(x)))
        E = mp.matrix(V.tolist()) * mp.diag([mp.exp(beta * mp.mpf(float(v))) for v in w]) \
            * mp.matrix(V.T.tolist())
        S = E if S is None else S + E
    ev, EV = mp.eigsy(S)
    L = EV * mp.diag([mp.log(e) / beta for e in ev]) * EV.T
    return np.array([[float(L[i, j]) for j in range(L.cols)] for i in range(L.rows)])


def rand_skew(rng, n, scale):
    A = rng.normal(size=(n, n))
    return scale * (A - A.T)


def rand_sym(rng, n):
    A = rng.normal(size=(n, n))
    return (A + A.T) / 2


def rand_density(rng, n, complex_=False):
    """Random density operator normalised so that tau(rho) = 1."""
    G = rng.normal(size=(n, n))
    if complex_:
        G = G + 1j * rng.normal(size=(n, n))
    r = G @ G.conj().T
    return n * r / np.real(np.trace(r))


def rand_orth(rng, n, scale):
    return expm(rand_skew(rng, n, scale))


def hermitian_basis(n, complex_=True):
    """Orthonormal (w.r.t. Tr(A^* B)) basis of Hermitian (or real symmetric) n x n matrices."""
    B = []
    for i in range(n):
        E = np.zeros((n, n), dtype=complex if complex_ else float)
        E[i, i] = 1
        B.append(E)
    for i in range(n):
        for j in range(i + 1, n):
            E = np.zeros((n, n), dtype=complex if complex_ else float)
            E[i, j] = E[j, i] = 1 / np.sqrt(2)
            B.append(E)
            if complex_:
                F = np.zeros((n, n), dtype=complex)
                F[i, j] = -1j / np.sqrt(2)
                F[j, i] = 1j / np.sqrt(2)
                B.append(F)
    return B


def herm_features(rho, basis):
    """Real coordinates of a Hermitian matrix in an orthonormal Hermitian basis."""
    return np.array([np.real(np.trace(b.conj().T @ rho)) for b in basis])
