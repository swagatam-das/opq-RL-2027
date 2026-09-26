"""E12: (non-)monotonicity of the aggregation x -> R + gamma LSE_beta(x) (Prop. 5).

Commuting sector (shared eigenbasis): monotone (Prop. 5).  Generic inputs:
x_i <= y_i (Loewner) with y_i = x_i + P_i, P_i random PSD, does not imply
LSE(x) <= LSE(y); we count violations lambda_min(LSE(y) - LSE(x)) < -1e-12 and
sweep the perturbation scale down to 1e-4.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from opq.core import lse

rng = np.random.default_rng(0)
K, N, gam, beta = 4, 5, 0.95, 2.0
def rsym():
    A = rng.normal(size=(N, N)); return (A + A.T) / 2
def rpsd(scale):
    G = rng.normal(size=(N, N)); P = G @ G.T; return scale * P / np.linalg.norm(P, 2)
R = rsym()
agg = lambda xs: R + gam * lse(xs, beta)

def trial(commuting, scale):
    if commuting:
        V = np.linalg.qr(rng.normal(size=(N, N)))[0]
        x = [V @ np.diag(rng.normal(size=N)) @ V.T for _ in range(K)]
        y = [xi + V @ np.diag(scale * rng.uniform(0, 1, size=N)) @ V.T for xi in x]
    else:
        x = [rsym() for _ in range(K)]
        y = [xi + rpsd(scale) for xi in x]
    return np.linalg.eigvalsh(agg(y) - agg(x))[0]

m = np.array([trial(True, 0.5) for _ in range(2000)])
print(f"commuting, scale 0.5: {np.sum(m < -1e-12)}/2000 violations, min margin {m.min():+.3e}")
m = np.array([trial(False, 0.5) for _ in range(2000)])
print(f"generic,   scale 0.5: {np.sum(m < -1e-12)}/2000 violations, worst lambda_min {m.min():+.3e}")
for sc in [0.5, 0.1, 0.01, 1e-3, 1e-4]:
    m = np.array([trial(False, sc) for _ in range(1000)])
    print(f"generic,   scale {sc:g}: {np.sum(m < -1e-12)}/1000 violations, "
          f"worst lambda_min {m.min():+.3e}")
