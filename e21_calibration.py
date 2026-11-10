"""E21: miscalibrated action operators, and the cost of M operators per state.

(a) The agent's model uses U~_a = U_a exp(eps A_a), A_a a random anti-Hermitian matrix
    with unit Frobenius norm (5 draws per eps); P and R are exact. Policies are built from
    the model and evaluated exactly on the true environment, at the training latent and 4
    held-out latents.  Two cases: the agent observes the true latent, or it tracks the
    latent from rho_0 and its own actions with the miscalibrated model.
    Methods: OpQ (greedy readout of the exact LSE recursion on the model, beta = 5) and
    exhaustive search on the model.
(b) Wall-clock cost of building and querying M operators per (t, m).
(c) Lower bound L_M(rho) = max over stored operators of tau(rho Gamma) from the point-based
    family (each Gamma is the value of a conditional plan), against V* and the certificate.
Usage (from the repository root): python experiments/e21_calibration.py [--lower-bound-only]
"""
import os, sys, json, time, copy
import numpy as np
from scipy.linalg import expm
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
from opq.envs import ENVS
from opq.core import tau
from opq.agents import op_soft_exact
from e20_tightening import qstar, pbvi, score_fn


def perturbed(env, eps, rng):
    model = copy.deepcopy(env)
    U = []
    for u in env.U:
        G = rng.normal(size=u.shape) + (1j * rng.normal(size=u.shape) if env.is_complex else 0)
        A = G - G.conj().T
        A = A / np.linalg.norm(A)
        U.append(u @ expm(eps * A))
    model.U = U
    return model


def make_policies(model, rho0, beta=5.0):
    Q = op_soft_exact(model, beta)

    def track(hist):
        r = rho0
        for (a, m2, rew) in hist:
            r = model.fwd(a, r)
        return r

    def opq(observed):
        def pol(t, m, rho, hist):
            r = rho if observed else track(hist)
            return int(np.argmax([tau(r @ q) for q in Q[t][m]]))
        return pol

    def plan(observed):
        def pol(t, m, rho, hist):
            r = rho if observed else track(hist)
            return int(np.argmax(qstar(model, t, m, r)))
        return pol
    return {('OpQ', 'observed'): opq(True), ('OpQ', 'tracked'): opq(False),
            ('exhaustive', 'observed'): plan(True), ('exhaustive', 'tracked'): plan(False)}


def pbvi_lower_bound(env, M, train_rhos, rng, rho):
    """Rebuild the point-based family of e20_tightening.pbvi and return L_M(rho) at t=0."""
    H, K, nM = env.H, env.K, env.nM
    pts = []
    for t in range(H):
        c = []
        for _ in range(M):
            r = train_rhos[rng.integers(len(train_rhos))]
            for a in rng.integers(0, K, size=t):
                r = env.fwd(a, r)
            c.append(r)
        pts.append(c)
    Z = np.zeros((env.N, env.N), dtype=complex if env.is_complex else float)
    G = [None] * (H + 1); G[H] = [[Z] for _ in range(nM)]
    for t in range(H - 1, -1, -1):
        G[t] = []
        for m in range(nM):
            ops = []
            for r in pts[t]:
                best, bv = None, -np.inf
                for a in range(K):
                    r2 = env.fwd(a, r); cont = 0
                    for m2 in range(nM):
                        if env.P[m, a, m2] > 0:
                            vals = [tau(r2 @ g) for g in G[t + 1][m2]]
                            cont = cont + env.P[m, a, m2] * G[t + 1][m2][int(np.argmax(vals))]
                    op = env.dual(a, env.R[m][a] + env.gamma * cont)
                    v = tau(r @ op)
                    if v > bv:
                        best, bv = op, v
                ops.append(best)
            G[t].append(ops)
    return float(sum(env.p0[m] * max(tau(rho @ g) for g in G[0][m]) for m in range(nM) if env.p0[m] > 0))


def lower_bounds():
    out = {}
    for name in ['random_oldp', 'pointing', 'qubit_prep']:
        env = ENVS[name](seed=0)
        rng = np.random.default_rng(999); held = [env.sample_rho0(rng) for _ in range(4)]
        Q = op_soft_exact(env, 5.0)
        cert = lambda r: sum(env.p0[m] * max(tau(r @ q) for q in Q[0][m]) for m in range(env.nM) if env.p0[m] > 0)
        row = {'vstar_train': env.optimal_values()[1], 'cert_train': cert(env.rho0),
               'vstar_held': float(np.mean([env.optimal_values(r)[1] for r in held])),
               'cert_held': float(np.mean([cert(r) for r in held]))}
        for M in [1, 2, 4, 8, 16, 32, 64]:
            lt = [pbvi_lower_bound(env, M, [env.rho0], np.random.default_rng(s), env.rho0) for s in range(5)]
            lh = []
            for s in range(5):
                tr = [env.sample_rho0(np.random.default_rng(100 + 10 * s + j)) for j in range(64)]
                lh.append(np.mean([pbvi_lower_bound(env, M, tr, np.random.default_rng(s), r) for r in held]))
            row[M] = {'L_train': float(np.mean(lt)), 'L_held': float(np.mean(lh))}
            assert np.mean(lt) <= row['vstar_train'] + 1e-9
        out[name] = row
        print(name, {k: (round(v['L_train'], 3), round(v['L_held'], 3)) if isinstance(v, dict) else round(v, 3) for k, v in row.items()}, flush=True)
    return out


def main():
    if '--lower-bound-only' in sys.argv:
        lb = lower_bounds()
        os.makedirs(os.path.join(ROOT, 'results', 'e21'), exist_ok=True)
        json.dump(lb, open(os.path.join(ROOT, 'results', 'e21', 'e21_lower_bound.json'), 'w'), indent=1, default=str)
        return
    out = {'calibration': {}, 'cost': {}}
    for name in ['random_oldp', 'pointing', 'qubit_prep']:
        env = ENVS[name](seed=0)
        rng = np.random.default_rng(999)
        lat = [env.rho0] + [env.sample_rho0(rng) for _ in range(4)]
        refs = [(env.optimal_values(r)[1], env.random_value(r)) for r in lat]
        res = {}
        for eps in [0.0, 0.01, 0.03, 0.1, 0.3]:
            draws = 1 if eps == 0 else 5
            acc = {}
            t0 = time.time()
            for d in range(draws):
                model = perturbed(env, eps, np.random.default_rng(1000 + d))
                for i, (r0, (vs, vr)) in enumerate(zip(lat, refs)):
                    for key, pol in make_policies(model, r0).items():
                        sc = (env.evaluate(pol, r0) - vr) / (vs - vr)
                        acc.setdefault(key, {'train': [], 'heldout': []})['train' if i == 0 else 'heldout'].append(sc)
            res[eps] = {f'{k[0]}|{k[1]}': {'train': float(np.mean(v['train'])), 'heldout': float(np.mean(v['heldout'])),
                                           'train_sd': float(np.std(v['train']))} for k, v in acc.items()}
            print(name, 'eps', eps, f'{time.time()-t0:.0f}s',
                  {k: (round(v['train'], 2), round(v['heldout'], 2)) for k, v in res[eps].items()}, flush=True)
        out['calibration'][name] = res
        # (b) cost of M operators
        cost = {}
        for M in [1, 8, 32, 64]:
            t0 = time.time(); pol = pbvi(env, M, [env.rho0], np.random.default_rng(0)); tb = time.time() - t0
            t0 = time.time(); [pol(0, m, env.rho0, ()) for m in range(env.nM) if env.p0[m] > 0]; tq = time.time() - t0
            cost[M] = dict(build_sec=tb, decision_ms=1000 * tq, stored_numbers=int(env.H * env.nM * M * env.N ** 2))
        out['cost'][name] = cost
        print(name, 'cost', cost, flush=True)
        os.makedirs(os.path.join(ROOT, 'results', 'e21'), exist_ok=True)
        json.dump(out, open(os.path.join(ROOT, 'results', 'e21', 'e21_results.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
