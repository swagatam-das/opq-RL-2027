"""E20: tightening the single-operator certificate (exact model, no learning).

Runs on the environments of the released code (opq/envs.py, seed 0, base horizons),
with the same 8 held-out latents as E18 (rng 999).
  (a) aggregators: LSE_beta (beta = 5, 20) vs the trace-minimal SDP upper bound
  (b) M operators per (t, m): point-based backups at M sampled reachable latents
  (c) slack spread along the greedy policy (sufficient condition, Prop. T-b)
  (d) C3 with delayed reward: depth-limited planner, zero leaf vs certificate leaf
Usage (from the repository root): python experiments/e20_tightening.py
Writes results/e20/e20_results.json.
"""
import sys, json, time
import numpy as np
import os
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
from opq.envs import ENVS, qubit_prep
from opq.core import tau, lse
from opq.agents import op_soft_exact, greedy_from_ops
import cvxpy as cp


def arrays(env):
    R = np.array([[env.R[m][a] for a in range(env.K)] for m in range(env.nM)])
    return np.stack(env.U), R


def latent_tree(env, rho, depth):
    Uc = np.stack(env.U); out = [rho[None]]
    for _ in range(depth):
        r = out[-1]
        out.append(np.einsum('aij,hjk,alk->hail', Uc, r, Uc.conj()).reshape(-1, env.N, env.N))
    return out


def qstar(env, t0, m0, rho, leaf=None):
    """Exact latent-aware Q*(t0, m0, rho, .) by search over all remaining histories.
    leaf: optional (t, m, rhos) -> values used at the search frontier (depth-limited)."""
    H, K, nM, g = env.H, env.K, env.nM, env.gamma
    _, R = arrays(env)
    depth = H - t0 if leaf is None else min(leaf[0], H - t0)
    lat = latent_tree(env, rho, depth)
    tend = t0 + depth
    if leaf is not None and tend < H:
        V = np.stack([leaf[1](tend, m, lat[depth]) for m in range(nM)], 1)
    else:
        V = np.zeros((K ** depth, nM))
    for d in range(depth - 1, -1, -1):
        rn = lat[d + 1].reshape(-1, K, env.N, env.N)
        r = np.real(np.einsum('mkij,nkji->nmk', R, rn)) / env.N
        Q = r + g * np.einsum('mkp,nkp->nmk', env.P, V.reshape(-1, K, nM))
        if d == 0:
            return Q[0, m0]
        V = Q.max(-1)


def score_fn(env, rhos):
    vs = [env.optimal_values(r)[1] for r in rhos]
    vr = [env.random_value(r) for r in rhos]
    return lambda pol: float(np.mean([(env.evaluate(pol, r) - b) / (a - b) for r, a, b in zip(rhos, vs, vr)])), vs


def cert_ratio(env, Q, rhos, vs):
    c = [sum(env.p0[m] * max(tau(r @ q) for q in Q[0][m]) for m in range(env.nM) if env.p0[m] > 0) for r in rhos]
    return float(np.mean(np.array(c) / np.array(vs)))


def sdp_agg(Qs):
    N = Qs[0].shape[0]; cplx = any(np.iscomplexobj(q) for q in Qs)
    Y = cp.Variable((N, N), hermitian=True) if cplx else cp.Variable((N, N), symmetric=True)
    cons = [Y - (q + q.conj().T) / 2 >> 0 for q in Qs]
    obj = cp.real(cp.trace(Y)) if cplx else cp.trace(Y)
    cp.Problem(cp.Minimize(obj), cons).solve(solver='SCS', eps=1e-9, max_iters=200000)
    Yv = np.asarray(Y.value)
    # repair tiny infeasibility so the certificate stays valid: shift by the worst violation
    viol = min(np.linalg.eigvalsh(Yv - (q + q.conj().T) / 2)[0] for q in Qs)
    return Yv + max(0.0, -viol) * np.eye(N), viol


def op_agg_exact(env, agg):
    H, K, nM, N = env.H, env.K, env.nM, env.N
    Z = np.zeros((N, N), dtype=complex if env.is_complex else float)
    Q = [[[Z.copy() for _ in range(K)] for _ in range(nM)] for _ in range(H + 1)]
    worst = 0.0
    for t in range(H - 1, -1, -1):
        V = []
        for m2 in range(nM):
            if t + 1 < H:
                y, v = agg(Q[t + 1][m2]); worst = min(worst, v); V.append(y)
            else:
                V.append(Z)
        for m in range(nM):
            for a in range(K):
                Q[t][m][a] = env.dual(a, env.R[m][a] + env.gamma * sum(env.P[m, a, m2] * V[m2] for m2 in range(nM)))
    return Q, worst


def pbvi(env, M, train_rhos, rng):
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

    def pol(t, m, rho, hist):
        q = []
        for a in range(K):
            r2 = env.fwd(a, rho)
            v = env.mean_reward(m, a, r2)
            for m2 in range(nM):
                if env.P[m, a, m2] > 0:
                    v += env.gamma * env.P[m, a, m2] * max(tau(r2 @ g) for g in G[t + 1][m2])
            q.append(v)
        return int(np.argmax(q))
    return pol


def slack_spread(env, Q, rho0):
    """Mean spread of tau(rho Q^beta) - q* across actions on nodes reached by the greedy
    policy (probability-weighted per step), the bound sum_t gamma^t E[spread], and the loss."""
    pol = greedy_from_ops(Q); g = env.gamma
    exp_spread = np.zeros(env.H); viol = 0.0
    frontier = [(env.p0[m], m, rho0) for m in range(env.nM) if env.p0[m] > 0]
    for t in range(env.H):
        nxt = []
        for w, m, rho in frontier:
            qs = qstar(env, t, m, rho)
            ce = np.array([tau(rho @ q) for q in Q[t][m]])
            d = ce - qs; viol = min(viol, d.min())
            exp_spread[t] += w * (d.max() - d.min())
            a = pol(t, m, rho, ())
            r2 = env.fwd(a, rho)
            for m2 in range(env.nM):
                if env.P[m, a, m2] > 0:
                    nxt.append((w * env.P[m, a, m2], m2, r2))
        frontier = nxt
    vstar = env.optimal_values(rho0)[1]
    loss = vstar - env.evaluate(pol, rho0)
    return dict(loss=float(loss), bound=float(np.sum(g ** np.arange(env.H) * exp_spread)),
                mean_spread=float(exp_spread.mean()), vstar=float(vstar), min_slack=float(viol))


def main():
    out = {}
    for name in ['random_oldp', 'pointing', 'qubit_prep']:
        env = ENVS[name](seed=0)
        rng = np.random.default_rng(999)
        held = [env.sample_rho0(rng) for _ in range(8)]
        sfix, vfix = score_fn(env, [env.rho0]); shel, vhel = score_fn(env, held)
        res = {}
        # (a) aggregators + Table 5 reproduction
        for lab, Q, worst in [('LSE b=5', op_soft_exact(env, 5.0), 0.0), ('LSE b=20', op_soft_exact(env, 20.0), 0.0)] + \
                [('SDP min-trace',) + op_agg_exact(env, sdp_agg)]:
            pol = greedy_from_ops(Q)
            res[lab] = dict(fixed=sfix(pol), heldout=shel(pol), cert_fixed=cert_ratio(env, Q, [env.rho0], vfix),
                            cert_heldout=cert_ratio(env, Q, held, vhel), sdp_worst_violation=worst)
            print(name, lab, res[lab], flush=True)
        # (c) slack spread for LSE b=5
        Q5 = op_soft_exact(env, 5.0)
        res['spread_fixed'] = slack_spread(env, Q5, env.rho0)
        res['spread_heldout'] = [slack_spread(env, Q5, r) for r in held]
        print(name, 'spread', res['spread_fixed'], flush=True)
        # (b) M operators
        fam = {}
        for M in [1, 2, 4, 8, 16, 32, 64]:
            f, h = [], []
            for s in range(5):
                r = np.random.default_rng(s)
                f.append(sfix(pbvi(env, M, [env.rho0], r)))
                tr = [env.sample_rho0(np.random.default_rng(100 + 10 * s + j)) for j in range(64)]
                h.append(shel(pbvi(env, M, tr, r)))
            fam[M] = dict(fixed=float(np.mean(f)), fixed_sd=float(np.std(f)), heldout=float(np.mean(h)), heldout_sd=float(np.std(h)))
            print(name, 'M', M, fam[M], flush=True)
        res['M_family'] = fam
        out[name] = res
        os.makedirs(os.path.join(ROOT, 'results', 'e20'), exist_ok=True)
        json.dump(out, open(os.path.join(ROOT, 'results', 'e20', 'e20_results.json'), 'w'), indent=1)
    # (d) delayed reward: C3 at longer horizons, target word of length H (E19 protocol)
    leaf = {}
    for H in [8, 10, 12]:
        env = qubit_prep(seed=0, H=H, depth=H)
        rng = np.random.default_rng(999); held = [env.sample_rho0(rng) for _ in range(4)]
        Q = op_soft_exact(env, 5.0)
        lv = lambda t, m, rhos: np.array([max(np.real(np.einsum('ij,ji->', q, r)) / env.N for q in Q[t][m]) for r in rhos])
        row = {}
        for lab, d, use in [('certificate greedy', None, None), ('depth 3, zero leaf', 3, False), ('depth 3, certificate leaf', 3, True),
                            ('depth 5, zero leaf', 5, False), ('depth 5, certificate leaf', 5, True)]:
            if d is None:
                pol = greedy_from_ops(Q)
            else:
                lf = (d, lv if use else (lambda t, m, rhos: np.zeros(len(rhos))))
                pol = (lambda lf: lambda t, m, rho, hist: int(np.argmax(qstar(env, t, m, rho, leaf=lf))))(lf)
            t0 = time.time()
            vals = [env.evaluate(pol)] + [env.evaluate(pol, r) for r in held]
            row[lab] = dict(train=float(vals[0]), heldout=float(np.mean(vals[1:])), sec=time.time() - t0)
        leaf[H] = row
        print('C3 leaf H', H, row, flush=True)
        out['leaf'] = leaf
        os.makedirs(os.path.join(ROOT, 'results', 'e20'), exist_ok=True)
        json.dump(out, open(os.path.join(ROOT, 'results', 'e20', 'e20_results.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
