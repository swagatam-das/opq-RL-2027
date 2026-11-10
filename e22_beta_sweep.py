"""E22: sensitivity of the sampled soft control update (OpQ-soft) to beta.

The sampled update has no convergence guarantee (LSE_beta is neither monotone nor
non-expansive).  For beta in {0.5, 1, 2, 5, 10, 20} we train OpQ-soft (learned R) on
C1-C3 in the fixed and random regimes (3 seeds, 3000 episodes) and record
  score      : exact normalised return of the greedy readout at each checkpoint
  exact      : score of the greedy readout of the exact-model recursion at the same beta
  max_norm   : largest operator norm of any Q(t,m,a) during training (divergence check)
  drift      : ||Q_3000 - Q_2500|| / ||Q_3000|| (Frobenius over the whole table)
  switch     : fraction of greedy actions that change between checkpoints 2500 and 3000
               on a fixed probe set of 600 (t, m, rho) nodes from random rollouts
Usage (from the repository root): python experiments/e22_beta_sweep.py
"""
import os, sys, json, time, copy
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..')
sys.path.insert(0, ROOT)
from multiprocessing import Pool
from opq.envs import ENVS
from opq.core import tau
from opq.agents import OpQSoft, op_soft_exact, greedy_from_ops
from opq.train import train

BETAS = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
CK = [100, 400, 1000, 2000, 2500, 3000]


def setup(name, regime):
    env = ENVS[name](seed=0)
    if regime == 'random':
        env.random_init = True
        rng = np.random.default_rng(999); rhos = [env.sample_rho0(rng) for _ in range(8)]
    else:
        rhos = [env.rho0]
    vs = [env.optimal_values(r)[1] for r in rhos]; vr = [env.random_value(r) for r in rhos]
    # probe set: nodes reached by uniformly random rollouts from the evaluation latents
    prng = np.random.default_rng(7); probe = []
    while len(probe) < 600:
        rho = rhos[prng.integers(len(rhos))]; m = prng.choice(env.nM, p=env.p0)
        for t in range(env.H):
            probe.append((t, m, rho))
            a = int(prng.integers(env.K)); rho = env.fwd(a, rho); m = prng.choice(env.nM, p=env.P[m, a])
    return env, rhos, vs, vr, probe[:600]


def job(args):
    name, regime, beta, seed = args
    env, rhos, vs, vr, probe = setup(name, regime)
    agent = OpQSoft(env, beta=beta, seed=seed)
    snaps, maxnorm = {}, [0.0]

    def ev(policy):
        Qs = copy.deepcopy(agent.Q[:env.H]); ep = len(snaps)
        snaps[CK[ep]] = Qs
        maxnorm[0] = max(maxnorm[0], max(np.linalg.norm(q, 2) for Qt in Qs for Qm in Qt for q in Qm))
        return float(np.mean([(env.evaluate(policy, r) - b) / (a - b) for r, a, b in zip(rhos, vs, vr)]))
    t0 = time.time()
    curve, _ = train(env, agent, 3000, ev, CK, seed=seed)
    flat = lambda Qs: np.concatenate([q.ravel() for Qt in Qs for Qm in Qt for q in Qm])
    a, b = flat(snaps[3000]), flat(snaps[2500])
    drift = float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-12))
    g = lambda Qs, t, m, rho: int(np.argmax([tau(rho @ q) for q in Qs[t][m]]))
    switch = float(np.mean([g(snaps[3000], t, m, r) != g(snaps[2500], t, m, r) for t, m, r in probe]))
    return dict(env=name, regime=regime, beta=beta, seed=seed, curve=curve, max_norm=maxnorm[0],
                drift=drift, switch=switch, finite=bool(np.isfinite(a).all()), sec=time.time() - t0)


def main():
    out = {'runs': [], 'exact': {}}
    for name in ['random_oldp', 'pointing', 'qubit_prep']:
        for regime in ['fixed', 'random']:
            env, rhos, vs, vr, _ = setup(name, regime)
            for beta in BETAS:
                pol = greedy_from_ops(op_soft_exact(env, beta))
                out['exact'][f'{name}|{regime}|{beta}'] = float(np.mean(
                    [(env.evaluate(pol, r) - b) / (a - b) for r, a, b in zip(rhos, vs, vr)]))
    print('exact done', flush=True)
    jobs = [(n, g, b, s) for n in ['random_oldp', 'pointing', 'qubit_prep'] for g in ['fixed', 'random']
            for b in BETAS for s in range(3)]
    os.makedirs(os.path.join(ROOT, 'results', 'e22'), exist_ok=True)
    with Pool(int(os.environ.get('NPROC', '2'))) as p:
        for r in p.imap_unordered(job, jobs):
            out['runs'].append(r)
            print(r['env'], r['regime'], r['beta'], r['seed'], 'score', round(r['curve'][-1][1], 3),
                  'switch', round(r['switch'], 3), 'drift', round(r['drift'], 4), 'max', round(r['max_norm'], 2),
                  f"{r['sec']:.0f}s", flush=True)
            json.dump(out, open(os.path.join(ROOT, 'results', 'e22', 'e22_results.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
