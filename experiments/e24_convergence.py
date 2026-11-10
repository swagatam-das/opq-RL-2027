"""E24: numerical check of Theorem (finite-horizon convergence of sampled OpQ-soft).

The theorem says: with Robbins-Monro step sizes, every (t, m, a) updated infinitely often
and a consistent reward estimate, the sampled soft update converges almost surely to the
exact finite-horizon recursion Q^beta_t of Eq. (4).  We train OpQ-soft (beta = 5) on C1-C3
with random initial latents and compare two step-size rules:
  rm    : alpha_n = n^{-0.6}                (Robbins-Monro; the theorem's assumption)
  floor : alpha_n = max(0.02, n^{-0.6})     (the rule used in E18-E23; not Robbins-Monro)
for both the learned and the known reward observable, 3 seeds, 30000 episodes.
At each checkpoint we record, over all visited entries (t, m, a) (unvisited entries are
unreachable from the initial modes and never affect the readout):
  err_max  : max ||Q_t(m,a) - Q^beta_t(m,a)||_op / max ||Q^beta_t(m,a)||_op
  err_mean : mean of the same ratio
  score    : exact normalised return of the greedy readout (as in E18)
Usage (from the repository root): python experiments/e24_convergence.py
"""
import os, sys, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..')
sys.path.insert(0, ROOT)
from multiprocessing import Pool
from opq.envs import ENVS
from opq.agents import OpQSoft, op_soft_exact
from opq.train import train

BETA = 5.0
EPISODES = int(os.environ.get('E24_EPISODES', '30000'))
CK = [c for c in [30, 100, 300, 1000, 3000, 10000, 30000] if c <= EPISODES]
ENV_NAMES = ['random_oldp', 'pointing', 'qubit_prep']


_CACHE = {}


def reachable_entries(env):
    reach = [set(np.nonzero(env.p0)[0])]
    for _ in range(env.H - 1):
        nxt = set()
        for m in reach[-1]:
            for a in range(env.K):
                nxt |= set(np.nonzero(env.P[m, a])[0])
        reach.append(nxt)
    return sum(len(r) for r in reach) * env.K


def setup(name):
    if name in _CACHE:
        return _CACHE[name]
    env = ENVS[name](seed=0)
    env.random_init = True
    rng = np.random.default_rng(999); rhos = [env.sample_rho0(rng) for _ in range(8)]
    vs = [env.optimal_values(r)[1] for r in rhos]; vr = [env.random_value(r) for r in rhos]
    _CACHE[name] = (env, rhos, vs, vr)
    return _CACHE[name]


def job(args):
    name, rule, known_R, seed = args
    env, rhos, vs, vr = setup(name)
    Qex = op_soft_exact(env, BETA)
    scale = max(np.linalg.norm(Qex[t][m][a], 2) for t in range(env.H) for m in range(env.nM) for a in range(env.K))
    agent = OpQSoft(env, beta=BETA, known_R=known_R, seed=seed, lr_min=(0.0 if rule == 'rm' else 0.02))
    rec = []

    def ev(policy):
        idx = [(t, m, a) for t in range(env.H) for m in range(env.nM) for a in range(env.K) if agent.n[t, m, a] > 0]
        errs = [np.linalg.norm(agent.Q[t][m][a] - Qex[t][m][a], 2) / scale for t, m, a in idx]
        score = float(np.mean([(env.evaluate(policy, r) - b) / (a - b) for r, a, b in zip(rhos, vs, vr)]))
        rec.append(dict(err_max=float(max(errs)), err_mean=float(np.mean(errs)), visited=len(idx),
                        min_visits=float(min(agent.n[t, m, a] for t, m, a in idx)), score=score))
        return score
    t0 = time.time()
    curve, _ = train(env, agent, EPISODES, ev, CK, seed=seed)
    for (ep, _), r in zip(curve, rec):
        r['episode'] = ep
    return dict(env=name, rule=rule, known_R=known_R, seed=seed, entries=reachable_entries(env), checkpoints=rec,
                sec=time.time() - t0)


def main():
    jobs = [(n, rule, kr, s) for n in ENV_NAMES for rule in ['rm', 'floor'] for kr in [False, True] for s in range(3)]
    out = {'beta': BETA, 'episodes': EPISODES, 'checkpoints': CK, 'runs': [], 'exact_score': {}}
    for n in ENV_NAMES:
        env, rhos, vs, vr = setup(n)
        from opq.agents import greedy_from_ops
        pol = greedy_from_ops(op_soft_exact(env, BETA))
        out['exact_score'][n] = float(np.mean([(env.evaluate(pol, r) - b) / (a - b) for r, a, b in zip(rhos, vs, vr)]))
    os.makedirs(os.path.join(ROOT, 'results', 'e24'), exist_ok=True)
    with Pool(int(os.environ.get('NPROC', '2'))) as p:
        for r in p.imap_unordered(job, jobs):
            out['runs'].append(r)
            c = r['checkpoints'][-1]
            print(r['env'], r['rule'], 'knownR' if r['known_R'] else 'learnedR', r['seed'],
                  'err_max', round(c['err_max'], 4), 'err_mean', round(c['err_mean'], 4),
                  'score', round(c['score'], 3), f"visited {c['visited']}/{r['entries']}", f"{r['sec']:.0f}s", flush=True)
            json.dump(out, open(os.path.join(ROOT, 'results', 'e24', 'e24_results.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
