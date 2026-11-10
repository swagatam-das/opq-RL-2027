"""E23: end-to-end control with a genuinely hidden initial latent.

The initial latent is drawn from a prior around a nominal state,
    v = normalise(v_nom + sigma g),   rho_0 = N v v^*,   g ~ N(0, I) (complex on C3),
and is NEVER observed: agents see the time, the mode and the rewards only.  They know
the operators K_a and the prior mean rho_bar_0 (a calibration constant), so they can
track the belief mean rho_bar_t = K_{a_{t-1}} ... K_{a_0}(rho_bar_0) from their own
actions.  Rewards are linear in rho, so the expected return of any policy of
(t, m, history) equals its exact value at rho_bar_0 (Proposition 1); in particular
the optimal belief-mean policy is computed exactly by dynamic programming at rho_bar_0
(the "blind optimum"), and every agent is evaluated exactly at rho_bar_0.

Agents trained without the latent (input = belief mean, rewards from the hidden latent):
  OpQ-soft (learned R, from rewards and the belief mean), OpQ-soft (known R),
  DQN on the belief mean, history table (model-free, ignores rho).
Agents trained WITH the latent under the same prior, then deployed on the belief mean
("plug-in"): OpQ-soft (known R) and DQN.  For linear values plugging in the mean is exact
(E[tau(rho Q)] = tau(rho_bar Q)); for a nonlinear Q-network it is not.
Score = (V - V_rand) / (V_blind* - V_rand), all at rho_bar_0.  Also reported: the
latent-aware oracle (mean over prior draws of V*(rho_0)), which no blind agent can reach.
Usage (from the repository root): python experiments/e23_hidden_latent.py
"""
import os, sys, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..')
sys.path.insert(0, ROOT)
from multiprocessing import Pool
from opq.envs import ENVS
from opq.core import tau
from opq.agents import OpQSoft, TabularQ, op_soft_exact, greedy_from_ops
from opq.deep_agents import DQN
from opq.train import train

SIGMA = 0.5
CK = [100, 400, 1000, 3000]


def prior(env):
    w, V = np.linalg.eigh(env.rho0); vnom = V[:, -1]
    def sample(rng):
        g = rng.normal(size=env.N) + (1j * rng.normal(size=env.N) if env.is_complex else 0)
        v = vnom + SIGMA * g; v = v / np.linalg.norm(v)
        return env.N * np.outer(v, v.conj())
    r = np.random.default_rng(2024)
    rbar = np.mean([sample(r) for _ in range(20000)], axis=0)
    return sample, rbar


class HiddenEnv:
    """Agents see the belief mean; transitions and rewards use the hidden latent."""
    def __init__(self, env, sample, rbar, observed=False):
        self.env, self.sample, self.rbar, self.observed = env, sample, rbar, observed
        self._true = None

    def __getattr__(self, k):
        return getattr(self.env, k)

    def reset(self, rng):
        self._true = self.sample(rng)
        _, m, _ = self.env.reset(rng)
        return 0, m, (self._true.copy() if self.observed else self.rbar.copy())

    def step(self, t, m, rho, a, rng):
        r, t2, m2, true2, done = self.env.step(t, m, self._true, a, rng)
        self._true = true2
        return r, t2, m2, (true2 if self.observed else self.env.fwd(a, rho)), done


AGENTS = {
    'OpQ-soft (learned R)': (False, lambda e, s: OpQSoft(e, beta=5.0, seed=s)),
    'OpQ-soft (known R)': (False, lambda e, s: OpQSoft(e, beta=5.0, known_R=True, seed=s)),
    'DQN on belief mean': (False, lambda e, s: DQN(e, seed=s)),
    'History table': (False, lambda e, s: TabularQ(e, 'history', seed=s)),
    'OpQ-soft (known R), plug-in': (True, lambda e, s: OpQSoft(e, beta=5.0, known_R=True, seed=s)),
    'DQN, plug-in': (True, lambda e, s: DQN(e, seed=s)),
}


def refs(name):
    env = ENVS[name](seed=0); sample, rbar = prior(env)
    vb = env.optimal_values(rbar)[1]; vr = env.random_value(rbar)
    return env, sample, rbar, vb, vr


def job(args):
    name, agent, seed = args
    env, sample, rbar, vb, vr = refs(name)
    observed, make = AGENTS[agent]
    henv = HiddenEnv(env, sample, rbar, observed=observed)
    ag = make(henv, seed)
    ev = lambda pol: float((env.evaluate(pol, rbar) - vr) / (vb - vr))
    t0 = time.time()
    curve, _ = train(henv, ag, 3000, ev, CK, seed=seed)
    pol = ag.policy(); t1 = time.time()
    for _ in range(20):
        pol(0, 0, rbar, ())
    dec_ms = (time.time() - t1) / 20 * 1000
    return dict(env=name, agent=agent, seed=seed, curve=curve, decision_ms=dec_ms, sec=time.time() - t0)


def main():
    out = {'refs': {}, 'runs': []}
    sys.path.insert(0, HERE)
    from e20_tightening import qstar
    for name in ['random_oldp', 'pointing', 'qubit_prep']:
        env, sample, rbar, vb, vr = refs(name)
        r = np.random.default_rng(5)
        oracle = float(np.mean([env.optimal_values(sample(r))[1] for _ in range(32)]))
        exact = (env.evaluate(greedy_from_ops(op_soft_exact(env, 5.0)), rbar) - vr) / (vb - vr)
        t0 = time.time(); qstar(env, 0, int(np.argmax(env.p0)), rbar); plan_ms = (time.time() - t0) * 1000
        # linearity check: a belief-tracking policy evaluated on hidden draws vs exactly at rho_bar_0
        g = greedy_from_ops(op_soft_exact(env, 5.0))
        def blind(t, m, rho, hist):
            b = rbar
            for (a, _, _) in hist:
                b = env.fwd(a, b)
            return g(t, m, b, hist)
        rr = np.random.default_rng(11)
        mc = [env.evaluate(blind, sample(rr)) for _ in range(400)]
        lin = dict(exact=env.evaluate(blind, rbar), mc_mean=float(np.mean(mc)), mc_se=float(np.std(mc) / 20))
        out['refs'][name] = dict(V_blind=vb, V_rand=vr, V_oracle_aware=oracle,
                                 oracle_score=(oracle - vr) / (vb - vr), opq_exact_model_score=float(exact),
                                 blind_planner_decision_ms=plan_ms,
                                 belief_purity=float(np.real(np.trace(rbar @ rbar))) / env.N ** 2,
                                 linearity_check=lin)
        print(name, out['refs'][name], flush=True)
    jobs = [(n, a, s) for n in ['random_oldp', 'pointing', 'qubit_prep'] for a in AGENTS for s in range(3)]
    os.makedirs(os.path.join(ROOT, 'results', 'e23'), exist_ok=True)
    with Pool(int(os.environ.get('NPROC', '2'))) as p:
        for r in p.imap_unordered(job, jobs):
            out['runs'].append(r)
            print(r['env'], r['agent'], r['seed'], [round(c[1], 3) for c in r['curve']],
                  f"{r['decision_ms']:.2f}ms", f"{r['sec']:.0f}s", flush=True)
            json.dump(out, open(os.path.join(ROOT, 'results', 'e23', 'e23_results.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
