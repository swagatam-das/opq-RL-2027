"""Generic episodic training loop with exact evaluation checkpoints (E18)."""
import time
import numpy as np


def train(env, agent, episodes, eval_fn, checkpoints, seed=0, eps_start=1.0, eps_end=0.05,
          eps_frac=0.6):
    """Train `agent` for `episodes` episodes; at each checkpoint call eval_fn(policy)
    (an exact, noise-free evaluation) and record the result."""
    rng = np.random.default_rng(seed + 12345)
    curve, t0 = [], time.time()
    checkpoints = set(int(c) for c in checkpoints)
    for ep in range(1, episodes + 1):
        eps = max(eps_end, eps_start - (eps_start - eps_end) * ep / (eps_frac * episodes))
        t, m, rho = env.reset(rng)
        hist, done = (), False
        while not done:
            a = agent.act(t, m, rho, hist, eps, rng)
            r, t2, m2, rho2, done = env.step(t, m, rho, a, rng)
            agent.observe(t, m, rho, a, r, t2, m2, rho2, done, hist)
            hist = hist + ((a, m2, r),)
            t, m, rho = t2, m2, rho2
        agent.end_episode()
        if ep in checkpoints:
            curve.append((ep, eval_fn(agent.policy())))
    return curve, time.time() - t0
