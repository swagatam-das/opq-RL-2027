"""Deep scalar baselines for E18 (JAX / optax).

DQN  : MLP Q-network on (one-hot t, one-hot m, real coordinates of rho_t).
       The latent is observed, so this is the strongest scalar baseline.
DRQN : GRU Q-network on the observation history (previous action, previous
       reward, current mode, one-hot t); it never sees rho_t.
Both use a replay buffer, a target network and Adam.
"""
import numpy as np
import jax
import jax.numpy as jnp
import optax
from .core import hermitian_basis, herm_features


def _dense(key, i, o):
    return {"w": jax.random.normal(key, (i, o)) * np.sqrt(2.0 / i), "b": jnp.zeros(o)}


def mlp_init(key, sizes):
    ks = jax.random.split(key, len(sizes) - 1)
    return [_dense(k, a, b) for k, a, b in zip(ks, sizes[:-1], sizes[1:])]


def mlp_apply(p, x):
    for layer in p[:-1]:
        x = jax.nn.relu(x @ layer["w"] + layer["b"])
    return x @ p[-1]["w"] + p[-1]["b"]


# ----------------------------------------------------------------------------------- DQN
class DQN:
    name = "DQN (MLP on rho_t)"

    def __init__(self, env, hidden=64, lr=1e-3, batch=64, buffer=50000, target_every=250,
                 updates_per_step=1, seed=0):
        self.env = env
        self.basis = hermitian_basis(env.N, env.is_complex)
        self.din = env.H + env.nM + len(self.basis)
        self.params = mlp_init(jax.random.PRNGKey(seed), [self.din, hidden, hidden, env.K])
        self.target = self.params
        self.opt = optax.adam(lr)
        self.ost = self.opt.init(self.params)
        self.buf = {k: np.zeros((buffer,) + s, dtype=np.float32) for k, s in
                    [("x", (self.din,)), ("x2", (self.din,)), ("a", ()), ("r", ()), ("d", ())]}
        self.size, self.ptr, self.cap = 0, 0, buffer
        self.batch, self.target_every, self.ups = batch, target_every, updates_per_step
        self.steps = 0
        self.rng = np.random.default_rng(seed)
        g = env.gamma

        def loss(p, tp, x, a, r, x2, d):
            q = mlp_apply(p, x)[jnp.arange(x.shape[0]), a.astype(jnp.int32)]
            y = r + g * (1 - d) * jnp.max(mlp_apply(tp, x2), axis=1)
            return jnp.mean((q - jax.lax.stop_gradient(y)) ** 2)

        @jax.jit
        def update(p, tp, ost, x, a, r, x2, d):
            l, gr = jax.value_and_grad(loss)(p, tp, x, a, r, x2, d)
            u, ost = self.opt.update(gr, ost, p)
            return optax.apply_updates(p, u), ost, l

        self._update = update
        self._q = jax.jit(mlp_apply)

    def feat(self, t, m, rho):
        x = np.zeros(self.din, dtype=np.float32)
        if t < self.env.H:
            x[t] = 1
        x[self.env.H + m] = 1
        x[self.env.H + self.env.nM:] = herm_features(rho, self.basis)
        return x

    def act(self, t, m, rho, hist, eps, rng):
        if rng.random() < eps:
            return int(rng.integers(self.env.K))
        return int(np.argmax(np.asarray(self._q(self.params, self.feat(t, m, rho)[None]))[0]))

    def observe(self, t, m, rho, a, r, t2, m2, rho2, done, hist):
        i = self.ptr
        self.buf["x"][i], self.buf["x2"][i] = self.feat(t, m, rho), self.feat(t2, m2, rho2)
        self.buf["a"][i], self.buf["r"][i], self.buf["d"][i] = a, r, float(done)
        self.ptr, self.size = (i + 1) % self.cap, min(self.size + 1, self.cap)
        self.steps += 1
        if self.size >= self.batch:
            for _ in range(self.ups):
                idx = self.rng.integers(self.size, size=self.batch)
                b = [self.buf[k][idx] for k in ("x", "a", "r", "x2", "d")]
                self.params, self.ost, _ = self._update(self.params, self.target, self.ost, *b)
        if self.steps % self.target_every == 0:
            self.target = self.params

    def end_episode(self):
        pass

    def policy(self):
        p = self.params

        def pol(t, m, rho, hist):
            return int(np.argmax(np.asarray(self._q(p, self.feat(t, m, rho)[None]))[0]))
        return pol


# ----------------------------------------------------------------------------------- DRQN
def gru_init(key, din, h, K):
    k1, k2, k3 = jax.random.split(key, 3)
    s = 1.0 / np.sqrt(h)
    return {"Wx": jax.random.uniform(k1, (din, 3 * h), minval=-s, maxval=s),
            "Wh": jax.random.uniform(k2, (h, 3 * h), minval=-s, maxval=s),
            "b": jnp.zeros(3 * h), "out": _dense(k3, h, K)}


def gru_apply(p, xs):
    """xs: (B, T, din) -> Q values (B, T, K)."""
    h0 = jnp.zeros((xs.shape[0], p["Wh"].shape[0]))
    H = p["Wh"].shape[0]

    def cell(h, x):
        gx = x @ p["Wx"] + p["b"]
        gh = h @ p["Wh"]
        z = jax.nn.sigmoid(gx[:, :H] + gh[:, :H])
        r = jax.nn.sigmoid(gx[:, H:2 * H] + gh[:, H:2 * H])
        n = jnp.tanh(gx[:, 2 * H:] + r * gh[:, 2 * H:])
        h = (1 - z) * n + z * h
        return h, h
    _, hs = jax.lax.scan(cell, h0, jnp.swapaxes(xs, 0, 1))
    hs = jnp.swapaxes(hs, 0, 1)
    return hs @ p["out"]["w"] + p["out"]["b"]


class DRQN:
    name = "DRQN (GRU on history)"

    def __init__(self, env, hidden=64, lr=1e-3, batch=32, buffer=5000, target_every=50,
                 updates_per_episode=4, seed=0):
        self.env = env
        self.basis = hermitian_basis(env.N, env.is_complex)
        self.nb = len(self.basis)
        self.din = (env.K + 1) + 1 + env.nM + env.H + self.nb   # last block: rho_0 (step 0 only)
        self.params = gru_init(jax.random.PRNGKey(seed), self.din, hidden, env.K)
        self.target = self.params
        self.opt = optax.adam(lr)
        self.ost = self.opt.init(self.params)
        H = env.H
        self.X = np.zeros((buffer, H + 1, self.din), dtype=np.float32)
        self.A = np.zeros((buffer, H), dtype=np.int32)
        self.Rw = np.zeros((buffer, H), dtype=np.float32)
        self.size, self.ptr, self.cap = 0, 0, buffer
        self.batch, self.target_every, self.upe = batch, target_every, updates_per_episode
        self.episodes = 0
        self.rng = np.random.default_rng(seed)
        self.cur_x, self.cur_a, self.cur_r = [], [], []
        g = env.gamma

        def loss(p, tp, X, A, Rw):
            q = gru_apply(p, X)                         # (B, H+1, K)
            qt = gru_apply(tp, X)
            qa = jnp.take_along_axis(q[:, :H], A[..., None], axis=2)[..., 0]
            nxt = jnp.max(qt[:, 1:], axis=2).at[:, -1].set(0.0)
            y = Rw + g * nxt
            return jnp.mean((qa - jax.lax.stop_gradient(y)) ** 2)

        @jax.jit
        def update(p, tp, ost, X, A, Rw):
            l, gr = jax.value_and_grad(loss)(p, tp, X, A, Rw)
            u, ost = self.opt.update(gr, ost, p)
            return optax.apply_updates(p, u), ost, l

        self._update = update
        self._q = jax.jit(gru_apply)

    def xvec(self, t, m, prev_a, prev_r, rho0=None):
        x = np.zeros(self.din, dtype=np.float32)
        if rho0 is not None:
            x[-self.nb:] = herm_features(rho0, self.basis)
        x[prev_a + 1 if prev_a is not None else 0] = 1
        x[self.env.K + 1] = prev_r
        x[self.env.K + 2 + m] = 1
        if t < self.env.H:
            x[self.env.K + 2 + self.env.nM + t] = 1
        return x

    def hist_x(self, m0, hist, t, m):
        xs = [self.xvec(0, m0, None, 0.0, self.rho0)]
        for k, (a, m2, r) in enumerate(hist):
            xs.append(self.xvec(k + 1, m2, a, r))
        return np.stack(xs)

    def act(self, t, m, rho, hist, eps, rng):
        if t == 0:
            self.m0, self.rho0 = m, rho
        if rng.random() < eps:
            return int(rng.integers(self.env.K))
        X = self.hist_x(self.m0, hist, t, m)[None]
        return int(np.argmax(np.asarray(self._q(self.params, X))[0, -1]))

    def observe(self, t, m, rho, a, r, t2, m2, rho2, done, hist):
        if t == 0:
            self.m0 = m
            self.cur_x = [self.xvec(0, m, None, 0.0, rho)]
            self.cur_a, self.cur_r = [], []
        self.cur_a.append(a)
        self.cur_r.append(r)
        self.cur_x.append(self.xvec(t2, m2, a, r))

    def end_episode(self):
        i = self.ptr
        self.X[i], self.A[i], self.Rw[i] = np.stack(self.cur_x), self.cur_a, self.cur_r
        self.ptr, self.size = (i + 1) % self.cap, min(self.size + 1, self.cap)
        self.episodes += 1
        if self.size >= self.batch:
            for _ in range(self.upe):
                idx = self.rng.integers(self.size, size=self.batch)
                self.params, self.ost, _ = self._update(self.params, self.target, self.ost,
                                                        self.X[idx], self.A[idx], self.Rw[idx])
        if self.episodes % self.target_every == 0:
            self.target = self.params

    def policy(self):
        p = self.params
        m0 = {}

        def pol(t, m, rho, hist):
            if t == 0:
                m0["m"], m0["rho"] = m, rho
            self.rho0 = m0["rho"]
            X = self.hist_x(m0["m"], hist, t, m)[None]
            return int(np.argmax(np.asarray(self._q(p, X))[0, -1]))
        return pol
