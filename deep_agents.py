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


# ------------------------------------------------------------------------------ DQN + known K_a
class DQNAug(DQN):
    """DQN on rho_t trained on real transitions PLUS extra transitions generated from the known
    operators K_a (the same knowledge OpQ uses).

    Each real transition (t, m, a, m', done) is stored.  At every update the batch contains
    `batch - n_syn` real transitions and `n_syn` synthetic ones: the structural part
    (t, m, a, m', done) is resampled from the stored real transitions (so the mode kernel is learned
    from data, as in OpQ), a fresh latent rho is drawn, rho' = K_a rho K_a^dag is computed with the
    known operator and the reward is tau(R_hat(m, a) rho') with the same least-squares reward
    model OpQ uses (or the true R if known_R).

      aug_rho = 'pure'  : rho uniform over pure states (the 'random' regime distribution)
      aug_rho = 'reach' : rho_t reached by t random actions from the initial latent (on-support only)
    """
    name = "DQN + known K_a"

    def __init__(self, env, aug_frac=0.5, known_R=False, aug_rho="pure", updates_per_step=1,
                 buffer=50000, **kw):
        from .agents import RewardModel
        super().__init__(env, buffer=buffer, updates_per_step=updates_per_step, **kw)
        self.Rm = RewardModel(env, known_R)
        self.Uc = np.stack(env.U)
        self.n_syn = int(round(self.batch * aug_frac))
        self.aug_rho = aug_rho
        self.struct = np.zeros((buffer, 5), dtype=np.int64)       # t, m, a, m', done
        self.Bc = np.stack(self.basis).conj()                      # (d, N, N)
        self.srng = np.random.default_rng(10_000 + kw.get("seed", 0))
        tag = {"pure": "", "reach": ", reachable rho"}[aug_rho]
        self.name = "DQN + known K_a" + tag + (" (known R)" if known_R else "")

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

    # -- batched features ------------------------------------------------------------
    def feat_batch(self, t, m, rho):
        H, nM, B = self.env.H, self.env.nM, len(t)
        x = np.zeros((B, self.din), dtype=np.float32)
        ok = t < H
        x[np.arange(B)[ok], t[ok]] = 1
        x[np.arange(B), H + m] = 1
        x[:, H + nM:] = np.real(np.einsum('dij,bij->bd', self.Bc, rho))
        return x

    def _rho_syn(self, t):
        e, B = self.env, len(t)
        if self.aug_rho == "pure" or e.random_init:
            base = None
        else:
            base = e.rho0
        if self.aug_rho == "pure":
            v = self.srng.normal(size=(B, e.N))
            if e.is_complex:
                v = v + 1j * self.srng.normal(size=(B, e.N))
            v /= np.linalg.norm(v, axis=1, keepdims=True)
            return e.N * np.einsum('bi,bj->bij', v, v.conj())
        # reachable: start from rho_0 (fixed) or a random pure state (random regime), apply t random actions
        if base is None:
            v = self.srng.normal(size=(B, e.N))
            if e.is_complex:
                v = v + 1j * self.srng.normal(size=(B, e.N))
            v /= np.linalg.norm(v, axis=1, keepdims=True)
            rho = e.N * np.einsum('bi,bj->bij', v, v.conj())
        else:
            rho = np.repeat(base[None], B, axis=0).astype(np.result_type(base, float))
        for k in range(e.H):
            idx = np.flatnonzero(t > k)
            if len(idx) == 0:
                break
            a = self.srng.integers(e.K, size=len(idx))
            U = self.Uc[a]
            rho[idx] = np.einsum('bij,bjk,blk->bil', U, rho[idx], U.conj())
        return rho

    def observe(self, t, m, rho, a, r, t2, m2, rho2, done, hist):
        self.Rm.update(m, a, rho2, r)
        i = self.ptr
        self.buf["x"][i], self.buf["x2"][i] = self.feat(t, m, rho), self.feat(t2, m2, rho2)
        self.buf["a"][i], self.buf["r"][i], self.buf["d"][i] = a, r, float(done)
        self.struct[i] = (t, m, a, m2, int(done))
        self.ptr, self.size = (i + 1) % self.cap, min(self.size + 1, self.cap)
        self.steps += 1
        if self.size >= self.batch:
            for _ in range(self.ups):
                self.params, self.ost, _ = self._update(self.params, self.target, self.ost,
                                                        *self._batch())
        if self.steps % self.target_every == 0:
            self.target = self.params

    def _batch(self):
        nr = self.batch - self.n_syn
        idx = self.rng.integers(self.size, size=nr)
        x, a, r = self.buf["x"][idx], self.buf["a"][idx], self.buf["r"][idx]
        x2, d = self.buf["x2"][idx], self.buf["d"][idx]
        if self.n_syn == 0:
            return x, a, r, x2, d
        j = self.rng.integers(self.size, size=self.n_syn)
        t, m, aa, m2, dd = self.struct[j].T
        rho = self._rho_syn(t)
        rho2 = np.einsum('bij,bjk,blk->bil', self.Uc[aa], rho, self.Uc[aa].conj())
        Rb = np.stack([self.Rm(int(mi), int(ai)) for mi, ai in zip(m, aa)])
        rs = np.real(np.einsum('bij,bji->b', Rb, rho2)) / self.env.N
        xs, x2s = self.feat_batch(t, m, rho), self.feat_batch(t + 1, m2, rho2)
        return (np.concatenate([x, xs]), np.concatenate([a, aa.astype(np.float32)]),
                np.concatenate([r, rs.astype(np.float32)]), np.concatenate([x2, x2s]),
                np.concatenate([d, dd.astype(np.float32)]))
