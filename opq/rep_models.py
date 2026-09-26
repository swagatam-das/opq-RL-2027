"""Sequence-return regression models for the representation experiments (class B).

Task: predict r(a_1..a_T) = (1/N) Tr(W A_{a_1} ... A_{a_T}) from the action word.
By Cor. 1 the operator model below is the fixed point of the compositional Bellman
recursion along each plan, so fitting it is fitting operator-valued Q-functions.

Models: OpQ(d) (ordered product of learned d x d operators + readout),
order-blind linear on one-hot (t, a_t), windowed table on (t, a_{t-1}, a_t),
GRU, LSTM, single-head self-attention, 1D-CNN on learned action embeddings.
"""
import numpy as np
import jax
import jax.numpy as jnp
import optax
from scipy.optimize import minimize

jax.config.update("jax_enable_x64", True)


# ------------------------------------------------------------------ closed-form baselines
def onehot_design(seqs, K, window):
    B, T = seqs.shape
    cols = []
    for t in range(T):
        if window == 1 or t == 0:
            idx = seqs[:, t] if window == 1 else seqs[:, t]
            size = K
        else:
            idx = seqs[:, t - 1] * K + seqs[:, t]
            size = K * K
        M = np.zeros((B, size))
        M[np.arange(B), idx] = 1
        cols.append(M)
    return np.hstack(cols + [np.ones((B, 1))])


def fit_linear(tr, te, K, window=1):
    (Xtr, ytr), (Xte, yte) = tr, te
    Dtr, Dte = onehot_design(Xtr, K, window), onehot_design(Xte, K, window)
    c = np.linalg.lstsq(Dtr, ytr, rcond=None)[0]
    rank = np.linalg.matrix_rank(Dtr)
    return float(np.mean((Dte @ c - yte) ** 2)), int(rank)


# ------------------------------------------------------------------ OpQ(d)
def opq_init(key, K, d, scale=0.3):
    k1, k2 = jax.random.split(key)
    G = jax.random.normal(k1, (K, d, d)) * scale
    Qs = jax.vmap(jax.scipy.linalg.expm)(G - jnp.swapaxes(G, 1, 2))
    return {"Q": Qs, "W": jax.random.normal(k2, (d, d)) / np.sqrt(d)}


def opq_apply(p, seqs):
    M = p["Q"][seqs[:, 0]]
    for t in range(1, seqs.shape[1]):
        M = M @ p["Q"][seqs[:, t]]
    return jnp.einsum('ij,bji->b', p["W"], M)


# ------------------------------------------------------------------ neural sequence models
def dense_init(key, i, o):
    return {"w": jax.random.normal(key, (i, o)) * np.sqrt(1.0 / i), "b": jnp.zeros(o)}


def dense(p, x):
    return x @ p["w"] + p["b"]


def rnn_init(key, K, h, cell):
    k1, k2, k3 = jax.random.split(key, 3)
    g = 3 if cell == "gru" else 4
    s = 1.0 / np.sqrt(h)
    return {"Wx": jax.random.uniform(k1, (K, g * h), minval=-s, maxval=s),
            "Wh": jax.random.uniform(k2, (h, g * h), minval=-s, maxval=s),
            "b": jnp.zeros(g * h), "out": dense_init(k3, h, 1)}


def rnn_apply(p, seqs, cell):
    h_dim = p["Wh"].shape[0]
    X = jax.nn.one_hot(seqs, p["Wx"].shape[0])
    B = seqs.shape[0]
    H = h_dim

    def gru(h, x):
        gx, gh = x @ p["Wx"] + p["b"], h @ p["Wh"]
        z = jax.nn.sigmoid(gx[:, :H] + gh[:, :H])
        r = jax.nn.sigmoid(gx[:, H:2 * H] + gh[:, H:2 * H])
        n = jnp.tanh(gx[:, 2 * H:] + r * gh[:, 2 * H:])
        return (1 - z) * n + z * h, None

    def lstm(hc, x):
        h, c = hc
        g = x @ p["Wx"] + h @ p["Wh"] + p["b"]
        i, f, o = (jax.nn.sigmoid(g[:, k * H:(k + 1) * H]) for k in range(3))
        c = f * c + i * jnp.tanh(g[:, 3 * H:])
        return (o * jnp.tanh(c), c), None

    Xs = jnp.swapaxes(X, 0, 1)
    if cell == "gru":
        h, _ = jax.lax.scan(gru, jnp.zeros((B, H)), Xs)
    else:
        (h, _), _ = jax.lax.scan(lstm, (jnp.zeros((B, H)), jnp.zeros((B, H))), Xs)
    return dense(p["out"], h)[:, 0]


def attn_init(key, K, T, w):
    ks = jax.random.split(key, 7)
    return {"emb": jax.random.normal(ks[0], (K, w)) * 0.5, "pos": jax.random.normal(ks[1], (T, w)) * 0.5,
            "q": dense_init(ks[2], w, w), "k": dense_init(ks[3], w, w), "v": dense_init(ks[4], w, w),
            "f1": dense_init(ks[5], w, 2 * w), "f2": dense_init(ks[6], 2 * w, w),
            "out": dense_init(jax.random.fold_in(key, 9), w, 1)}


def attn_apply(p, seqs):
    x = p["emb"][seqs] + p["pos"][None]
    q, k, v = dense(p["q"], x), dense(p["k"], x), dense(p["v"], x)
    a = jax.nn.softmax(q @ jnp.swapaxes(k, 1, 2) / np.sqrt(x.shape[-1]), axis=-1)
    x = x + a @ v
    x = x + dense(p["f2"], jax.nn.relu(dense(p["f1"], x)))
    return dense(p["out"], x.mean(axis=1))[:, 0]


def cnn_init(key, K, e=3, c=16):
    ks = jax.random.split(key, 4)
    return {"emb": jax.random.normal(ks[0], (K, e)),
            "c1": {"w": jax.random.normal(ks[1], (3, e, c)) / np.sqrt(3 * e), "b": jnp.zeros(c)},
            "c2": {"w": jax.random.normal(ks[2], (3, c, c)) / np.sqrt(3 * c), "b": jnp.zeros(c)},
            "out": dense_init(ks[3], c, 1)}


def conv(p, x):
    y = sum(x[:, k:x.shape[1] - 2 + k] @ p["w"][k] for k in range(3))
    return jax.nn.relu(y + p["b"])


def cnn_apply(p, seqs):
    x = p["emb"][seqs]
    x = conv(p["c2"], conv(p["c1"], x))
    return dense(p["out"], x.mean(axis=1))[:, 0]


def n_params(p):
    return int(sum(np.size(x) for x in jax.tree_util.tree_leaves(p)))


# ------------------------------------------------------------------ training
def fit(init_fn, apply_fn, tr, te, epochs, lr, restarts, seed=0, polish=0, clip=1.0):
    """Full-batch Adam with cosine decay, best-of-restarts by training loss;
    optional L-BFGS polishing (used for the operator model)."""
    (Xtr, ytr), (Xte, yte) = tr, te
    Xtr, Xte = jnp.asarray(Xtr), jnp.asarray(Xte)
    ytr, yte = jnp.asarray(ytr), jnp.asarray(yte)
    loss = lambda p, X, y: jnp.mean((apply_fn(p, X) - y) ** 2)
    sched = optax.cosine_decay_schedule(lr, epochs)
    opt = optax.chain(optax.clip_by_global_norm(clip), optax.adam(sched))

    @jax.jit
    def step(p, s):
        l, g = jax.value_and_grad(loss)(p, Xtr, ytr)
        u, s = opt.update(g, s, p)
        return optax.apply_updates(p, u), s, l

    best = None
    for r in range(restarts):
        p = init_fn(jax.random.PRNGKey(1000 * seed + r))
        s = opt.init(p)
        for _ in range(epochs):
            p, s, _ = step(p, s)
        if polish:
            flat, unravel = jax.flatten_util.ravel_pytree(p)
            f = jax.jit(lambda v: loss(unravel(v), Xtr, ytr))
            g = jax.jit(jax.grad(lambda v: loss(unravel(v), Xtr, ytr)))
            res = minimize(lambda v: float(f(v)), np.asarray(flat), jac=lambda v: np.asarray(g(v)),
                           method="L-BFGS-B", options={"maxiter": polish, "gtol": 1e-30, "ftol": 1e-30})
            p = unravel(jnp.asarray(res.x))
        tl = float(loss(p, Xtr, ytr))
        if not np.isfinite(tl):
            continue
        if best is None or tl < best[0]:
            best = (tl, float(loss(p, Xte, yte)), p)
    return {"train_mse": best[0], "test_mse": best[1], "params": n_params(best[2])}


import jax.flatten_util  # noqa: E402  (registers ravel_pytree)
