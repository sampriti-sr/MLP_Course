"""Recommender-system helpers for Part E: splits, ranking metrics and small readable models.

Everything is plain NumPy/SciPy so a student can read each function in a minute. Conventions:

* Interactions are a sparse matrix ``X`` (users x items). Rows and columns come from an
  :class:`Index` built on the *whole* data set, so train, test and every model share one shape.
* A model returns a dense **score matrix** (users x items). The evaluation code hides what a user
  already has in the training data and ranks the rest, so a model never needs to do that itself.
* Metrics are returned **per user**, so two models can be compared on the same users (a paired test).
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import svds

# --- Indexing -------------------------------------------------------------------------------


class Index(NamedTuple):
    users: np.ndarray   # row i of every matrix is user users[i]
    items: np.ndarray   # column j of every matrix is item items[j]


def make_index(df: pd.DataFrame, user: str = "userId", item: str = "movieId") -> Index:
    """Sorted unique user and item ids. Build it once on the full data, not on the training part."""
    return Index(np.sort(df[user].unique()), np.sort(df[item].unique()))


def to_csr(df: pd.DataFrame, index: Index, *, user: str = "userId", item: str = "movieId",
           value: str | None = None) -> sp.csr_matrix:
    """Users x items matrix. ``value=None`` gives 1 for every interaction; repeated pairs are summed."""
    rows = pd.Index(index.users).get_indexer(df[user])
    cols = pd.Index(index.items).get_indexer(df[item])
    if (rows < 0).any() or (cols < 0).any():
        raise ValueError("df contains users or items that are not in the Index.")
    data = np.ones(len(df), dtype=np.float32) if value is None else df[value].to_numpy(np.float32)
    return sp.csr_matrix((data, (rows, cols)), shape=(len(index.users), len(index.items)))


# --- Splitting ------------------------------------------------------------------------------


def split_per_user(df: pd.DataFrame, *, user: str = "userId", time: str = "timestamp",
                   test_frac: float = 0.2, how: str = "time", seed: int = 0):
    """Hold out a share of every user's interactions.

    ``how="time"`` holds out each user's *latest* ``test_frac`` (at least one, never all), which is how
    the model will be used: trained on the past, asked about the future. ``how="random"`` holds out a
    random share, which lets the model train on interactions that happened after some it is tested on.
    Returns ``(train, test)``.
    """
    if how not in ("time", "random"):
        raise ValueError("how must be 'time' or 'random'")
    d = df.copy()
    if how == "time":
        key = d[time].to_numpy()
    else:
        key = np.random.default_rng(seed).random(len(d))
    d["_key"] = key
    d["_pos"] = np.arange(len(d))                       # stable tie-break: file order
    d = d.sort_values([user, "_key", "_pos"])
    n = d.groupby(user)[user].transform("size").to_numpy()
    rank = d.groupby(user).cumcount().to_numpy()        # 0 = earliest
    n_test = np.clip(np.round(n * test_frac).astype(int), 1, n - 1)
    is_test = rank >= n - n_test
    d = d.drop(columns=["_key", "_pos"])
    return d[~is_test], d[is_test]


def assert_chronological(train: pd.DataFrame, test: pd.DataFrame, *, user: str = "userId",
                         time: str = "timestamp") -> None:
    """Raise if any user has a training interaction later than one of their test interactions."""
    last_train = train.groupby(user)[time].max()
    first_test = test.groupby(user)[time].min()
    both = last_train.index.intersection(first_test.index)
    bad = both[(last_train[both] > first_test[both]).to_numpy()]
    if len(bad):
        raise AssertionError(f"{len(bad)} users have training interactions later than their test interactions.")


# --- Ranking -------------------------------------------------------------------------------


def top_k(scores: np.ndarray, exclude: sp.spmatrix | None, k: int = 10, chunk: int = 512) -> np.ndarray:
    """The ``k`` best-scoring items per user, best first, never an item in ``exclude``.

    Ties are broken by item position, so results are reproducible. Returns (n_users, k) item columns.
    """
    n_users, n_items = scores.shape
    if np.isnan(scores).any():
        raise ValueError("scores contain NaN")
    if exclude is not None and n_items - exclude.getnnz(axis=1).max() < k:
        raise ValueError("some user has fewer than k unseen items to recommend")
    out = np.empty((n_users, k), dtype=np.int32)
    for start in range(0, n_users, chunk):
        block = np.array(scores[start:start + chunk], dtype=np.float32)
        if exclude is not None:
            coo = exclude[start:start + chunk].tocoo()
            block[coo.row, coo.col] = -np.inf
        out[start:start + chunk] = np.argsort(-block, axis=1, kind="stable")[:, :k]
    return out


def rank_metrics(top: np.ndarray, relevant: sp.spmatrix, chunk: int = 512) -> pd.DataFrame:
    """Per-user precision, recall, hit, AP and NDCG at ``k = top.shape[1]`` (binary relevance).

    Only users with at least one relevant item are scored (nothing can be "found" for the others);
    the frame is indexed by user row. AP is divided by ``min(k, n_relevant)`` and NDCG by the best
    possible DCG, so a perfect list scores 1.0.
    """
    k = top.shape[1]
    relevant = sp.csr_matrix(relevant)
    n_rel = np.asarray(relevant.sum(axis=1)).ravel()
    users = np.flatnonzero(n_rel > 0)
    discount = 1.0 / np.log2(np.arange(2, k + 2))
    ideal_cum = np.cumsum(discount)
    rows = []
    for start in range(0, len(users), chunk):
        u = users[start:start + chunk]
        dense = relevant[u].toarray() > 0
        hits = np.take_along_axis(dense, top[u], axis=1).astype(np.float64)
        nr = n_rel[u]
        cum = np.cumsum(hits, axis=1)
        prec_at = cum / np.arange(1, k + 1)
        rows.append(pd.DataFrame({
            "precision": hits.sum(1) / k,
            "recall": hits.sum(1) / nr,
            "hit": (hits.sum(1) > 0).astype(float),
            "ap": (prec_at * hits).sum(1) / np.minimum(k, nr),
            "ndcg": (hits * discount).sum(1) / ideal_cum[np.minimum(k, nr).astype(int) - 1],
            "n_relevant": nr,
        }, index=u))
    return pd.concat(rows) if rows else pd.DataFrame(columns=["precision", "recall", "hit", "ap", "ndcg", "n_relevant"])


def evaluate(scores: np.ndarray, train: sp.spmatrix, relevant: sp.spmatrix, k: int = 10) -> pd.DataFrame:
    """Rank everything the user has not already got in ``train`` and score the top ``k`` per user."""
    return rank_metrics(top_k(scores, train, k), relevant)


def paired_difference(a: pd.Series, b: pd.Series) -> dict:
    """Mean of ``a - b`` over the users both were scored on, with a 95% interval (normal approximation).

    If the interval contains 0, the data cannot tell the two models apart.
    """
    d = (a - b.reindex(a.index)).dropna()
    se = d.std(ddof=1) / np.sqrt(len(d))
    return {"diff": d.mean(), "se": se, "lo": d.mean() - 1.96 * se, "hi": d.mean() + 1.96 * se, "n_users": len(d)}


# --- Beyond accuracy -------------------------------------------------------------------------


def coverage(top: np.ndarray, n_items: int) -> float:
    """Share of the catalogue that appears in at least one user's list."""
    return len(np.unique(top)) / n_items


def gini(top: np.ndarray, n_items: int) -> float:
    """Concentration of recommendations over items: 0 = every item recommended equally often, 1 = one item only."""
    counts = np.sort(np.bincount(top.ravel(), minlength=n_items)).astype(np.float64)
    n = len(counts)
    return float((2 * np.arange(1, n + 1) - n - 1) @ counts / (n * counts.sum()))


def mean_popularity(top: np.ndarray, item_counts: np.ndarray) -> float:
    """Average number of training interactions of the recommended items."""
    return float(item_counts[top].mean())


def novelty(top: np.ndarray, item_counts: np.ndarray, n_users: int) -> float:
    """Mean self-information ``-log2(popularity share)`` of the recommended items; higher = more obscure."""
    p = np.maximum(item_counts, 1) / n_users
    return float((-np.log2(np.minimum(p, 1.0)))[top].mean())


def intra_list_diversity(top: np.ndarray, features: sp.spmatrix) -> float:
    """Average pairwise cosine *distance* between the items inside each list, then averaged over users."""
    f = sp.csr_matrix(features, dtype=np.float32)
    norm = np.sqrt(np.asarray(f.multiply(f).sum(1)).ravel())
    norm[norm == 0] = 1.0
    f = sp.diags(1.0 / norm) @ f
    k = top.shape[1]
    total = 0.0
    for row in top:
        g = f[row]
        sim = (g @ g.T).toarray()
        total += 1.0 - (sim.sum() - np.trace(sim)) / (k * (k - 1))
    return total / len(top)


# --- Baseline models -------------------------------------------------------------------------


def popularity(train: sp.spmatrix) -> np.ndarray:
    """Item score = how many users interacted with it. The same list for everybody; shape (n_items,)."""
    return np.asarray(sp.csr_matrix(train).getnnz(axis=0), dtype=np.float32)


def damped_mean(ratings: sp.spmatrix, damping: float = 10.0) -> np.ndarray:
    """Mean rating pulled toward the overall mean as ``(sum + damping * mu) / (count + damping)``.

    With ``damping=0`` this is the plain mean, which crowns items with a single 5-star rating.
    """
    r = sp.csr_matrix(ratings)
    count = np.asarray(r.getnnz(axis=0), dtype=np.float64)
    total = np.asarray(r.sum(axis=0)).ravel()
    mu = r.data.mean()
    return ((total + damping * mu) / (count + damping)).astype(np.float32)


def broadcast(item_scores: np.ndarray, n_users: int) -> np.ndarray:
    """Give every user the same item scores (for non-personalised models)."""
    return np.broadcast_to(item_scores, (n_users, len(item_scores)))


def item_similarity(X: sp.spmatrix, k: int = 50, shrink: float = 0.0, chunk: int = 512) -> sp.csr_matrix:
    """Item-item cosine similarity, keeping only each item's ``k`` most similar neighbours.

    ``shrink`` multiplies a similarity built from ``n`` co-occurring users by ``n / (n + shrink)`` so
    that two items seen together by one person are not "perfectly similar". Needs the co-occurrence
    count, so it is computed on the binary version of ``X``.
    """
    X = sp.csr_matrix(X, dtype=np.float32)
    B = X.copy()
    B.data[:] = 1.0
    norm = np.sqrt(np.asarray(X.multiply(X).sum(axis=0)).ravel())
    norm[norm == 0] = 1.0
    n_items = X.shape[1]
    XT = X.T.tocsr()
    BT = B.T.tocsr()
    rows, cols, vals = [], [], []
    for start in range(0, n_items, chunk):
        stop = min(start + chunk, n_items)
        dot = (XT[start:stop] @ X).toarray()
        co = (BT[start:stop] @ B).toarray()
        sim = dot / (norm[start:stop, None] * norm[None, :])
        if shrink:
            sim *= co / (co + shrink)
        sim[np.arange(stop - start), np.arange(start, stop)] = 0.0       # an item is not its own neighbour
        kk = min(k, n_items - 1)
        idx = np.argpartition(-sim, kk - 1, axis=1)[:, :kk]
        v = np.take_along_axis(sim, idx, axis=1)
        keep = v > 0
        rows.append(np.repeat(np.arange(start, stop), kk)[keep.ravel()])
        cols.append(idx.ravel()[keep.ravel()])
        vals.append(v.ravel()[keep.ravel()])
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n_items, n_items))


def item_knn_scores(X: sp.spmatrix, S: sp.spmatrix) -> np.ndarray:
    """Score of item j for a user = sum of similarities between j and the items the user has."""
    return np.asarray((sp.csr_matrix(X) @ sp.csr_matrix(S).T).todense(), dtype=np.float32)


def user_similarity(X: sp.spmatrix, k: int = 50) -> sp.csr_matrix:
    """User-user cosine similarity, keeping each user's ``k`` nearest neighbours."""
    X = sp.csr_matrix(X, dtype=np.float32)
    norm = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    norm[norm == 0] = 1.0
    sim = (X @ X.T).toarray() / (norm[:, None] * norm[None, :])
    np.fill_diagonal(sim, 0.0)
    kk = min(k, sim.shape[0] - 1)
    idx = np.argpartition(-sim, kk - 1, axis=1)[:, :kk]
    v = np.take_along_axis(sim, idx, axis=1)
    keep = v > 0
    rows = np.repeat(np.arange(sim.shape[0]), kk)[keep.ravel()]
    return sp.csr_matrix((v.ravel()[keep.ravel()], (rows, idx.ravel()[keep.ravel()])), shape=sim.shape)


def user_knn_scores(X: sp.spmatrix, S: sp.spmatrix) -> np.ndarray:
    """Score of item j for a user = sum of the similarities of the neighbours who have j."""
    return np.asarray((sp.csr_matrix(S) @ sp.csr_matrix(X)).todense(), dtype=np.float32)


def predict_ratings_knn(R: sp.spmatrix, S: sp.spmatrix, *, axis: str = "item") -> np.ndarray:
    """Predict *rating values* from neighbours' mean-centred ratings: ``mean_u + sum(s * dev) / sum(|s|)``.

    ``axis="item"``: ``S`` is item-item, deviations are the user's own ratings of similar items.
    ``axis="user"``: ``S`` is user-user, deviations are similar users' ratings of the item.
    Where no neighbour has a rating, the user's mean rating is returned.
    """
    R = sp.csr_matrix(R, dtype=np.float32)
    counts = np.asarray(R.getnnz(axis=1)).ravel()
    user_mean = np.divide(np.asarray(R.sum(axis=1)).ravel(), counts, out=np.zeros(R.shape[0]), where=counts > 0)
    centred = R.copy()
    centred.data = centred.data - np.repeat(user_mean, counts).astype(np.float32)
    seen = R.copy()
    seen.data[:] = 1.0
    S = sp.csr_matrix(S)
    if axis == "item":
        num = (centred @ S.T).toarray()
        den = (seen @ abs(S).T).toarray()
    else:
        num = (S @ centred).toarray()
        den = (abs(S) @ seen).toarray()
    return (user_mean[:, None] + np.divide(num, den, out=np.zeros_like(num), where=den > 0)).astype(np.float32)


def pure_svd(X: sp.spmatrix, factors: int = 20, seed: int = 0) -> np.ndarray:
    """Truncated SVD of the interaction matrix; the rank-``factors`` reconstruction is the score matrix."""
    X = sp.csr_matrix(X, dtype=np.float64)
    u, s, vt = svds(X, k=factors, v0=np.random.default_rng(seed).standard_normal(min(X.shape)))
    return ((u * s) @ vt).astype(np.float32)


# --- Matrix factorisation --------------------------------------------------------------------


def bias_baseline(R: sp.spmatrix, reg_user: float = 5.0, reg_item: float = 10.0, iters: int = 10):
    """Global mean plus a user offset and an item offset, fitted by alternating shrunken averages.

    Returns ``(mu, b_user, b_item)``; the prediction is ``mu + b_user[u] + b_item[i]``.
    """
    R = sp.csr_matrix(R, dtype=np.float64)
    coo = R.tocoo()
    mu = coo.data.mean()
    bu = np.zeros(R.shape[0])
    bi = np.zeros(R.shape[1])
    n_u = np.bincount(coo.row, minlength=R.shape[0])
    n_i = np.bincount(coo.col, minlength=R.shape[1])
    for _ in range(iters):
        bi = np.bincount(coo.col, weights=coo.data - mu - bu[coo.row], minlength=R.shape[1]) / (reg_item + n_i)
        bu = np.bincount(coo.row, weights=coo.data - mu - bi[coo.col], minlength=R.shape[0]) / (reg_user + n_u)
    return mu, bu, bi


class MF(NamedTuple):
    user_f: np.ndarray
    item_f: np.ndarray
    offset: np.ndarray | float = 0.0   # added to every prediction (explicit: mu + biases; implicit: 0)

    def scores(self) -> np.ndarray:
        return (self.user_f @ self.item_f.T + self.offset).astype(np.float32)


def _solve_rows(indptr, indices, data, fixed, reg, extra=None, conf=None, weighted_reg=True):
    """One ALS half-step: re-solve every row's factor vector with the other side held fixed."""
    n_rows, f = len(indptr) - 1, fixed.shape[1]
    out = np.zeros((n_rows, f), dtype=np.float64)
    eye = np.eye(f)
    for r in range(n_rows):
        lo, hi = indptr[r], indptr[r + 1]
        if lo == hi:
            continue
        Y = fixed[indices[lo:hi]]
        if conf is None:                                   # explicit: only the observed entries count
            lam = reg * (hi - lo) if weighted_reg else reg
            out[r] = np.linalg.solve(Y.T @ Y + lam * eye, Y.T @ data[lo:hi])
        else:                                              # implicit (Hu, Koren, Volinsky 2008)
            c = conf[lo:hi]
            A = extra + (Y.T * (c - 1.0)) @ Y + reg * eye
            out[r] = np.linalg.solve(A, Y.T @ c)
    return out


def als_explicit(R: sp.spmatrix, factors: int = 32, reg: float = 0.1, iters: int = 15, seed: int = 0,
                 baseline: tuple | None = None) -> MF:
    """Biased matrix factorisation fitted by ALS on the *observed* ratings only.

    Prediction ``mu + b_u + b_i + p_u . q_i``. Each half-step appends a column of ones to the fixed side,
    so the user (or item) bias is solved together with its factors; ``baseline`` (default
    :func:`bias_baseline`) only provides the starting biases. Minimises the squared error plus
    ``reg * n`` times the squared length of each user's / item's parameters (``n`` = its rating count).
    """
    R = sp.csr_matrix(R, dtype=np.float64)
    RT = R.T.tocsr()
    mu, bu, bi = baseline if baseline is not None else bias_baseline(R)
    n_u, n_i = R.shape
    rng = np.random.default_rng(seed)
    Q = rng.normal(0, 0.1, (n_i, factors))
    for _ in range(iters):
        sol = _solve_rows(R.indptr, R.indices, R.data - mu - bi[R.indices], np.c_[Q, np.ones(n_i)], reg)
        P, bu = sol[:, :factors], sol[:, factors]
        sol = _solve_rows(RT.indptr, RT.indices, RT.data - mu - bu[RT.indices], np.c_[P, np.ones(n_u)], reg)
        Q, bi = sol[:, :factors], sol[:, factors]
    return MF(P, Q, mu + bu[:, None] + bi[None, :])


def als_implicit(X: sp.spmatrix, factors: int = 32, reg: float = 0.1, alpha: float = 20.0, iters: int = 10,
                 seed: int = 0) -> MF:
    """Implicit-feedback ALS: every cell is a 0/1 *preference*, observed cells get confidence ``1 + alpha * x``.

    Unlike :func:`als_explicit` it uses the cells that are *missing*, as weak evidence of "not wanted".
    """
    X = sp.csr_matrix(X, dtype=np.float64)
    XT = X.T.tocsr()
    rng = np.random.default_rng(seed)
    P = rng.normal(0, 0.1, (X.shape[0], factors))
    Q = rng.normal(0, 0.1, (X.shape[1], factors))
    for _ in range(iters):
        P = _solve_rows(X.indptr, X.indices, None, Q, reg, extra=Q.T @ Q, conf=1.0 + alpha * X.data)
        Q = _solve_rows(XT.indptr, XT.indices, None, P, reg, extra=P.T @ P, conf=1.0 + alpha * XT.data)
    return MF(P, Q, 0.0)


def fold_in(item_f: np.ndarray, items: np.ndarray, weights: np.ndarray | None = None, reg: float = 0.1,
            alpha: float = 20.0) -> np.ndarray:
    """Factor vector for a user who was not in training, from the items they have (implicit model).

    Solves the same equation as :func:`als_implicit` for one user with the item factors held fixed,
    so a new user can be scored after a handful of interactions without retraining anything.
    """
    f = item_f.shape[1]
    w = np.ones(len(items)) if weights is None else np.asarray(weights, dtype=np.float64)
    c = 1.0 + alpha * w
    Y = item_f[items]
    A = item_f.T @ item_f + (Y.T * (c - 1.0)) @ Y + reg * np.eye(f)
    return np.linalg.solve(A, Y.T @ c)


# --- Content features ------------------------------------------------------------------------


def genre_matrix(movies: pd.DataFrame, index: Index) -> tuple[sp.csr_matrix, list[str]]:
    """Multi-hot genre matrix (items x genres) aligned to ``index.items``; 'no genres listed' is dropped."""
    g = movies.set_index("movieId")["genres"].reindex(index.items).fillna("")
    names = sorted({x for s in g for x in s.split("|") if x and x != "(no genres listed)"})
    col = {n: i for i, n in enumerate(names)}
    rows, cols = [], []
    for r, s in enumerate(g):
        for x in s.split("|"):
            if x in col:
                rows.append(r)
                cols.append(col[x])
    return sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(g), len(names))), names
