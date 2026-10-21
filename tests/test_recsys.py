import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from mlp_data import recsys as rs


def _frame(rows):
    return pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"])


# --- indexing and splitting -------------------------------------------------------------------


def test_index_and_matrix_shape_follow_the_full_data():
    df = _frame([(10, 7, 4.0, 1), (10, 9, 3.0, 2), (30, 7, 5.0, 3)])
    ix = rs.make_index(df)
    X = rs.to_csr(df, ix, value="rating")
    assert X.shape == (2, 2) and X[0, 1] == 3.0 and X[1, 0] == 5.0 and X[1, 1] == 0.0
    assert rs.to_csr(df, ix).sum() == 3
    with pytest.raises(ValueError):
        rs.to_csr(_frame([(99, 7, 1.0, 1)]), ix)


def _many_users():
    rng = np.random.default_rng(1)
    rows = [(u, i, 3.0, int(t)) for u in range(1, 31) for i, t in enumerate(rng.permutation(np.arange(10, 10 + 25)))]
    return _frame(rows)


def test_time_split_never_trains_on_the_future_and_keeps_every_user_in_both_parts():
    df = _many_users()
    train, test = rs.split_per_user(df, test_frac=0.2, how="time")
    rs.assert_chronological(train, test)
    assert len(train) + len(test) == len(df)
    assert set(train.userId) == set(test.userId) == set(df.userId)
    assert (test.groupby("userId").size() == 5).all()
    assert not set(map(tuple, train[["userId", "movieId"]].values)) & set(map(tuple, test[["userId", "movieId"]].values))


def test_random_split_breaks_chronology_and_the_guard_notices():
    train, test = rs.split_per_user(_many_users(), test_frac=0.2, how="random", seed=3)
    with pytest.raises(AssertionError, match="later than"):
        rs.assert_chronological(train, test)
    again, _ = rs.split_per_user(_many_users(), test_frac=0.2, how="random", seed=3)
    pd.testing.assert_frame_equal(train, again)


def test_split_keeps_at_least_one_train_and_one_test_row():
    df = _frame([(1, 1, 3.0, 1), (1, 2, 3.0, 2), (2, 1, 3.0, 1), (2, 2, 3.0, 2), (2, 3, 3.0, 3)])
    train, test = rs.split_per_user(df, test_frac=0.9)
    assert (train.groupby("userId").size() >= 1).all() and (test.groupby("userId").size() >= 1).all()


# --- ranking metrics (hand-computed) ------------------------------------------------------------


def _rel(n_items, *rows):
    m = np.zeros((len(rows), n_items))
    for r, items in enumerate(rows):
        m[r, list(items)] = 1
    return sp.csr_matrix(m)


def test_metrics_match_hand_calculation():
    top = np.array([[2, 0, 5]])
    m = rs.rank_metrics(top, _rel(6, {2, 5})).iloc[0]
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == 1.0 and m["hit"] == 1.0
    assert m["ap"] == pytest.approx((1 / 1 + 2 / 3) / 2)
    dcg = 1 / np.log2(2) + 1 / np.log2(4)
    idcg = 1 / np.log2(2) + 1 / np.log2(3)
    assert m["ndcg"] == pytest.approx(dcg / idcg)


def test_perfect_and_empty_lists():
    perfect = rs.rank_metrics(np.array([[4, 1, 3]]), _rel(6, {4, 1, 3})).iloc[0]
    assert perfect[["precision", "recall", "hit", "ap", "ndcg"]].tolist() == pytest.approx([1, 1, 1, 1, 1])
    miss = rs.rank_metrics(np.array([[0, 1, 2]]), _rel(6, {5})).iloc[0]
    assert miss[["precision", "recall", "hit", "ap", "ndcg"]].tolist() == [0, 0, 0, 0, 0]


def test_users_without_relevant_items_are_not_scored():
    out = rs.rank_metrics(np.array([[0, 1], [0, 1]]), _rel(4, set(), {1}))
    assert list(out.index) == [1]


def test_rank_matters_for_ndcg_and_ap_but_not_for_precision():
    rel = _rel(5, {3})
    early = rs.rank_metrics(np.array([[3, 0, 1]]), rel).iloc[0]
    late = rs.rank_metrics(np.array([[0, 1, 3]]), rel).iloc[0]
    assert early["precision"] == late["precision"]
    assert early["ndcg"] > late["ndcg"] and early["ap"] > late["ap"]


def test_top_k_hides_seen_items_and_breaks_ties_by_position():
    scores = np.array([[5.0, 5.0, 5.0, 1.0]])
    seen = sp.csr_matrix(np.array([[1, 0, 0, 0]]))
    assert rs.top_k(scores, seen, k=2).tolist() == [[1, 2]]
    assert rs.top_k(scores, None, k=2).tolist() == [[0, 1]]
    with pytest.raises(ValueError, match="fewer than k"):
        rs.top_k(scores, sp.csr_matrix(np.array([[1, 1, 1, 0]])), k=2)
    with pytest.raises(ValueError, match="NaN"):
        rs.top_k(np.array([[np.nan, 1.0, 2.0]]), None, k=1)


def test_evaluate_never_credits_an_item_the_user_already_had():
    train = sp.csr_matrix(np.array([[1, 0, 0, 0]]))
    scores = np.array([[9.0, 1.0, 2.0, 0.0]])                  # the best-scored item is one the user already has
    m = rs.evaluate(scores, train, _rel(4, {0, 2}), k=1).iloc[0]
    assert m["hit"] == 1.0                                       # item 2 is the best *unseen* item


def test_paired_difference_detects_a_shift_and_a_tie():
    idx = np.arange(200)
    rng = np.random.default_rng(0)
    base = pd.Series(rng.random(200), index=idx)
    assert rs.paired_difference(base + 0.1, base)["lo"] > 0.09
    tie = rs.paired_difference(base, base + rng.normal(0, 0.05, 200))
    assert tie["lo"] < 0 < tie["hi"] and tie["n_users"] == 200


# --- beyond accuracy ----------------------------------------------------------------------------


def test_coverage_gini_popularity_and_novelty():
    top = np.array([[0, 1], [2, 3]])
    assert rs.coverage(top, 8) == 0.5
    assert rs.gini(np.array([[0, 1], [2, 3]]), 4) == pytest.approx(0.0)
    assert rs.gini(np.array([[0, 0], [0, 0]]), 4) == pytest.approx(0.75)         # one of four items gets everything
    counts = np.array([100, 1, 1, 1])
    assert rs.mean_popularity(np.array([[0, 0]]), counts) == 100
    assert rs.novelty(np.array([[1]]), counts, 100) > rs.novelty(np.array([[0]]), counts, 100)


def test_intra_list_diversity_extremes():
    feats = sp.csr_matrix(np.array([[1, 0], [1, 0], [0, 1]]))
    assert rs.intra_list_diversity(np.array([[0, 1]]), feats) == pytest.approx(0.0)   # identical items
    assert rs.intra_list_diversity(np.array([[0, 2]]), feats) == pytest.approx(1.0)   # orthogonal items


# --- baselines and neighbourhood models ------------------------------------------------------------


def test_damped_mean_stops_one_lucky_rating_from_winning():
    R = np.zeros((20, 3))
    R[0, 0] = 5.0                                              # item 0: a single 5-star rating
    R[:, 1] = 4.5                                              # item 1: twenty 4.5-star ratings
    R[:, 2] = 2.0                                              # item 2: twenty 2-star ratings (pulls the overall mean down)
    plain, damped = rs.damped_mean(sp.csr_matrix(R), 0), rs.damped_mean(sp.csr_matrix(R), 10)
    assert plain[0] > plain[1] and damped[0] < damped[1]


def test_popularity_counts_users():
    X = sp.csr_matrix(np.array([[1, 1, 0], [1, 0, 0]]))
    assert rs.popularity(X).tolist() == [2, 1, 0]


def test_item_similarity_finds_items_bought_together_and_shrinks_thin_evidence():
    X = sp.csr_matrix(np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0]]))
    S = rs.item_similarity(X, k=3).toarray()
    assert S[0, 1] > S[0, 2] and np.allclose(np.diag(S), 0)
    assert rs.item_similarity(X, k=3, shrink=5).toarray()[0, 1] < S[0, 1]
    assert (rs.item_similarity(X, k=1).getnnz(axis=1) <= 1).all()


def test_item_knn_recommends_the_partner_item():
    X = sp.csr_matrix(np.array([[1, 1, 0, 0]] * 5 + [[0, 0, 1, 1]] * 5 + [[1, 0, 0, 0]]))
    scores = rs.item_knn_scores(X, rs.item_similarity(X, k=3))
    assert scores[10, 1] > scores[10, 2] and scores[10, 1] > scores[10, 3]


def test_user_knn_recommends_what_similar_users_have():
    X = sp.csr_matrix(np.array([[1, 1, 1, 0]] * 4 + [[0, 0, 0, 1]] * 4 + [[1, 1, 0, 0]]))
    scores = rs.user_knn_scores(X, rs.user_similarity(X, k=4))
    assert scores[8, 2] > scores[8, 3]


def test_knn_rating_prediction_moves_toward_neighbours_ratings():
    R = np.zeros((6, 3))
    R[:5] = [[5, 5, 1]] * 5                                    # five users like items 0,1 and dislike 2
    R[5, :2] = [5, 1]                                          # a user who rated only items 0 and 1
    Rs = sp.csr_matrix(R)
    pred_item = rs.predict_ratings_knn(Rs, rs.item_similarity(Rs, k=2), axis="item")
    pred_user = rs.predict_ratings_knn(Rs, rs.user_similarity(Rs, k=5), axis="user")
    assert pred_item.shape == pred_user.shape == (6, 3)
    assert np.isfinite(pred_item).all() and np.isfinite(pred_user).all()
    assert pred_user[0, 2] == pytest.approx(R[R[:, 2] > 0][:, 2].mean(), abs=1.0)


def test_pure_svd_shape():
    X = sp.random(30, 20, density=0.3, format="csr", random_state=0)
    assert rs.pure_svd(X, factors=3).shape == (30, 20)


# --- matrix factorisation -------------------------------------------------------------------------


def _low_rank(seed=0, n_u=60, n_i=40, f=2, obs=0.5):
    rng = np.random.default_rng(seed)
    P, Q = rng.normal(0, 1, (n_u, f)), rng.normal(0, 1, (n_i, f))
    bu, bi = rng.normal(0, 0.5, n_u), rng.normal(0, 0.5, n_i)
    full = 3.5 + bu[:, None] + bi[None, :] + P @ Q.T
    mask = rng.random(full.shape) < obs
    return full, mask


def test_bias_baseline_recovers_additive_structure():
    rng = np.random.default_rng(0)
    bu, bi = rng.normal(0, 0.6, 50), rng.normal(0, 0.6, 40)
    full = 3.5 + bu[:, None] + bi[None, :]
    mu, fu, fi = rs.bias_baseline(sp.csr_matrix(full), reg_user=0, reg_item=0)
    assert np.abs(mu + fu[:, None] + fi[None, :] - full).max() < 0.05


def test_explicit_als_fits_low_rank_data_and_improves_with_iterations():
    full, mask = _low_rank()
    R = sp.csr_matrix(np.where(mask, full, 0.0))
    rmse = lambda m: np.sqrt(np.mean((m.scores()[mask] - full[mask]) ** 2))
    one = rs.als_explicit(R, factors=2, reg=0.01, iters=1)
    many = rs.als_explicit(R, factors=2, reg=0.01, iters=15)
    assert rmse(many) < rmse(one) and rmse(many) < 0.15
    held = ~mask
    assert np.sqrt(np.mean((many.scores()[held] - full[held]) ** 2)) < 0.6        # it generalises to unseen cells


def _blocks():
    X = np.zeros((40, 20))
    X[:20, :10] = 1
    X[20:, 10:] = 1
    rng = np.random.default_rng(0)
    held = rng.random(X.shape) < 0.25
    return X, held


def test_implicit_als_learns_which_items_go_with_which_users():
    X, held = _blocks()
    train = sp.csr_matrix(np.where(held, 0, X))
    mf = rs.als_implicit(train, factors=4, reg=0.1, alpha=10, iters=8)
    s = mf.scores()
    assert s[:20, :10][held[:20, :10]].mean() > s[:20, 10:].mean() + 0.3           # held-out block cells beat the other block
    assert s[20:, 10:][held[20:, 10:]].mean() > s[20:, :10].mean() + 0.3


def test_fold_in_places_a_new_user_with_the_right_group():
    X, _ = _blocks()
    mf = rs.als_implicit(sp.csr_matrix(X), factors=4, reg=0.1, alpha=10, iters=8)
    new_user = rs.fold_in(mf.item_f, np.array([0, 1, 2]), reg=0.1, alpha=10)       # likes three "group A" items
    s = mf.item_f @ new_user
    assert s[:10].mean() > s[10:].mean() + 0.3


def test_genre_matrix_aligns_to_the_index():
    movies = pd.DataFrame({"movieId": [1, 2, 3], "genres": ["Comedy|Drama", "Drama", "(no genres listed)"]})
    ix = rs.Index(np.array([1, 2, 3, 4]), np.array([3, 2, 1, 4]))
    G, names = rs.genre_matrix(movies, ix)
    assert names == ["Comedy", "Drama"] and G.shape == (4, 2)
    assert G.toarray().tolist() == [[0, 0], [0, 1], [1, 1], [0, 0]]
