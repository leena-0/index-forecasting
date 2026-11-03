"""2단계 모델(GRU·GARCH·GNN) 단위 테스트 (합성 데이터, 네트워크 없음).

    .venv/Scripts/python -m unittest
"""
import importlib.util
import unittest

import numpy as np
import pandas as pd

from src.models import garch, gnn, registry
from tests.test_models import ml_panel

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_SCIPY = importlib.util.find_spec("scipy") is not None


def simulate_garch(n, a, b, s2, seed=0):
    rng = np.random.default_rng(seed)
    w = s2 * (1 - a - b)
    sig2, r = s2, np.empty(n)
    for t in range(n):
        r[t] = np.sqrt(sig2) * rng.standard_normal()
        sig2 = w + a * r[t] ** 2 + b * sig2
    return r


@unittest.skipUnless(HAS_SCIPY, "scipy 없음 (requirements-models.txt)")
class GarchTest(unittest.TestCase):
    def test_recovers_parameters(self):
        r = simulate_garch(4000, 0.08, 0.90, 4e-4)
        a, b = garch.fit_one(r, 250)
        self.assertAlmostEqual(a, 0.08, delta=0.04)
        self.assertAlmostEqual(b, 0.90, delta=0.05)
        self.assertIsNone(garch.fit_one(r[:100], 250))  # 관측 부족

    def test_forecast_and_no_leak(self):
        p = ml_panel(n_days=400, tickers=("A", "B"))
        p["split"] = np.where(p["origin_date"] < p["origin_date"].sort_values().unique()[300], "train", "val")
        pred, info = garch.fit_predict(p, 5, {"min_obs": 100})
        self.assertEqual(info["n_fitted"], 2)
        self.assertTrue(np.isfinite(pred[p["split"] == "val"]).all())
        cut = p["origin_date"].sort_values().unique()[350]
        p2 = p.copy()
        p2.loc[p2["origin_date"] > cut, "ret"] = 0.5  # 검증 구간 컷오프 이후 값을 바꿔도
        pred2, _ = garch.fit_predict(p2, 5, {"min_obs": 100})
        early = p["origin_date"] <= cut
        np.testing.assert_allclose(pred[early], pred2[early])  # 컷오프까지 예측은 같다

    def test_short_history_uses_median_params(self):
        p = ml_panel(n_days=400, tickers=("A", "B"))
        dates = p["origin_date"].sort_values().unique()
        p["split"] = np.where(p["origin_date"] < dates[300], "train", "val")
        p.loc[(p["ticker"] == "B") & (p["origin_date"] < dates[250]), "ret"] = np.nan  # B는 학습 이력이 짧음
        pred, info = garch.fit_predict(p, 5, {"min_obs": 100})
        self.assertEqual(info["n_fitted"], 1)
        b = p["ticker"] == "B"
        self.assertTrue(np.isfinite(pred[b & (p["origin_date"] >= dates[250 + 59])]).all())  # warmup 60개 뒤부터
        self.assertTrue(pred[b & (p["origin_date"] < dates[250 + 59])].isna().all())


class GraphTest(unittest.TestCase):
    def test_graph_no_future_leak(self):
        p = ml_panel(n_days=120, tickers=("A", "B", "C", "D"))
        g1, _ = gnn.build_graphs(p, k=2, window=30, refresh=5)
        cut = p["origin_date"].sort_values().unique()[80]
        p2 = p.copy()
        p2.loc[p2["origin_date"] > cut, "ret"] = -p2.loc[p2["origin_date"] > cut, "ret"] * 3
        g2, _ = gnn.build_graphs(p2, k=2, window=30, refresh=5)
        for d, (rows, nb, mask) in g1.items():
            if d <= cut:
                np.testing.assert_array_equal(np.sort(nb, axis=1), np.sort(g2[d][1], axis=1))
                np.testing.assert_array_equal(rows, g2[d][0])
        rows, nb, mask = next(iter(g1.values()))
        self.assertEqual(nb.shape, (len(rows), 2))


@unittest.skipUnless(HAS_TORCH, "torch 없음 (requirements-models.txt)")
class TorchStage2Test(unittest.TestCase):
    def test_gru(self):
        p = ml_panel()
        pred, info = registry.fit_predict("gru", p, {"lookback": 5, "hidden": 4, "epochs": 1, "batch_size": 16})
        self.assertEqual(info["arch"], "gru")
        self.assertGreater(pred.notna().sum(), 0)

    def test_gnn_reproducible(self):
        p = ml_panel(n_days=80, tickers=("A", "B", "C"))
        params = {"k": 2, "window": 20, "hidden": 4, "epochs": 2}
        pred1, info = registry.fit_predict("gnn", p, params, seed=0)
        pred2, _ = registry.fit_predict("gnn", p, params, seed=0)
        np.testing.assert_allclose(pred1.dropna(), pred2.dropna())
        self.assertEqual(len(info["train_loss_by_epoch"]), 2)
        self.assertGreater(pred1.notna().sum(), 0)


if __name__ == "__main__":
    unittest.main()
