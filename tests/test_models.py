"""src.models 단위 테스트 (합성 데이터, 네트워크 없음).

    .venv/Scripts/python -m unittest
"""
import importlib.util
import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.models import ewma, har, lgbm, lstm
from src.models.__main__ import run
from src.pipeline.features import build_features
from src.pipeline.target import forward_yz
from tests.test_features import make_prices


class EwmaTest(unittest.TestCase):
    def test_recursion(self):
        rv = np.array([np.nan, 1.0, 2.0, 3.0, np.nan, 5.0])
        out = ewma.ewma_var(rv, lam=0.5, warmup=2)
        self.assertTrue(np.isnan(out[:2]).all())  # 관측 2개(1행, 2행)가 모일 때까지 예측 없음
        self.assertAlmostEqual(out[2], 1.5)  # 초기값 = 처음 2개 평균
        self.assertAlmostEqual(out[3], 0.5 * 1.5 + 0.5 * 3.0)
        self.assertAlmostEqual(out[4], out[3])  # 결측일은 갱신하지 않음
        self.assertAlmostEqual(out[5], 0.5 * out[4] + 0.5 * 5.0)

    def test_too_short(self):
        self.assertTrue(np.isnan(ewma.ewma_var([1.0, 2.0], warmup=3)).all())

    def test_predict_per_ticker_and_no_leak(self):
        p = pd.DataFrame({"ticker": ["A"] * 4 + ["B"] * 4,
                          "origin_date": list(pd.bdate_range("2023-01-02", periods=4)) * 2,
                          "rv_d": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0]})
        p = p.sample(frac=1, random_state=0)  # 순서가 섞여도 종목·날짜 순으로 계산
        pred = ewma.predict(p, lam=0.5, warmup=2)
        a = pred[(p["ticker"] == "A") & (p["origin_date"] == pd.Timestamp("2023-01-04"))].iloc[0]
        self.assertAlmostEqual(a, 0.5 * 1.5 + 0.5 * 3.0)
        p2 = p.copy()
        p2.loc[p2["origin_date"] > pd.Timestamp("2023-01-04"), "rv_d"] = 999.0
        pred2 = ewma.predict(p2, lam=0.5, warmup=2)
        early = p["origin_date"] <= pd.Timestamp("2023-01-04")
        pd.testing.assert_series_equal(pred[early], pred2[early])


class HarTest(unittest.TestCase):
    def make_panel(self, n=400, seed=0):
        rng = np.random.default_rng(seed)
        rv = np.exp(rng.normal(-7, 0.5, size=(n, 3)))
        p = pd.DataFrame(rv, columns=har.REGRESSORS)
        p["log_var"] = 1.0 + 0.2 * np.log(p["rv_d"]) + 0.3 * np.log(p["rv_w"]) + 0.4 * np.log(p["rv_m"])
        p["split"] = np.where(np.arange(n) < n // 2, "train", "val")
        p["embargo"] = False
        return p

    def test_fit_recovers_beta(self):
        beta, n = har.fit(self.make_panel())
        np.testing.assert_allclose(beta, [1.0, 0.2, 0.3, 0.4], atol=1e-10)
        self.assertEqual(n, 200)

    def test_fit_uses_train_only(self):
        p = self.make_panel()
        p.loc[p["split"] == "val", "log_var"] = 100.0  # 검증 구간 값이 계수에 영향을 주면 안 됨
        p.loc[0, "embargo"] = True
        p.loc[0, "log_var"] = 100.0
        beta, n = har.fit(p)
        np.testing.assert_allclose(beta, [1.0, 0.2, 0.3, 0.4], atol=1e-10)
        self.assertEqual(n, 199)

    def test_nonpositive_rv(self):
        p = self.make_panel(n=10)
        p.loc[3, "rv_d"] = 0.0
        pred = har.predict(p, np.array([1.0, 0.2, 0.3, 0.4]))
        self.assertTrue(math.isnan(pred[3]))
        self.assertAlmostEqual(pred[4], p.loc[4, "log_var"])


class RunTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        cal, prices = make_prices(n_days=80, tickers=("A", "B"))
        build_features(prices, cal, pd.DataFrame(columns=["date", "ticker"])).to_parquet(self.tmp / "features.parquet")
        t = forward_yz(prices, cal, 5)
        t["split"] = np.where(t["origin_date"] < cal[50], "train", "val")
        t["embargo"] = False
        t.to_parquet(self.tmp / "targets_h5.parquet")
        (self.tmp / "meta.json").write_text(json.dumps({"reportable": False, "hold_reasons": ["test"]}),
                                            encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_run_writes_prediction_table(self):
        for model in ("ewma", "har"):
            table = run(model, 5, self.tmp, self.tmp / "out")
            self.assertEqual(list(table.columns),
                             ["ticker", "origin_date", "horizon", "pred_log_var", "model", "seed", "version"])
            self.assertTrue(np.isfinite(table["pred_log_var"]).all())
            self.assertFalse(table.duplicated(["ticker", "origin_date", "horizon"]).any())
        self.assertTrue((self.tmp / "out" / "EWMA_h5.parquet").exists())
        meta = json.loads((self.tmp / "out" / "HAR_h5.meta.json").read_text(encoding="utf-8"))
        self.assertFalse(meta["reportable"])
        self.assertEqual(set(meta["params"]["beta"]), {"const", "rv_d", "rv_w", "rv_m"})



HAS_LGBM = importlib.util.find_spec("lightgbm") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None


def ml_panel(n_days=80, tickers=("A", "B")):
    cal, prices = make_prices(n_days=n_days, tickers=tickers)
    f = build_features(prices, cal, pd.DataFrame(columns=["date", "ticker"])).rename(columns={"date": "origin_date"})
    t = forward_yz(prices, cal, 5)
    p = f.merge(t, on=["ticker", "origin_date"])
    p["log_var"] = np.log(p["yz_var"].where(p["yz_var"] > 0))
    p["split"] = np.where(p["origin_date"] < cal[50], "train", "val")
    p["embargo"] = False
    return p.sort_values(["ticker", "origin_date"], ignore_index=True)


class SequenceTest(unittest.TestCase):
    def test_windows_per_ticker_without_missing(self):
        p = ml_panel()
        X, rows = lstm.sequences(p, 5)
        self.assertEqual(X.shape[1:], (5, 3))
        self.assertTrue(np.isfinite(X).all())
        z = lstm.daily_features(p)
        r = rows[10]
        np.testing.assert_array_equal(X[10], z[r - 4:r + 1])  # 창 끝 = 예측 시점 t
        self.assertTrue((p["ticker"].to_numpy()[rows - 4] == p["ticker"].to_numpy()[rows]).all())  # 종목 경계를 넘지 않음
        first_b = p.index[p["ticker"] == "B"][0]
        self.assertNotIn(first_b + 3, set(rows))  # B의 첫 5일 전에는 창이 없음

    def test_no_future_leak(self):
        p = ml_panel()
        X1, r1 = lstm.sequences(p, 5)
        p2 = p.copy()
        cut = p2["origin_date"].sort_values().unique()[40]
        p2.loc[p2["origin_date"] > cut, ["rv_d", "ret", "log_turnover"]] = 9.0
        X2, r2 = lstm.sequences(p2, 5)
        w1, w2 = dict(zip(r1, X1)), dict(zip(r2, X2))
        early = [r for r in r1 if p.loc[r, "origin_date"] <= cut]
        self.assertGreater(len(early), 0)
        for r in early:  # 컷오프까지의 창은 컷오프 이후 값을 바꿔도 같다
            np.testing.assert_array_equal(w1[r], w2[r])


@unittest.skipUnless(HAS_LGBM, "lightgbm 없음 (requirements-models.txt)")
class LgbmTest(unittest.TestCase):
    def test_trains_on_train_only(self):
        p = ml_panel()
        m1, n = lgbm.fit(p, params={"num_boost_round": 20})
        p2 = p.copy()
        p2.loc[p2["split"] == "val", "log_var"] = 100.0  # 검증 구간 값은 학습에 영향 없음
        m2, _ = lgbm.fit(p2, params={"num_boost_round": 20})
        np.testing.assert_allclose(lgbm.predict(p, m1), lgbm.predict(p, m2))
        self.assertEqual(n, int(((p["split"] == "train") & p["log_var"].notna()).sum()))


@unittest.skipUnless(HAS_TORCH, "torch 없음 (requirements-models.txt)")
class LstmTest(unittest.TestCase):
    def test_fit_predict(self):
        p = ml_panel()
        params = {"lookback": 5, "hidden": 4, "epochs": 2, "batch_size": 16}
        pred, info = lstm.fit_predict(p, seed=0, params=params)
        pred2, _ = lstm.fit_predict(p, seed=0, params=params)
        np.testing.assert_allclose(pred.dropna(), pred2.dropna())  # 같은 seed → 같은 결과
        _, rows = lstm.sequences(p, 5)
        self.assertEqual(pred.notna().sum(), len(rows))
        self.assertEqual(len(info["train_loss_by_epoch"]), 2)

if __name__ == "__main__":
    unittest.main()
