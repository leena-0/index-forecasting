"""src.models 단위 테스트 (합성 데이터, 네트워크 없음).

    .venv/Scripts/python -m unittest
"""
import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.models import ewma, har
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


if __name__ == "__main__":
    unittest.main()
