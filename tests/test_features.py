"""src.pipeline.features 단위 테스트 (합성 데이터, 네트워크 없음).

    .venv/Scripts/python -m unittest
"""
import math
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.features import FEATURES, build_features, load_panel
from src.pipeline.target import forward_yz


def make_prices(n_days=60, tickers=("A", "B"), seed=0):
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range("2022-01-03", periods=n_days)
    rows = []
    for t in tickers:
        close = 100.0
        for d in cal:
            o = close * math.exp(rng.normal(0, 0.01))
            c = o * math.exp(rng.normal(0, 0.02))
            h = max(o, c) * math.exp(abs(rng.normal(0, 0.01)))
            l = min(o, c) * math.exp(-abs(rng.normal(0, 0.01)))
            rows.append((d, t, o, h, l, c, 1000.0 + rng.integers(0, 500), c * 1e6, 1e6))
            close = c
    cols = ["date", "ticker", "open", "high", "low", "close", "volume", "mcap", "shares"]
    return cal, pd.DataFrame(rows, columns=cols)


class FeatureTest(unittest.TestCase):
    def setUp(self):
        self.cal, self.prices = make_prices()
        self.members = pd.DataFrame({"date": self.cal[30:], "ticker": "A"})
        self.f = build_features(self.prices, self.cal, self.members)

    def row(self, ticker, i):
        return self.f[(self.f["ticker"] == ticker) & (self.f["date"] == self.cal[i])].iloc[0]

    def test_rv_and_har(self):
        g = self.prices[self.prices["ticker"] == "A"].reset_index(drop=True)

        def rv(i):
            O, H, L, C, C0 = g.loc[i, "open"], g.loc[i, "high"], g.loc[i, "low"], g.loc[i, "close"], g.loc[i - 1, "close"]
            o, c, u, d = math.log(O / C0), math.log(C / O), math.log(H / O), math.log(L / O)
            return o * o + u * (u - c) + d * (d - c)

        r = self.row("A", 30)
        self.assertAlmostEqual(r["rv_d"], rv(30))
        self.assertAlmostEqual(r["rv_w"], np.mean([rv(i) for i in range(26, 31)]))
        self.assertAlmostEqual(r["rv_m"], np.mean([rv(i) for i in range(9, 31)]))
        self.assertAlmostEqual(r["ret"], math.log(g.loc[30, "close"] / g.loc[29, "close"]))
        self.assertTrue(math.isnan(self.row("A", 0)["rv_d"]))  # 첫날은 전일 종가가 없음
        self.assertTrue(math.isnan(self.row("A", 20)["rv_m"]))  # 22일이 안 됨

    def test_yz_past_is_previous_target(self):
        t = forward_yz(self.prices, self.cal, 5)
        prev = t[(t["ticker"] == "A") & (t["origin_date"] == self.cal[25])]["yz_var"].iloc[0]
        self.assertAlmostEqual(self.row("A", 30)["yz_5"], prev)  # 구간 26 ~ 30 = origin 25의 타깃

    def test_member_and_n_obs(self):
        self.assertTrue(self.row("A", 30)["is_member"])
        self.assertFalse(self.row("A", 29)["is_member"])
        self.assertFalse(self.row("B", 40)["is_member"])
        self.assertEqual(self.row("B", 40)["n_obs"], 41)

    def test_no_future_leak(self):
        cut = self.cal[40]
        p2 = self.prices.copy()
        after = p2["date"] > cut
        p2.loc[after, ["open", "high", "low", "close"]] *= 3.0
        p2.loc[after, "volume"] *= 10
        f2 = build_features(p2, self.cal, self.members)
        a = self.f[self.f["date"] <= cut].reset_index(drop=True)
        b = f2[f2["date"] <= cut].reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b)
        self.assertFalse(np.allclose(self.f["rv_d"].fillna(0), f2["rv_d"].fillna(0)))  # 컷오프 이후는 실제로 바뀜

    def test_missing_and_corp_action(self):
        p = self.prices.copy()
        halt = (p["ticker"] == "A") & (p["date"] == self.cal[30])
        p.loc[halt, ["open", "high", "low", "close", "volume"]] = np.nan
        p["corp_action"] = (p["ticker"] == "B") & (p["date"] == self.cal[35])
        f = build_features(p, self.cal, self.members)
        a = f[f["ticker"] == "A"].set_index("date")
        self.assertTrue(a.loc[self.cal[30:35], "rv_w"].isna().all())  # 결측일이 든 5일 창
        self.assertFalse(math.isnan(a.loc[self.cal[36], "rv_w"]))  # 31일은 전일 종가가 없어 rv_d 결측 → 36일부터
        b = f[f["ticker"] == "B"].set_index("date")
        self.assertTrue(math.isnan(b.loc[self.cal[35], "rv_d"]))
        self.assertTrue(math.isnan(b.loc[self.cal[35], "ret"]))
        self.assertFalse(math.isnan(b.loc[self.cal[35], "log_turnover"]))

    def test_base_price_drives_overnight_and_close_return(self):
        p = self.prices.copy()
        day = self.cal[35]
        row = (p["ticker"] == "A") & (p["date"] == day)
        p["base_price"] = p.groupby("ticker")["close"].shift(1)
        p["base_price_valid"] = p["base_price"].notna()
        p.loc[row, ["open", "high", "low", "close", "base_price"]] = [50, 53, 49, 52, 50]
        p["corp_action"] = row  # 기준가가 있으면 기존 휴리스틱은 계산을 막지 않는다
        f = build_features(p, self.cal, self.members)
        got = f[(f["ticker"] == "A") & (f["date"] == day)].iloc[0]
        c, u, d = math.log(52 / 50), math.log(53 / 50), math.log(49 / 50)
        self.assertAlmostEqual(got["ret"], math.log(52 / 50))
        self.assertAlmostEqual(got["rv_d"], u * (u - c) + d * (d - c))


class LoadPanelTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_load_panel(self):
        cal, prices = make_prices(n_days=40, tickers=("A",))
        build_features(prices, cal, pd.DataFrame(columns=["date", "ticker"])).to_parquet(self.tmp / "features.parquet")
        t = forward_yz(prices, cal, 5).assign(split="train", embargo=False)
        t.loc[10, "yz_var"] = 0.0
        t.to_parquet(self.tmp / "targets_h5.parquet")
        p = load_panel(self.tmp, 5)
        self.assertEqual(len(p), 40)
        self.assertEqual(list(p.columns), ["ticker", "origin_date", "horizon"] + FEATURES
                         + ["yz_var", "log_var", "split", "embargo"])
        self.assertTrue(math.isnan(p.loc[10, "log_var"]))
        self.assertAlmostEqual(p.loc[11, "log_var"], math.log(p.loc[11, "yz_var"]))


if __name__ == "__main__":
    unittest.main()
