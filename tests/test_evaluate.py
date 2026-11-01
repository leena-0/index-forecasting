"""src.evaluate 단위 테스트 (합성 데이터, 네트워크 없음).

    .venv/Scripts/python -m unittest
"""
import math
import unittest

import numpy as np
import pandas as pd

from src.evaluate.__main__ import main as evaluate_main
from src.evaluate.compare import block_bootstrap_index, daily_losses, dm_test, mcs, newey_west_var
from src.evaluate.losses import loss_panel, mse, qlike
from src.evaluate.sensitivity import control_flag, event_control_losses, no_baseline_post, sensitivity


def make_pred(targets, model, log_var, seed=0):
    return pd.DataFrame({"ticker": targets["ticker"], "origin_date": targets["origin_date"],
                         "horizon": targets["horizon"], "pred_log_var": log_var, "model": model,
                         "seed": seed, "version": "test"})


class LossTest(unittest.TestCase):
    def test_qlike_mse(self):
        self.assertAlmostEqual(qlike(2.0, 2.0), 0.0)
        r = 3.0 / 2.0
        self.assertAlmostEqual(qlike(3.0, 2.0), r - math.log(r) - 1)
        self.assertGreater(qlike(1.0, 4.0), 0)
        self.assertAlmostEqual(mse(3.0, 2.0), 1.0)

    def test_loss_panel(self):
        d = pd.bdate_range("2023-01-02", periods=4)
        t = pd.DataFrame({"ticker": "A", "origin_date": d, "horizon": 5, "yz_var": [1e-4, 2e-4, 0.0, 1e-4],
                          "split": "val", "embargo": [False, False, False, True]})
        a0 = make_pred(t, "M1", np.log(1e-4), seed=0)
        a1 = make_pred(t, "M1", np.log(2e-4), seed=1)
        b = make_pred(t.iloc[:1], "M2", np.log(1e-4))
        panel, c = loss_panel(pd.concat([a0, a1, b]), t)
        # 0 이하(셋째 날)·경계 공백(넷째 날) 제외 → M1(seed 2개) 4행 → M1 2날짜, M2 1날짜, 공통 집합은 첫날만
        self.assertEqual(c["no_usable_target"], 4)
        self.assertEqual(c["dropped_not_common"], 1)
        self.assertEqual(len(panel), 2)
        m1 = panel[panel["model"] == "M1"].iloc[0]
        self.assertEqual(m1["n_seeds"], 2)  # seed별 손실의 평균
        self.assertAlmostEqual(m1["qlike"], (qlike(1e-4, 1e-4) + qlike(1e-4, 2e-4)) / 2)

    def test_check_predictions(self):
        t = pd.DataFrame({"ticker": ["A"], "origin_date": [pd.Timestamp("2023-01-02")], "horizon": [5],
                          "yz_var": [1e-4], "split": ["val"], "embargo": [False]})
        p = make_pred(t, "M1", [0.0])
        with self.assertRaises(ValueError):
            loss_panel(pd.concat([p, p]), t)  # 중복
        with self.assertRaises(ValueError):
            loss_panel(p.drop(columns="version"), t)
        with self.assertRaises(ValueError):
            loss_panel(p.assign(horizon=10), t)


class CompareTest(unittest.TestCase):
    def test_newey_west(self):
        d = np.array([1.0, -1.0, 2.0, 0.0, 3.0])
        e = d - d.mean()
        self.assertAlmostEqual(newey_west_var(d, 0), e @ e / 5)
        g1 = e[1:] @ e[:-1] / 5
        self.assertAlmostEqual(newey_west_var(d, 1), e @ e / 5 + 2 * 0.5 * g1)

    def test_dm(self):
        rng = np.random.default_rng(0)
        base = rng.gamma(2.0, 1.0, 500)
        r = dm_test(base, base + 0.5 + rng.normal(0, 0.1, 500), h=5)
        self.assertLess(r["stat"], 0)  # 모델 1의 손실이 작다
        self.assertLess(r["p_value"], 0.01)
        r2 = dm_test(base, base + rng.normal(0, 1, 500), h=5)
        self.assertGreater(r2["p_value"], 0.01)

    def test_block_bootstrap_index(self):
        idx = block_bootstrap_index(10, 3, 4, np.random.default_rng(0))
        self.assertEqual(idx.shape, (4, 10))
        for row in idx:  # 블록 안에서는 원형으로 1씩 증가
            for s in range(0, 9, 3):
                blk = row[s:s + 3]
                self.assertTrue(all((blk[i + 1] - blk[i]) % 10 == 1 for i in range(len(blk) - 1)))

    def test_mcs(self):
        rng = np.random.default_rng(1)
        T = 400
        base = rng.gamma(2.0, 1.0, T)
        daily = pd.DataFrame({"good": base, "good2": base + rng.normal(0, 0.05, T),
                              "bad": base + 1.0 + rng.normal(0, 0.1, T)})
        res = mcs(daily, alpha=0.10, B=500, block=5, seed=0).set_index("model")
        self.assertFalse(res.loc["bad", "in_mcs"])
        self.assertEqual(res.loc["bad", "eliminated_order"], 1)
        self.assertTrue(res.loc["good", "in_mcs"])
        self.assertTrue(res.loc["good2", "in_mcs"])

    def test_daily_losses(self):
        panel = pd.DataFrame({"origin_date": pd.to_datetime(["2023-01-02"] * 2 + ["2023-01-03"]),
                              "ticker": ["A", "B", "A"], "horizon": 5, "split": "val", "model": "M1",
                              "qlike": [1.0, 3.0, 5.0]})
        d = daily_losses(panel)
        self.assertEqual(d["M1"].tolist(), [2.0, 5.0])


class SensitivityTest(unittest.TestCase):
    def setUp(self):
        # 달력 20일. E: 편입 이벤트 종목 (기준 0~4, 단기 충격 10~14), C1·C2: 대조군, X: 편출 이벤트 종목
        self.cal = pd.bdate_range("2023-01-02", periods=20)
        cal = self.cal
        rows = []
        for i, d in enumerate(cal):
            for tk in ["E", "C1", "C2", "X"]:
                q = 1.0
                if tk == "E":
                    q = 4.0 if 10 <= i <= 14 else 2.0
                if tk == "C1" and 10 <= i <= 14:
                    q = 2.0
                if tk == "X":
                    q = 100.0  # 이벤트 윈도우 안이라 대조군에서 빠져야 함
                rows.append((tk, d, q))
        self.panel = pd.DataFrame(rows, columns=["ticker", "origin_date", "qlike"]).assign(
            horizon=5, split="val", model="M1", mse=0.0)
        mem = [(d, tk) for i, d in enumerate(cal) for tk in ["C1", "C2", "X"] + (["E"] if i >= 10 else [])]
        self.members = pd.DataFrame(mem, columns=["date", "ticker"])
        ph = []
        for eid, tk, direction in [(0, "E", "편입"), (1, "X", "편출")]:
            for phase, s, e, comp in [("baseline", 0, 4, True), ("short_shock", 10, 14, True), ("post", 15, 19, True)]:
                ph.append({"event_id": eid, "ticker": tk, "event_type": "정기변경", "direction": direction,
                           "phase": phase, "start": cal[s], "end": cal[e], "complete": comp,
                           "split": "val", "crosses_split": False})
        self.phases = pd.DataFrame(ph)

    def test_control_flag(self):
        f = control_flag(self.panel, self.members, self.phases)
        ctrl = self.panel[f]
        self.assertEqual(set(ctrl["ticker"]), {"C1", "C2"})  # E는 윈도우 안, X는 편출 윈도우 안
        self.assertEqual(len(ctrl), 2 * 20)

    def test_sensitivity(self):
        ec = event_control_losses(self.panel, self.members, self.phases)
        self.assertEqual(set(ec["event_id"]), {0})  # 편입만
        s = sensitivity(ec).set_index("phase")
        # 단기 충격: L^E 4/2 = 2배, L^C (2+1)/2 / 1 = 1.5배 → ln 2 − ln 1.5
        self.assertAlmostEqual(s.loc["short_shock", "sens"], math.log(2) - math.log(1.5))
        self.assertAlmostEqual(s.loc["post", "sens"], 0.0)
        self.assertEqual(s.loc["short_shock", "n_events"], 1)

    def test_no_baseline(self):
        ph = self.phases.copy()
        ph.loc[(ph["event_id"] == 0) & (ph["phase"] == "baseline"), "complete"] = False
        ec = event_control_losses(self.panel, self.members, ph)
        self.assertTrue(sensitivity(ec).empty)
        nb = no_baseline_post(ec)
        self.assertAlmostEqual(nb.iloc[0]["log_ratio"], math.log(2.0 / 1.0))


class CliTest(unittest.TestCase):
    def test_test_split_needs_final(self):
        with self.assertRaises(SystemExit):
            evaluate_main(["--pred", "없음.parquet", "--split", "test"])


if __name__ == "__main__":
    unittest.main()
