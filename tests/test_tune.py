"""src.tune·registry·7.4 선택 단위 테스트 (합성 데이터, 네트워크 없음).

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

from src.evaluate.selection import horizon_choice
from src.evaluate.sensitivity import event_window_rows
from src.models import registry
from src.pipeline.features import build_features
from src.pipeline.target import forward_yz
from src.tune.__main__ import tune
from src.tune.runner import expand
from tests.test_features import make_prices


def write_data_dir(d, n_days=120):
    cal, prices = make_prices(n_days=n_days, tickers=("A", "B", "C"))
    build_features(prices, cal, pd.DataFrame(columns=["date", "ticker"])).to_parquet(d / "features.parquet")
    t = forward_yz(prices, cal, 5)
    t["split"] = np.where(t["origin_date"] < cal[70], "train", np.where(t["origin_date"] < cal[100], "val", "test"))
    t["embargo"] = False
    t.to_parquet(d / "targets_h5.parquet")
    ph = pd.DataFrame([{"event_id": 0, "ticker": "A", "direction": "편입", "phase": "short_shock",
                        "start": cal[75], "end": cal[85], "complete": True, "split": "val", "crosses_split": False}])
    ph.to_parquet(d / "event_phases.parquet")
    (d / "meta.json").write_text(json.dumps({"reportable": False, "hold_reasons": ["test"],
                                             "config": {"split": "test"}}), encoding="utf-8")
    return cal


class RegistryTest(unittest.TestCase):
    def test_smearing_uses_train_only(self):
        p = pd.DataFrame({"split": ["train", "train", "val"], "embargo": [False, False, False],
                          "log_var": [1.0, 3.0, 100.0]})
        c = registry.smearing_constant(p, np.array([1.0, 1.0, 0.0]))
        self.assertAlmostEqual(c, math.log((1 + math.exp(2)) / 2))  # 검증 행(100)은 쓰지 않음

    def test_expand(self):
        self.assertEqual(expand({}), [{}])
        self.assertEqual(len(expand({"a": [1, 2], "b": [3, 4, 5]})), 6)


class TuneTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.cal = write_data_dir(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_tune_records_table_and_best(self):
        table = tune("ewma", 5, self.tmp, {"lam": [0.9, 0.97]}, self.tmp / "tuning", self.tmp / "preds")
        self.assertEqual(len(table), 4)  # 설정 2 × 편향 보정 2
        self.assertTrue((table["qlike_val"].diff().dropna() >= 0).all())  # 순위대로 정렬
        self.assertEqual(table["n_val"].nunique(), 1)  # 모든 설정이 같은 표본
        out = self.tmp / "tuning" / "ewma_h5"
        self.assertTrue((out / "results.md").exists())
        self.assertEqual(len(list((out / "val_preds").glob("*.parquet"))), 2)
        best = pd.read_parquet(self.tmp / "preds" / "EWMA_h5.parquet")
        self.assertEqual(list(best.columns), ["ticker", "origin_date", "horizon", "pred_log_var", "model", "seed",
                                              "version"])
        meta = json.loads((self.tmp / "preds" / "EWMA_h5.meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["config_id"], table.loc[0, "config_id"])
        self.assertFalse(meta["reportable"])


class SelectionTest(unittest.TestCase):
    def test_horizon_choice(self):
        s = pd.DataFrame({"horizon": [5] * 4 + [22] * 4, "model": ["EWMA", "HAR", "LightGBM", "LSTM"] * 2,
                          "qlike": [1.0, 0.9, 1.1, 0.8, 1.0, 0.7, 0.8, 0.9]})
        hc = horizon_choice(s).set_index("horizon")
        self.assertAlmostEqual(hc.loc[5, "improvement"], (0.1 - 0.1 + 0.2) / 3)
        self.assertAlmostEqual(hc.loc[22, "improvement"], (0.3 + 0.2 + 0.1) / 3)
        self.assertTrue(hc.loc[22, "chosen"])
        self.assertFalse(hc.loc[5, "chosen"])

    def test_event_window_rows(self):
        cal = pd.bdate_range("2023-01-02", periods=10)
        panel = pd.DataFrame({"ticker": ["A"] * 10 + ["B"] * 10, "origin_date": list(cal) * 2, "split": "val"})
        ph = pd.DataFrame([{"ticker": "A", "direction": "편입", "start": cal[2], "end": cal[4], "complete": True,
                            "split": "val", "crosses_split": False},
                           {"ticker": "B", "direction": "편출", "start": cal[2], "end": cal[4], "complete": True,
                            "split": "val", "crosses_split": False},
                           {"ticker": "A", "direction": "편입", "start": cal[7], "end": cal[8], "complete": False,
                            "split": "val", "crosses_split": False}])
        m = event_window_rows(panel, ph)
        self.assertEqual(m.sum(), 3)  # A의 2~4일만 (편출·불완전 구간 제외)


if __name__ == "__main__":
    unittest.main()
