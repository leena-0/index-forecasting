"""슬라이딩 윈도우(이슈 #50) 단위 테스트 (합성 데이터, 네트워크 없음).

    .venv/Scripts/python -m unittest
"""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.models import registry
from src.pipeline.features import FEATURES, load_panel
from src.pipeline.windows import TEST_YEARS, assign_window, window_split_of
from src.tune.__main__ import tune

CAL = pd.bdate_range("2015-01-01", "2024-12-31")


def make_targets(h, tickers=("A",)):
    out = []
    for tk in tickers:
        t = pd.DataFrame({"ticker": tk, "origin_date": CAL, "horizon": h, "yz_var": 1e-3})
        t.loc[len(CAL) - h:, "yz_var"] = np.nan
        out.append(t)
    return pd.concat(out, ignore_index=True).assign(split="fixed", embargo=True)  # 고정 분할 열은 무시돼야 한다


def write_data_dir(d, h=5, seed=0):
    rng = np.random.default_rng(seed)
    targets = make_targets(h, ("A", "B"))
    f = targets[["ticker", "origin_date"]].rename(columns={"origin_date": "date"})
    for c in FEATURES:
        f[c] = np.exp(rng.normal(-7, 0.5, len(f)))
    f["is_member"], f["n_obs"] = True, np.tile(np.arange(len(CAL)), 2)
    targets["yz_var"] = targets["yz_var"].where(targets["yz_var"].isna(), np.exp(rng.normal(-7, 0.5, len(f))))
    f.to_parquet(d / "features.parquet")
    targets.to_parquet(d / f"targets_h{h}.parquet")
    (d / "meta.json").write_text(json.dumps({"reportable": False, "hold_reasons": ["test"], "config": {}}),
                                 encoding="utf-8")


class WindowSplitTest(unittest.TestCase):
    def test_years(self):
        d = pd.to_datetime(["2015-12-31", "2016-01-01", "2018-12-31", "2019-01-01", "2019-12-31", "2020-06-01",
                            "2021-12-31", "2022-01-03"])
        self.assertEqual(list(window_split_of(d, 2020)),
                         ["pre", "train", "train", "val", "val", "test", "next", "after"])

    def test_all_windows_three_one_one(self):
        self.assertEqual(TEST_YEARS, (2020, 2021, 2022, 2023, 2024))
        for y in TEST_YEARS:
            s = pd.Series(window_split_of(CAL, y), index=CAL.year).groupby(level=0).first()
            self.assertEqual(list(s[s == "train"].index), [y - 4, y - 3, y - 2])
            self.assertEqual(list(s[s == "val"].index), [y - 1])
            self.assertEqual(list(s[s == "test"].index), [y])

    def test_rejects_other_year(self):
        with self.assertRaises(ValueError):
            assign_window(make_targets(5), CAL, 2019)

    def test_embargo_at_fitted_boundaries_only(self):
        for h in (5, 22):
            t = assign_window(make_targets(h), CAL, 2020).set_index("origin_date")
            self.assertFalse((t["split"] == "after").any())
            self.assertEqual(t.index.max().year, 2021)
            for split in ("train", "val"):  # 학습·검증 끝의 h일은 비운다
                g = t[t["split"] == split]
                self.assertTrue(g["embargo"].iloc[-h:].all())
                self.assertFalse(g["embargo"].iloc[:-h].any())
            self.assertFalse(t.loc[t["split"] == "test", "embargo"].any())  # 테스트끼리 잇는 곳은 비우지 않는다
            last = assign_window(make_targets(h), CAL, 2024).set_index("origin_date")
            g = last[last["split"] == "test"]  # 달력 밖으로 나가는 타깃은 뺀다
            self.assertTrue(g["embargo"].iloc[-h:].all())
            self.assertFalse(g["embargo"].iloc[:-h].any())


class WindowPanelTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        write_data_dir(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_load_panel_uses_window(self):
        p = load_panel(self.tmp, 5, 2021)
        self.assertEqual(sorted(p.loc[p["split"] == "train", "origin_date"].dt.year.unique()), [2017, 2018, 2019])
        self.assertEqual(sorted(p["split"].unique()), ["next", "pre", "test", "train", "val"])
        self.assertEqual(load_panel(self.tmp, 5)["split"].unique().tolist(), ["fixed"])  # 기본은 고정 분할 그대로

    def test_fit_ignores_rows_outside_train(self):
        p = load_panel(self.tmp, 5, 2020)
        q = p.copy()
        out = q["split"] != "train"
        q.loc[out, "log_var"] = q.loc[out, "log_var"] + 10  # 학습 밖 타깃을 바꿔도 예측이 같아야 한다
        a, _ = registry.fit_predict("har", p, {"smearing": True})
        b, _ = registry.fit_predict("har", q, {"smearing": True})
        pd.testing.assert_series_equal(a, b)

    def test_tune_writes_window_outputs(self):
        table = tune("ewma", 5, self.tmp, {"lam": [0.9, 0.97]}, self.tmp / "tuning", self.tmp / "preds", window=2020)
        self.assertTrue(table["qlike_event"].isna().all())
        n_val = int((CAL.year == 2019).sum()) - 5  # 검증 = 2019년, 끝 5일은 공백
        self.assertEqual(table["n_val"].unique().tolist(), [2 * n_val])
        pred = pd.read_parquet(self.tmp / "preds" / "w2020" / "EWMA_h5.parquet")
        self.assertEqual(sorted(pred["origin_date"].dt.year.unique()), [2019, 2020, 2021])
        meta = json.loads((self.tmp / "preds" / "w2020" / "EWMA_h5.meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["window"], 2020)
        self.assertTrue((self.tmp / "tuning" / "w2020" / "ewma_h5" / "results.csv").exists())


if __name__ == "__main__":
    unittest.main()
