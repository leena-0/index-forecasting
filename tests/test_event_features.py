"""이벤트 단위 변수(이슈 #50 규칙 4) 단위 테스트 (합성 데이터, 네트워크 없음).

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

from src.evaluate import walk_forward as wf
from src.models import gnn, lgbm, lstm, registry
from src.pipeline.event_features import EVENT_FEATURES, build_event_features, event_matrix
from src.pipeline.features import load_panel
from src.pipeline.windows import TEST_YEARS
from src.tune.__main__ import tune
from tests import test_event_sensitivity as wf_case
from tests.test_models import ml_panel
from tests.test_walk_forward import CAL as WF_CAL
from tests.test_walk_forward import write_data_dir

HAS_LGBM = importlib.util.find_spec("lightgbm") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None
CAL = pd.bdate_range("2020-01-01", "2020-12-31")
TICKERS = ["A", "B", "C", "D"]
KEYS = pd.DataFrame([(t, d) for t in TICKERS for d in CAL], columns=["ticker", "date"])


def make_events(rows):
    """(ticker, event_type, direction, 공지일 위치, 적용일 위치)"""
    return pd.DataFrame([{"event_id": i, "ticker": t, "event_type": ty, "direction": d,
                          "announce_date": CAL[a], "effective_date": CAL[f]} for i, (t, ty, d, a, f) in enumerate(rows)])


def build(events, known_from=None):
    return build_event_features(KEYS, events, CAL, known_from).set_index(["ticker", "date"])


class BuildTest(unittest.TestCase):
    def test_period_and_values(self):
        f = build(make_events([("A", "정기변경", "편입", 50, 62), ("B", "정기변경", "편출", 50, 62),
                               ("C", "수시변경", "편입", 100, 105)]))
        a = f.loc["A"]
        self.assertEqual(a.loc[CAL[50], "event_active"], 0)        # 공지 당일에는 쓰지 않는다
        self.assertEqual(list(np.flatnonzero(a["event_active"]))[0], 51)
        self.assertEqual(list(np.flatnonzero(a["event_active"]))[-1], 62 + 21)
        self.assertEqual(a.loc[CAL[51], "event_dist"], -11)
        self.assertEqual(a.loc[CAL[62], ["event_active", "event_dist"]].tolist(), [1, 0])  # 적용일: 거리 0, active 1
        self.assertEqual(a.loc[CAL[83], "event_dist"], 21)
        self.assertEqual((a.loc[CAL[60], "event_dir"], f.loc[("B", CAL[60]), "event_dir"]), (1, -1))
        self.assertEqual((a.loc[CAL[60], "event_type"], f.loc[("C", CAL[103]), "event_type"]), (1, 0))
        self.assertAlmostEqual(a.loc[CAL[60], "event_n"], math.log(3))  # A, B 두 종목
        self.assertAlmostEqual(f.loc[("C", CAL[103]), "event_n"], math.log(2))
        self.assertTrue((f.loc["D"] == 0).all().all())
        self.assertFalse(f.isna().any().any())

    def test_holiday_announcement_starts_next_trading_day(self):
        ev = make_events([("A", "정기변경", "편입", 50, 62)])
        ev["announce_date"] = CAL[CAL.dayofweek == 4][8] + pd.Timedelta(days=1)  # 토요일 공지
        f = build(ev).loc["A"]
        first = f.index[np.flatnonzero(f["event_active"])[0]]
        self.assertEqual(first, ev["announce_date"].iloc[0] + pd.Timedelta(days=2))  # 월요일

    def test_later_announced_event_wins(self):
        f = build(make_events([("A", "정기변경", "편입", 50, 62), ("A", "수시변경", "편출", 70, 75)])).loc["A"]
        self.assertEqual(f.loc[CAL[70], "event_dir"], 1)   # 둘째 공지 당일까지는 첫 이벤트
        self.assertEqual(f.loc[CAL[71], ["event_dir", "event_type", "event_dist"]].tolist(), [-1, 0, -4])

    def test_known_from_delays_only(self):
        ev = make_events([("A", "수시변경", "편입", 50, 62)])
        f = build(ev, {str(CAL[62].date()): str(CAL[53].date())}).loc["A"]
        self.assertEqual(np.flatnonzero(f["event_active"])[0], 54)
        f = build(ev, {str(CAL[62].date()): str(CAL[40].date())}).loc["A"]  # 공지일보다 빠르면 공지일
        self.assertEqual(np.flatnonzero(f["event_active"])[0], 51)

    def test_no_leakage(self):
        base = make_events([("A", "정기변경", "편입", 50, 62), ("B", "정기변경", "편출", 50, 62),
                            ("A", "수시변경", "편출", 70, 75), ("C", "정기변경", "편입", 58, 62)])
        f = build(base)
        for cut in (50, 55, 58, 70):  # cut일에 공지된 정보까지 바꿔도 cut일까지의 변수는 같아야 한다
            changed = base.copy()
            late = changed["announce_date"] >= CAL[cut]
            changed.loc[late, "effective_date"] = CAL[200]
            changed.loc[late, "direction"], changed.loc[late, "event_type"] = "편출", "특례편입"
            extra = make_events([("D", "정기변경", "편입", cut, 62), ("D", "수시변경", "편출", cut + 1, 62)])
            changed = pd.concat([changed, extra.assign(event_id=[90, 91])], ignore_index=True)
            removed = base[~late]
            for other in (changed, removed):
                g = build(other)
                keep = f.index.get_level_values("date") <= CAL[cut]
                pd.testing.assert_frame_equal(f[keep], g[keep])
        # 같은 적용일이라도 나중에 공지된 종목은 그 전의 event_n에 세지 않는다
        self.assertAlmostEqual(f.loc[("A", CAL[58]), "event_n"], math.log(3))
        self.assertAlmostEqual(f.loc[("A", CAL[59]), "event_n"], math.log(4))


def with_events(panel):
    days = np.sort(panel["origin_date"].unique())
    rng = np.random.default_rng(0)
    p = panel.copy()
    on = p["origin_date"].isin(days[30:45]) & (p["ticker"] == "A")
    for c in EVENT_FEATURES:
        p[c] = np.where(on, rng.normal(size=len(p)), 0.0)
    p["event_active"] = on.astype(float)
    return p


class ModelInputTest(unittest.TestCase):
    def test_inputs_grow_only_with_event_columns(self):
        p = ml_panel()
        q = with_events(p)
        self.assertEqual(event_matrix(p).shape, (len(p), 0))
        self.assertEqual(event_matrix(q).shape, (len(p), 5))
        self.assertEqual(list(lgbm.design(q).columns)[-5:], EVENT_FEATURES)
        self.assertEqual(lgbm.design(p).shape[1] + 5, lgbm.design(q).shape[1])
        self.assertEqual(gnn.node_features(p).shape[1] + 5, gnn.node_features(q).shape[1])
        xp, rp = lstm.sequences(p, 5)
        xq, rq = lstm.sequences(q, 5)
        self.assertEqual((xp.shape[2], xq.shape[2]), (3, 8))
        np.testing.assert_array_equal(rp, rq)  # 변수에 결측이 없어 예측하는 행이 같다

    def test_distance_is_scaled(self):
        q = with_events(ml_panel())
        q["event_dist"] = 21.0
        self.assertTrue((event_matrix(q)[:, EVENT_FEATURES.index("event_dist")] == 1.0).all())

    def test_econometric_models_ignore_event_columns(self):
        p = ml_panel()
        for m in ("ewma", "har"):
            a, _ = registry.fit_predict(m, p)
            b, _ = registry.fit_predict(m, with_events(p))
            pd.testing.assert_series_equal(a, b)

    @unittest.skipUnless(HAS_LGBM, "lightgbm 없음 (requirements-models.txt)")
    def test_lgbm_runs(self):
        pred, _ = registry.fit_predict("lgbm", with_events(ml_panel()), {"num_boost_round": 3, "min_data_in_leaf": 5})
        self.assertTrue(pred.notna().all())

    @unittest.skipUnless(HAS_TORCH, "torch 없음 (requirements-models.txt)")
    def test_neural_models_run(self):
        p = ml_panel()
        for m in ("lstm", "gru", "tcn", "transformer"):
            tiny = {"lookback": 5, "hidden": 4, "epochs": 1, "batch_size": 16}
            a, _ = registry.fit_predict(m, p, tiny)
            b, info = registry.fit_predict(m, with_events(p), tiny)
            self.assertEqual(info["n_inputs"], 8)
            np.testing.assert_array_equal(a.notna(), b.notna())
            self.assertTrue(np.isfinite(b.dropna()).all())
        pred, info = registry.fit_predict("gnn", with_events(p), {"hidden": 4, "epochs": 1, "k": 1})
        self.assertEqual(info["n_inputs"], len(gnn.NODE_FEATURES) + 5)


class TuneEventTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        write_data_dir(self.tmp)
        f = pd.read_parquet(self.tmp / "features.parquet", columns=["ticker", "date"])
        ev = pd.DataFrame([{"event_id": 0, "ticker": "A", "event_type": "정기변경", "direction": "편입",
                            "announce_date": WF_CAL[900], "effective_date": WF_CAL[912]}])
        build_event_features(f, ev, WF_CAL).to_parquet(self.tmp / "event_features.parquet")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_panel_columns(self):
        self.assertFalse(set(EVENT_FEATURES) & set(load_panel(self.tmp, 5, 2020)))
        p = load_panel(self.tmp, 5, 2020, event_features=True)
        self.assertEqual(list(p.columns[-5:]), EVENT_FEATURES)
        self.assertEqual(int(p["event_active"].sum()), 12 + 21)

    def test_rejects_models_without_event_inputs(self):
        with self.assertRaises(ValueError):
            tune("har", 5, self.tmp, {}, self.tmp / "t", self.tmp / "p", window=2020, event_features=True)

    @unittest.skipUnless(HAS_LGBM, "lightgbm 없음 (requirements-models.txt)")
    def test_names_and_folders(self):
        grid = {"num_boost_round": [2], "min_data_in_leaf": [30]}
        for ev in (False, True):
            tune("lgbm", 5, self.tmp, grid, self.tmp / "t", self.tmp / "p", window=2020, event_features=ev)
        self.assertTrue((self.tmp / "t" / "w2020" / "lgbm_ev_h5" / "results.csv").exists())
        self.assertTrue((self.tmp / "t" / "w2020" / "lgbm_h5" / "results.csv").exists())
        pred = pd.read_parquet(self.tmp / "p" / "w2020" / "LightGBM+EV_h5.parquet")
        self.assertEqual(pred["model"].unique().tolist(), ["LightGBM+EV"])
        meta = json.loads((self.tmp / "p" / "w2020" / "LightGBM+EV_h5.meta.json").read_text(encoding="utf-8"))
        self.assertTrue(meta["event_features"])


class EvaluateMainTest(unittest.TestCase):
    setUp, tearDown = wf_case.WalkForwardTest.setUp, wf_case.WalkForwardTest.tearDown

    def test_main_keeps_primary_without_events(self):
        for y in TEST_YEARS:
            d = self.tmp / "preds" / f"w{y}"
            p = pd.read_parquet(d / "M1_h5.parquet")
            # 변수를 넣은 예측은 B 종목이 없다 → 함께 평가하면 공통 표본이 줄지만 주 결과는 그대로여야 한다
            p[p["ticker"] != "B"].assign(model="M1+EV", pred_log_var=0.0).to_parquet(d / "M1+EV_h5.parquet", index=False)
        wf.main(["--preds", str(self.tmp / "preds"), "--data-dir", str(self.tmp), "--out", str(self.tmp / "out"),
                 "--B", "20", "--final"])
        s = pd.read_csv(self.tmp / "out" / "summary.csv")
        e = pd.read_csv(self.tmp / "out" / "with_events" / "summary.csv")
        self.assertEqual(sorted(s["model"]), ["M1", "M2"])
        self.assertEqual(sorted(e["model"]), ["M1", "M1+EV", "M2"])
        self.assertEqual(s["n"].iloc[0] * 2, e["n"].iloc[0] * 3)
        t = pd.read_csv(self.tmp / "out" / "with_events" / "event_feature_effect.csv")
        self.assertEqual(t["model"].tolist(), ["M1"])
        self.assertGreater(t["qlike_without_minus_with"].iloc[0], 0)  # M1+EV는 실제값(분산 1)을 그대로 예측한다


class EvaluateEventTest(unittest.TestCase):
    def preds(self, *names):
        return {y: pd.DataFrame({"model": list(names), "horizon": 5}) for y in TEST_YEARS}

    def test_split_keeps_main_result_without_events(self):
        base, both = wf.split_event_preds(self.preds("EWMA", "LSTM", "LSTM+EV"))
        self.assertEqual(sorted(base[2020]["model"]), ["EWMA", "LSTM"])
        self.assertEqual(len(both[2020]), 3)
        self.assertIsNone(wf.split_event_preds(self.preds("EWMA", "LSTM"))[1])
        with self.assertRaises(ValueError):
            wf.split_event_preds(self.preds("EWMA", "LSTM+EV"))

    def test_effect_compares_same_model(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            pd.DataFrame([("LSTM", "LSTM+EV", 0.01, 2.0, 0.04, 100), ("EWMA", "LSTM+EV", 0.5, 9.0, 0.0, 100)],
                         columns=["model_1", "model_2", "mean_diff", "stat", "p_value", "n"]).to_csv(tmp / "dm_h5.csv")
            t = wf.event_feature_effect(tmp, [5])
            self.assertEqual(t["model"].tolist(), ["LSTM"])
            self.assertEqual(t["qlike_without_minus_with"].iloc[0], 0.01)
        finally:
            shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main()
