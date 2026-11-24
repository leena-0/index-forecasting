"""슬라이딩 윈도우 평가와 이벤트별 민감도(이슈 #50 규칙 1~3) 단위 테스트 (합성 데이터, 네트워크 없음).

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

from src.evaluate import event_sensitivity as es_mod
from src.evaluate import walk_forward as wf
from src.evaluate.event_sensitivity import (aggregate, bootstrap_ci, control_windows, event_losses, event_sens,
                                            leave_one_out, no_baseline_post, select_events, target_phases, usage)
from src.evaluate.losses import qlike
from src.pipeline.windows import TEST_YEARS, event_phases

CAL = pd.bdate_range("2015-01-01", "2024-12-31")
CFG = {"baseline": {"anchor": "announce", "range": [-120, -61]}, "anticipation": {"anchor": "announce", "range": [-60, -1]},
       "short_shock": {"anchor": "effective", "range": [0, 21]}, "post": {"anchor": "effective", "range": [22, 120]}}
SPLIT = {"train_end": "2021-12-31", "val_end": "2023-12-31"}


def make_phases(events, tickers):
    ev = pd.DataFrame(events, columns=["event_id", "ticker", "announce_date", "effective_date"])
    for c in ("announce_date", "effective_date"):
        ev[c] = pd.to_datetime(ev[c])
    ev = ev.assign(event_type="정기변경", direction="편입", announce_source="file")
    prices = pd.concat([pd.DataFrame({"date": CAL, "ticker": t}) for t in tickers])
    return event_phases(ev, CAL, prices, CFG, SPLIT)


class TargetPhaseTest(unittest.TestCase):
    def setUp(self):
        self.ph = make_phases([(0, "A", "2021-05-18", "2021-06-11")], ["A"])  # 공지~편입 18영업일

    def test_target_stays_inside_phase(self):
        for h in (5, 22):
            s = target_phases(self.ph, CAL, h).set_index("phase")
            for r in self.ph.itertuples():
                q = s.loc[r.phase]
                if q["n_max"] == 0:
                    continue
                i, j = CAL.get_loc(q["start"]), CAL.get_loc(q["end"])
                self.assertEqual(CAL[i + 1], r.start)     # 첫 타깃 날 = 구간 시작
                self.assertEqual(CAL[j + h], r.end)       # 마지막 타깃 날 = 구간 끝
                self.assertEqual(q["n_max"], j - i + 1)

    def test_h22_short_phases(self):
        n = target_phases(self.ph, CAL, 22).set_index("phase")["n_max"]
        self.assertEqual(n["announce_to_effective"], 0)
        self.assertEqual(n["short_shock"], 1)
        self.assertEqual(target_phases(self.ph, CAL, 5).set_index("phase")["n_max"]["baseline"], 56)


class SelectEventsTest(unittest.TestCase):
    def test_span_and_window(self):
        ph = make_phases([(0, "A", "2020-05-18", "2020-06-12"),   # 기준 구간이 2019년에 시작
                          (1, "A", "2020-11-18", "2020-12-11"),   # 2020-06 ~ 2021-05 → 윈도우 2020
                          (2, "A", "2021-05-18", "2021-06-11"),   # 2020-12 ~ 2021-11 → 윈도우 2020
                          (3, "A", "2024-11-18", "2024-12-13")], ["A"])  # 사후 구간이 as_of 이후
        e = select_events(ph, CAL).set_index("event_id")
        self.assertEqual(e.loc[0, "excluded"], "테스트 기간 전에 시작")
        self.assertEqual(e.loc[3, "excluded"], "테스트 기간 뒤까지 이어짐")
        self.assertEqual(list(e.loc[[1, 2], "excluded"]), ["", ""])
        self.assertEqual(list(e.loc[[1, 2], "window"]), [2020, 2020])

    def test_window_is_year_of_first_origin(self):
        # 기준 구간이 2021년 첫 거래일에 시작하면 첫 예측 시점(전날)은 2020년이다
        a = CAL[CAL.get_loc(CAL[CAL.year == 2021][0]) + 120]
        ph = make_phases([(0, "A", a, CAL[CAL.get_loc(a) + 10])], ["A"])
        self.assertEqual(select_events(ph, CAL)["window"].iloc[0], 2020)


def losses_table():
    """이벤트 3개 × 구간 2개 × 모델 1개. 손실 평균: (기준 E, 기준 C, 충격 E, 충격 C)."""
    rows = []
    for eid, (be, bc, se, sc, n) in {0: (1, 1, 2, 1, 10), 1: (1, 2, 3, 2, 10), 2: (1, 1, 100, 1, 5)}.items():
        rows.append((eid, "baseline", "M", 40, 40 * be, 40 * bc))
        rows.append((eid, "short_shock", "M", n, n * se, n * sc))
    return pd.DataFrame(rows, columns=["event_id", "phase", "model", "n", "sum_E", "sum_C"])


def shifted_table(n_max=56):
    return pd.DataFrame([{"event_id": e, "phase": p, "complete": True, "baseline_overlaps_prev": False,
                          "n_max": n_max if p == "baseline" else 18}
                         for e in range(3) for p in ("baseline", "short_shock")])


EVENTS = pd.DataFrame({"event_id": [0, 1, 2], "excluded": ["", "", ""],
                       "effective": pd.to_datetime(["2021-06-11", "2021-06-11", "2022-06-10"])})


class SensitivityTest(unittest.TestCase):
    def test_event_mean_median_pooled(self):
        ls = losses_table()
        use = usage(shifted_table(), EVENTS, ls, 0.5, 5)
        self.assertTrue((use["reason"] == "").all())
        es = event_sens(ls, use)
        self.assertEqual([round(x, 10) for x in es.sort_values("event_id")["sens"]],
                         [round(x, 10) for x in (math.log(2), math.log(3), math.log(100))])
        a = aggregate(es, EVENTS).iloc[0]
        self.assertAlmostEqual(a["sens"], (math.log(2) + math.log(3) + math.log(100)) / 3)
        self.assertAlmostEqual(a["median"], math.log(3))
        E, C, n = 20 + 30 + 500, 10 + 20 + 5, 25  # 이전 방식: 손실을 모두 합친 평균
        self.assertAlmostEqual(a["pooled"], math.log(E / n) - math.log(C / n) + math.log((40 * 4 / 120)))
        self.assertEqual((a["n_events"], a["n_effective_dates"]), (3, 2))
        self.assertAlmostEqual(a["loo_min"], (math.log(2) + math.log(3)) / 2)

    def test_conditions_and_reasons(self):
        ls = losses_table()
        ls.loc[(ls["event_id"] == 1) & (ls["phase"] == "baseline"), "n"] = 20  # 20/56 = 36%
        ls.loc[(ls["event_id"] == 2) & (ls["phase"] == "short_shock"), "n"] = 4
        def reason(base_min, phase_min):
            u = usage(shifted_table(), EVENTS, ls, base_min, phase_min)
            return u[u["phase"] == "short_shock"].set_index("event_id")["reason"].to_dict()
        self.assertEqual(reason(0.50, 5), {0: "", 1: "기준 구간 예측 비율 부족", 2: "구간 예측 수 부족"})
        self.assertEqual(reason(0.50, 1), {0: "", 1: "기준 구간 예측 비율 부족", 2: ""})
        self.assertEqual(reason(0.33, 5), {0: "", 1: "", 2: "구간 예측 수 부족"})
        self.assertEqual(reason(0.0, 5), {0: "", 1: "", 2: "구간 예측 수 부족"})
        ls = ls[~((ls["event_id"] == 0) & (ls["phase"] == "baseline"))]  # 조건이 없어도 1개는 있어야 한다
        self.assertEqual(reason(0.0, 5)[0], "기준 구간 예측 없음")

    def test_half_is_included(self):
        ls = losses_table()
        ls.loc[(ls["event_id"] == 0) & (ls["phase"] == "baseline"), "n"] = 28  # 정확히 50%
        u = usage(shifted_table(), EVENTS, ls, 0.5, 5)
        self.assertEqual(u.loc[u["event_id"] == 0, "reason"].tolist(), ["", ""])

    def test_incomplete_empty_phase_and_excluded_event(self):
        ls, sh = losses_table(), shifted_table()
        sh.loc[(sh["event_id"] == 0) & (sh["phase"] == "short_shock"), "complete"] = False
        sh.loc[(sh["event_id"] == 1) & (sh["phase"] == "short_shock"), "n_max"] = 0
        ev = EVENTS.assign(excluded=["", "", "테스트 기간 전에 시작"])
        u = usage(sh, ev, ls, 0.5, 5)
        self.assertEqual(u[u["phase"] == "short_shock"].set_index("event_id")["reason"].to_dict(),
                         {0: "구간 불완전", 1: "타깃이 구간에 들어가지 않음", 2: "테스트 기간 전에 시작"})
        self.assertTrue(event_sens(ls, u).empty)

    def test_overlapping_baseline(self):
        ls, sh = losses_table(), shifted_table()
        sh["baseline_overlaps_prev"] = sh["event_id"] == 1
        u = usage(sh, EVENTS, ls, 0.5, 5)
        self.assertEqual(u[u["phase"] == "short_shock"].set_index("event_id")["reason"].to_dict(),
                         {0: "", 1: "기준 구간이 같은 종목의 이전 이벤트와 겹침", 2: ""})
        self.assertEqual(sorted(event_sens(ls, u)["event_id"]), [0, 2])
        self.assertTrue((usage(sh, EVENTS, ls, 0.5, 5, True)["reason"] == "").all())  # 겹침 포함 (견고성)
        sh.loc[sh["event_id"] == 1, "complete"] = False  # 관찰되지 않은 기준 구간이 먼저다
        u = usage(sh, EVENTS, ls, 0.5, 5)
        self.assertEqual(u.loc[u["event_id"] == 1, "reason"].unique().tolist(), ["기준 구간 불완전"])

    def test_variants_change_one_condition(self):
        main = es_mod.VARIANTS["main"]
        for name, cfg in es_mod.VARIANTS.items():
            if name != "main":
                self.assertEqual(sum(a != b for a, b in zip(main, cfg)), 1, name)

    def test_bootstrap_resamples_effective_dates(self):
        v = np.array([[1.0], [3.0], [10.0]])
        d = np.array(["a", "a", "b"])
        lo, hi = bootstrap_ci(v, d, B=2000, seed=0)
        self.assertAlmostEqual(lo[0], 2.0)    # 두 번 다 a → (1+3)/2
        self.assertAlmostEqual(hi[0], 10.0)   # 두 번 다 b
        self.assertEqual(bootstrap_ci(v, d, B=50, seed=1)[0][0], bootstrap_ci(v, d, B=50, seed=1)[0][0])
        self.assertTrue(np.isnan(bootstrap_ci(v, np.array(["a"] * 3))[0][0]))  # 편입일이 하나면 구할 수 없다

    def test_leave_one_out(self):
        ls = losses_table()
        out = leave_one_out(event_sens(ls, usage(shifted_table(), EVENTS, ls, 0.5, 5)))
        self.assertEqual(out["event_id"].iloc[0], 2)  # 가장 크게 바꾸는 이벤트
        self.assertAlmostEqual(out["mean_without"].iloc[0], (math.log(2) + math.log(3)) / 2)

    def test_no_baseline_post(self):
        ls = pd.DataFrame([(0, "post", "M", 10, 20.0, 10.0), (1, "post", "M", 2, 8.0, 2.0)],
                          columns=["event_id", "phase", "model", "n", "sum_E", "sum_C"])
        use = pd.DataFrame({"event_id": [0, 1], "phase": "post", "reason": "기준 구간 예측 없음"})
        out = no_baseline_post(ls, use).iloc[0]
        self.assertEqual(out["n_events"], 1)
        self.assertAlmostEqual(out["mean"], math.log(2))


class WalkForwardTest(unittest.TestCase):
    """종목 A(편입), B·C(대조군). 윈도우마다 예측 분산을 다르게 두어 어느 모델을 썼는지 확인한다."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.tickers = ["A", "B", "C"]
        self.ph = make_phases([(0, "A", "2020-11-18", "2020-12-11"), (1, "A", "2019-11-18", "2019-12-13")], self.tickers)
        self.ph.to_parquet(self.tmp / "event_phases.parquet")
        pd.DataFrame([(d, t) for d in CAL for t in self.tickers], columns=["date", "ticker"]).to_parquet(
            self.tmp / "members.parquet")
        pd.DataFrame({"date": CAL}).to_parquet(self.tmp / "features.parquet")
        (self.tmp / "meta.json").write_text(json.dumps({"reportable": False, "hold_reasons": ["t"]}), encoding="utf-8")
        t = pd.concat([pd.DataFrame({"ticker": k, "origin_date": CAL, "horizon": 5, "yz_var": 1.0}) for k in self.tickers])
        t.loc[t["origin_date"] > CAL[-6], "yz_var"] = np.nan
        t.assign(split="train", embargo=True).to_parquet(self.tmp / "targets_h5.parquet")  # 고정 분할 열은 무시
        self.var = {y: math.exp(0.1 * (i + 1)) for i, y in enumerate(TEST_YEARS)}
        for y in TEST_YEARS:
            d = self.tmp / "preds" / f"w{y}"
            d.mkdir(parents=True)
            days = CAL[(CAL.year >= y - 1) & (CAL.year <= y + 1)]
            for m, k in (("M1", 1.0), ("M2", 2.0)):
                p = pd.concat([pd.DataFrame({"ticker": x, "origin_date": days}) for x in self.tickers])
                # A는 2021년에 예측이 더 나빠진다 (해를 넘는 부분)
                bad = np.where((p["ticker"] == "A") & (p["origin_date"].dt.year == 2021), 0.5, 0.0)
                p = p.assign(horizon=5, pred_log_var=k * math.log(self.var[y]) + bad, model=m, seed=0, version="v")
                p.to_parquet(d / f"{m}_h5.parquet", index=False)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_event_uses_one_window_model(self):
        preds = wf.load_window_preds(self.tmp / "preds")
        wf.evaluate(preds, self.tmp, self.tmp / "out", B=50)
        es = pd.read_csv(self.tmp / "out" / "event_sens.csv")
        self.assertEqual(es["event_id"].unique().tolist(), [0])  # 이벤트 1은 테스트 기간 전에 시작
        r = es[(es["model"] == "M1") & (es["phase"] == "post")].iloc[0]
        self.assertEqual(r["window"], 2020)
        shifted = target_phases(self.ph, CAL, 5)
        q = shifted[(shifted["event_id"] == 0) & (shifted["phase"] == "post")].iloc[0]
        days = CAL[(CAL >= q["start"]) & (CAL <= q["end"])]
        # 사후 구간의 2020년·2021년 예측 모두 윈도우 2020 모델의 값이어야 한다
        expected = sum(float(qlike(1.0, self.var[2020] * math.exp(0.5 if d.year == 2021 else 0.0))) for d in days)
        self.assertAlmostEqual(r["sum_E"], expected, places=8)
        self.assertAlmostEqual(r["sum_C"], len(days) * float(qlike(1.0, self.var[2020])), places=8)

    def test_stitched_uses_each_years_model(self):
        wf.evaluate(wf.load_window_preds(self.tmp / "preds"), self.tmp, self.tmp / "out", B=50)
        s = pd.read_csv(self.tmp / "out" / "summary_by_year.csv").query("model == 'M1'").set_index("test_year")
        self.assertEqual(s.index.tolist(), list(TEST_YEARS))
        self.assertAlmostEqual(s.loc[2022, "qlike"], float(qlike(1.0, self.var[2022])), places=8)
        n_days = int((CAL.year == 2020).sum())
        self.assertEqual(s.loc[2020, "n"], 3 * n_days)      # 테스트 연도 끝도 비우지 않는다
        self.assertEqual(s.loc[2024, "n"], 3 * (int((CAL.year == 2024).sum()) - 5))
        v = pd.read_csv(self.tmp / "out" / "sensitivity.csv")
        self.assertEqual(sorted(v["variant"].unique()), sorted([*es_mod.VARIANTS, "stitched"]))
        for f in ("dm_h5", "mcs_h5", "exclusions", "leave_one_out", "no_baseline_post", "summary"):
            self.assertTrue((self.tmp / "out" / f"{f}.csv").exists(), f)

    def test_event_itself_is_not_control(self):
        panel = pd.DataFrame([(t, d, "M", 1.0 if t != "A" else 100.0) for d in CAL[CAL.year == 2020] for t in self.tickers],
                             columns=["ticker", "origin_date", "model", "qlike"])
        ev = select_events(self.ph, CAL)
        ls = event_losses(panel, pd.read_parquet(self.tmp / "members.parquet"), control_windows(self.ph, CAL, 5),
                          target_phases(self.ph, CAL, 5), ev[ev["excluded"] == ""])
        r = ls[ls["phase"] == "baseline"].iloc[0]
        self.assertEqual((r["n"], r["sum_E"], r["sum_C"]), (56, 5600.0, 56.0))

    def test_missing_model_in_one_window(self):
        (self.tmp / "preds" / "w2022" / "M2_h5.parquet").unlink()
        with self.assertRaises(ValueError):
            wf.load_window_preds(self.tmp / "preds")

    def test_requires_final(self):
        with self.assertRaises(SystemExit):
            wf.main(["--preds", str(self.tmp / "preds"), "--data-dir", str(self.tmp)])


if __name__ == "__main__":
    unittest.main()
