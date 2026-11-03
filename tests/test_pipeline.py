"""src.pipeline 단위 테스트 (합성 데이터, 네트워크 없음).

    .venv/Scripts/python -m unittest
"""
import math
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline import legacy
from src.pipeline.target import forward_yz, yz_k
from src.pipeline.windows import assign_splits, event_phases, split_of

SPLIT = {"train_end": "2022-12-31", "val_end": "2023-12-31"}
PHASES = {
    "baseline": {"anchor": "announce", "range": [-120, -61]},
    "anticipation": {"anchor": "announce", "range": [-60, -1]},
    "short_shock": {"anchor": "effective", "range": [0, 21]},
    "post": {"anchor": "effective", "range": [22, 120]},
}


def make_prices(n_days=60, tickers=("A", "B"), seed=0, start="2022-01-03"):
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range(start, periods=n_days)
    rows = []
    for t in tickers:
        close = 100.0
        for d in cal:
            o = close * math.exp(rng.normal(0, 0.01))
            c = o * math.exp(rng.normal(0, 0.02))
            h = max(o, c) * math.exp(abs(rng.normal(0, 0.01)))
            l = min(o, c) * math.exp(-abs(rng.normal(0, 0.01)))
            rows.append((d, t, o, h, l, c))
            close = c
    return cal, pd.DataFrame(rows, columns=["date", "ticker", "open", "high", "low", "close"])


def yz_reference(O, H, L, C, C0, demean=True):
    """CLAUDE.md 5.1 식을 반복문으로 그대로 계산 (forward_yz 대조용)."""
    n = len(O)
    o = [math.log(O[0] / C0)] + [math.log(O[i] / C[i - 1]) for i in range(1, n)]
    c = [math.log(C[i] / O[i]) for i in range(n)]
    u = [math.log(H[i] / O[i]) for i in range(n)]
    d = [math.log(L[i] / O[i]) for i in range(n)]
    if demean:
        mo, mc = sum(o) / n, sum(c) / n
        vo = sum((x - mo) ** 2 for x in o) / (n - 1)
        vc = sum((x - mc) ** 2 for x in c) / (n - 1)
    else:
        vo = sum(x * x for x in o) / n
        vc = sum(x * x for x in c) / n
    vrs = sum(u[i] * (u[i] - c[i]) + d[i] * (d[i] - c[i]) for i in range(n)) / n
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    return vo + k * vc + (1 - k) * vrs


class YangZhangTest(unittest.TestCase):
    def test_k_values(self):
        self.assertAlmostEqual(yz_k(5), 0.1197, places=4)
        self.assertAlmostEqual(yz_k(22), 0.1396, places=4)

    def test_matches_reference(self):
        cal, p = make_prices()
        for demean in (True, False):
            for h in (5, 22):
                t = forward_yz(p, cal, h, demean).set_index(["ticker", "origin_date"])["yz_var"]
                g = p[p.ticker == "A"].set_index("date")
                for i in (0, 10, len(cal) - h - 1):  # origin = cal[i] → 구간 i+1 ~ i+h
                    w = g.iloc[i + 1:i + 1 + h]
                    ref = yz_reference(w.open.tolist(), w.high.tolist(), w.low.tolist(), w.close.tolist(),
                                       g.close.iloc[i], demean)
                    self.assertAlmostEqual(t[("A", cal[i])], ref, places=12, msg=(demean, h, i))

    def test_tail_is_nan(self):
        cal, p = make_prices()
        t = forward_yz(p, cal, 5)
        a = t[t.ticker == "A"].set_index("origin_date")["yz_var"]
        self.assertTrue(a.iloc[-5:].isna().all())   # 다음 5일이 다 없음
        self.assertTrue(a.iloc[:-5].notna().all())

    def test_halt_makes_windows_nan(self):
        cal, p = make_prices()
        halt_day = cal[30]
        p.loc[(p.ticker == "A") & (p.date == halt_day), ["open", "high", "low", "close"]] = np.nan
        a = forward_yz(p, cal, 5).query("ticker == 'A'").set_index("origin_date")["yz_var"]
        # 거래정지일(30)이나 그다음 날(31, 직전 종가 없음)이 구간에 들어가는 origin 25~30은 NaN
        self.assertTrue(a.loc[cal[25]:cal[30]].isna().all())
        self.assertTrue(np.isfinite(a.loc[cal[24]]))  # 구간 25~29
        self.assertTrue(np.isfinite(a.loc[cal[31]]))  # 구간 32~36, o_1은 31일 종가 사용

    def test_missing_calendar_day_is_nan(self):
        cal, p = make_prices()
        p = p[~((p.ticker == "A") & (p.date == cal[30]))]  # 행 자체가 없음
        a = forward_yz(p, cal, 5).query("ticker == 'A'").set_index("origin_date")["yz_var"]
        self.assertTrue(a.loc[cal[25]:cal[30]].isna().all())

    def test_no_lookahead_leak(self):
        """origin t의 타깃은 t 종가 ~ t+h 데이터만 쓴다. 그 밖을 바꿔도 값이 같아야 한다 (CLAUDE.md 10.6)."""
        cal, p = make_prices()
        h, i = 5, 20
        base = forward_yz(p, cal, h).set_index(["ticker", "origin_date"])["yz_var"]
        q = p.copy()
        after = (q.ticker == "A") & (q.date > cal[i + h])
        before = (q.ticker == "A") & (q.date < cal[i])
        q.loc[after | before, ["open", "high", "low", "close"]] *= 3.0
        q.loc[(q.ticker == "A") & (q.date == cal[i]), ["open", "high", "low"]] *= 3.0  # t의 종가 외
        changed = forward_yz(q, cal, h).set_index(["ticker", "origin_date"])["yz_var"]
        self.assertAlmostEqual(changed[("A", cal[i])], base[("A", cal[i])], places=14)
        # 구간 안의 값을 바꾸면 달라진다
        q2 = p.copy()
        q2.loc[(q2.ticker == "A") & (q2.date == cal[i + h]), "high"] *= 1.05
        self.assertNotAlmostEqual(forward_yz(q2, cal, h).set_index(["ticker", "origin_date"])
                                  ["yz_var"][("A", cal[i])], base[("A", cal[i])], places=14)


class SplitTest(unittest.TestCase):
    def test_split_of(self):
        s = split_of(pd.to_datetime(["2022-12-30", "2023-01-02", "2023-12-29", "2024-01-02"]), SPLIT)
        self.assertEqual(list(s), ["train", "val", "val", "test"])

    def test_embargo_is_h_days_before_each_boundary(self):
        cal = pd.bdate_range("2022-11-01", "2024-02-29")
        for h in (5, 22):
            t = pd.DataFrame({"ticker": "A", "origin_date": cal, "horizon": h, "yz_var": 1.0})
            t = assign_splits(t, cal, SPLIT)
            for b in ("2023-01-01", "2024-01-01"):
                pos = cal.searchsorted(pd.Timestamp(b))  # 경계 뒤 첫 거래일
                emb = t.iloc[pos - h - 3:pos + 3]["embargo"].tolist()
                self.assertEqual(emb, [False] * 3 + [True] * h + [False] * 3, (h, b))
            self.assertTrue(t["embargo"].iloc[-h:].all())  # 끝 h일은 타깃 구간이 달력 밖

    def test_origin_outside_calendar_raises(self):
        cal = pd.bdate_range("2023-01-02", periods=10)
        t = pd.DataFrame({"ticker": "A", "origin_date": [pd.Timestamp("2023-01-01")], "horizon": 5})
        with self.assertRaises(ValueError):
            assign_splits(t, cal, SPLIT)


class EventPhaseTest(unittest.TestCase):
    def setUp(self):
        self.cal = pd.bdate_range("2021-01-04", "2024-12-31")
        self.prices = pd.DataFrame({"date": list(self.cal) * 2, "ticker": ["A"] * len(self.cal) + ["B"] * len(self.cal)})

    def ev(self, rows):
        e = pd.DataFrame(rows, columns=["ticker", "announce_date", "effective_date", "announce_source"])
        e["announce_date"] = pd.to_datetime(e["announce_date"])
        e["effective_date"] = pd.to_datetime(e["effective_date"])
        e.insert(0, "event_id", range(len(e)))
        e["event_type"], e["direction"] = "정기변경", "편입"
        return e

    def phases(self, rows, prices=None):
        return event_phases(self.ev(rows), self.cal, self.prices if prices is None else prices, PHASES,
                            SPLIT).set_index(["event_id", "phase"])

    def test_offsets(self):
        A, E = self.cal[300], self.cal[310]
        ph = self.phases([("A", A, E, "known")])
        exp = {"baseline": (180, 239), "anticipation": (240, 299), "announce_to_effective": (300, 309),
               "short_shock": (310, 331), "post": (332, 430)}
        for name, (s, e) in exp.items():
            self.assertEqual((ph.loc[(0, name), "start"], ph.loc[(0, name), "end"]), (self.cal[s], self.cal[e]))
            self.assertTrue(ph.loc[(0, name), "complete"], name)

    def test_placeholder_gives_empty_announce_phase(self):
        E = self.cal[300]
        ph = self.phases([("A", E, E, "placeholder")])
        self.assertFalse(ph.loc[(0, "announce_to_effective"), "complete"])
        self.assertIn("임시 공지일", ph.loc[(0, "announce_to_effective"), "reason"])
        self.assertTrue(ph.loc[(0, "baseline"), "complete"])

    def test_incomplete_edges(self):
        ph = self.phases([("A", self.cal[50], self.cal[50], "placeholder"),      # 기준 구간이 달력 시작 전
                          ("A", self.cal[-30], self.cal[-30], "placeholder")])  # 사후 구간이 as_of 이후
        self.assertEqual(ph.loc[(0, "baseline"), "reason"], "달력 시작 전")
        self.assertTrue(ph.loc[(0, "short_shock"), "complete"])
        self.assertEqual(ph.loc[(1, "post"), "reason"], "as_of 이후")
        self.assertTrue(ph.loc[(1, "short_shock"), "complete"])  # 편입일 ~ +21은 남은 30일 안

    def test_short_history_ticker(self):
        prices = self.prices[~((self.prices.ticker == "B") & (self.prices.date < self.cal[280]))]
        ph = self.phases([("B", self.cal[300], self.cal[300], "placeholder")], prices)
        self.assertEqual(ph.loc[(0, "baseline"), "reason"], "종목 데이터 기간 밖")
        self.assertTrue(ph.loc[(0, "short_shock"), "complete"])

    def test_split_and_crossing(self):
        day = lambda s: self.cal[self.cal.searchsorted(pd.Timestamp(s))]
        ph = self.phases([("A", day("2023-03-01"), day("2023-03-01"), "placeholder"),
                          ("A", day("2023-07-03"), day("2023-07-03"), "placeholder")])
        self.assertEqual(ph.loc[(0, "baseline"), "split"], "val")
        self.assertTrue(ph.loc[(0, "baseline"), "crosses_split"])    # 기준 구간이 2022년
        self.assertFalse(ph.loc[(1, "baseline"), "crosses_split"])   # ±120영업일이 2023년 안

    def test_baseline_overlap_flag(self):
        ph = self.phases([("A", self.cal[300], self.cal[300], "placeholder"),
                          ("A", self.cal[400], self.cal[400], "placeholder"),   # 기준 280~339 ⟂ 직전 사후 322~420
                          ("B", self.cal[400], self.cal[400], "placeholder")])
        self.assertFalse(ph.loc[(0, "baseline"), "baseline_overlaps_prev"])
        self.assertTrue(ph.loc[(1, "baseline"), "baseline_overlaps_prev"])
        self.assertFalse(ph.loc[(2, "baseline"), "baseline_overlaps_prev"])


def make_legacy(d):
    """기존 데이터 폴더(kosdaq150_forecast/data) 구조를 흉내 낸 작은 폴더 (종목 000001, 000002, 2015-07-13~16)."""
    (d / "raw/aie_history/stocks").mkdir(parents=True)
    (d / "raw/stocks").mkdir(parents=True)
    (d / "processed").mkdir()
    days = ["2015-07-13", "2015-07-14", "2015-07-15", "2015-07-16"]
    pd.DataFrame({"date": days, "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1, "value": 1,
                  "mktcap": 1}).to_csv(d / "raw/index_2203.csv", index=False)
    pd.DataFrame({"date": days, "open": [10, 0, 11, 12], "high": [11, 0, 12, 13], "low": [9, 0, 10, 11],
                  "close": [10, 10, 11, 12], "volume": [5, 0, 6, 7], "chg_pct": 0}).to_csv(
        d / "raw/aie_history/stocks/000001.csv", index=False)
    pd.DataFrame({"date": days[2:], "open": 20, "high": 21, "low": 19, "close": 20, "volume": [1, 0],
                  "chg_pct": 0}).to_csv(d / "raw/stocks/000002.csv", index=False)  # 07-16: 거래량 0
    pd.DataFrame({"date": [d.replace("-", "") for d in days] * 2,
                  "ticker": ["000001"] * 4 + ["000002"] * 4, "mcap": 100, "shares": 10}).to_csv(
        d / "raw/aie_history/mcap_daily.csv", index=False)
    pd.DataFrame({"date": ["20150713", "20150714", "20150715", "20150715", "20150716"],
                  "ticker": ["000001", "000001", "000001", "000002", "000002"]}).to_csv(
        d / "raw/aie_history/snapshots_2203.csv", index=False)
    pd.DataFrame({"ticker": ["000001", "000002", "000001"],
                  "event_type": ["initial", "regular_add", "regular_del"],
                  "effective_date": ["2015-07-13", "2015-07-15", "2015-07-16"],
                  "listing_date": "", "announce_date": ""}).to_csv(d / "processed/events.csv", index=False)



class LegacyTest(unittest.TestCase):
    """기존 데이터 폴더 구조를 흉내 낸 작은 폴더로 어댑터를 확인한다."""

    def setUp(self):
        self.d = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.d)
        make_legacy(self.d)

    def test_load(self):
        data = legacy.load(self.d, "2015-07-15")
        p = data["prices"]
        self.assertEqual(len(data["calendar"]), 3)          # as_of로 잘림
        self.assertEqual(p["date"].max(), pd.Timestamp("2015-07-15"))
        halt = p[(p.ticker == "000001") & (p.date == "2015-07-14")]
        self.assertTrue(halt[["open", "high", "low", "close", "volume"]].isna().all(axis=None))
        self.assertEqual(p["mcap"].notna().sum(), len(p))
        ev = data["events"]
        self.assertEqual(ev[["ticker", "event_type", "direction"]].values.tolist(),
                         [["000002", "정기변경", "편입"]])   # initial 제외, as_of 이후 편출 제외
        self.assertEqual(ev["announce_source"].tolist(), ["placeholder"])
        self.assertEqual(ev["announce_date"].tolist(), ev["effective_date"].tolist())

    def test_zero_volume_is_halt(self):
        p = legacy.load(self.d, "2015-07-16")["prices"]
        row = p[(p.ticker == "000002") & (p.date == "2015-07-16")]
        self.assertTrue(row[["open", "high", "low", "close", "volume"]].isna().all(axis=None))
        self.assertTrue(p[(p.ticker == "000002") & (p.date == "2015-07-15")]["close"].notna().all())

    def test_check_events(self):
        data = legacy.load(self.d, "2015-07-16")
        self.assertEqual(legacy.check_events(data["events"], data["members"], data["calendar"]), [])
        bad = data["events"].assign(effective_date=pd.Timestamp("2015-07-14"))
        self.assertEqual(len(legacy.check_events(bad, data["members"], data["calendar"])), 2)


class MissingPolicyTest(unittest.TestCase):
    def reconciled(self, missing=0, unexpected=0):
        from src.events.loader import Reconciliation
        keys = pd.DataFrame({"ticker": ["000001"] * 2, "event_type": "정기변경", "direction": "편입",
                             "effective_date": pd.to_datetime(["2023-06-09", "2023-12-15"])})
        return Reconciliation(events=keys.iloc[:1], missing=keys.iloc[:missing], unexpected=keys.iloc[:unexpected])

    def test_drop_and_error(self):
        from src.events import EventDataError
        from src.pipeline.__main__ import apply_missing_policy
        events, dropped = apply_missing_policy(self.reconciled(missing=1), "drop")
        self.assertEqual((len(events), len(dropped)), (1, 1))
        with self.assertRaises(EventDataError):
            apply_missing_policy(self.reconciled(missing=1), "error")
        with self.assertRaises(EventDataError):  # 구성종목 변화가 없는 공지는 drop이어도 중단
            apply_missing_policy(self.reconciled(unexpected=1), "drop")
        with self.assertRaises(ValueError):
            apply_missing_policy(self.reconciled(), "skip")


if __name__ == "__main__":
    unittest.main()
