"""편입·편출 공지일 데이터 로딩·검증 테스트."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.events import EventDataError, load_announcements, reconcile_events
from src.events.notices import load_notices, notices_to_announcements


def announcement(**changes):
    row = {
        "ticker": "1234",
        "event_type": "정기변경",
        "direction": "편입",
        "announce_date": "2024-05-24",
        "effective_date": "2024-06-14",
        "source_url": "https://example.com/notice/1",
        "source_type": "manual",
        "collected_at": "2026-10-28T10:00:00+09:00",
    }
    row.update(changes)
    return row


class AnnouncementTest(unittest.TestCase):
    def write(self, rows):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        path = Path(folder.name) / "announcements.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def test_loads_normalizes_and_filters(self):
        path = self.write(
            [
                announcement(),
                announcement(
                    ticker="567890",
                    direction="편출",
                    announce_date="2025-11-28",
                    effective_date="2025-12-12",
                    source_type="crawled",
                ),
            ]
        )
        events = load_announcements(path, as_of="2024-12-31")
        self.assertEqual(len(events), 1)
        self.assertEqual(events.loc[0, "ticker"], "001234")
        self.assertEqual(events.loc[0, "announce_date"], pd.Timestamp("2024-05-24"))

    def test_rejects_missing_column_and_bad_values(self):
        path = self.write([announcement()])
        frame = pd.read_csv(path)
        frame.drop(columns="source_url").to_csv(path, index=False)
        with self.assertRaisesRegex(EventDataError, "필수 열"):
            load_announcements(path)

        with self.assertRaisesRegex(EventDataError, "알 수 없는 event_type"):
            load_announcements(self.write([announcement(event_type="기타")]))
        with self.assertRaisesRegex(EventDataError, "빨라야"):
            load_announcements(self.write([announcement(announce_date="2024-06-14")]))

    def test_rejects_duplicate_event_key(self):
        with self.assertRaisesRegex(EventDataError, "중복"):
            load_announcements(self.write([announcement(), announcement(source_type="crawled")]))

    def test_reconciles_and_reports_both_directions(self):
        expected = pd.DataFrame(
            [
                {
                    "event_id": 10,
                    "ticker": "001234",
                    "event_type": "정기변경",
                    "direction": "편입",
                    "announce_date": pd.Timestamp("2024-06-14"),
                    "effective_date": pd.Timestamp("2024-06-14"),
                    "announce_source": "placeholder",
                },
                {
                    "event_id": 11,
                    "ticker": "999999",
                    "event_type": "수시변경",
                    "direction": "편출",
                    "announce_date": pd.Timestamp("2024-07-01"),
                    "effective_date": pd.Timestamp("2024-07-01"),
                    "announce_source": "placeholder",
                },
            ]
        )
        announcements = load_announcements(
            self.write(
                [
                    announcement(),
                    announcement(
                        ticker="555555",
                        event_type="특례편입",
                        announce_date="2024-08-01",
                        effective_date="2024-08-02",
                    ),
                ]
            )
        )
        result = reconcile_events(expected, announcements)
        self.assertFalse(result.complete)
        self.assertEqual(result.missing["ticker"].tolist(), ["999999"])
        self.assertEqual(result.unexpected["ticker"].tolist(), ["555555"])
        self.assertEqual(result.events.loc[0, "announce_source"], "manual")
        self.assertEqual(result.events.loc[0, "announce_date"], pd.Timestamp("2024-05-24"))
        with self.assertRaisesRegex(EventDataError, "공지 누락 1건"):
            result.require_complete()


NOTICE_HEADER = "변경적용일,변경유형,링크,게시물번호,공지일,제목,비고\n"
NOTICE_ROWS = [
    "2016-06-10,정기변경,u1,1,2016-05-25,정기변경 안내,",
    "2018-02-09,수시변경,u2,2,2018-02-06,이전상장 지수조치,",
    "2018-02-09,수시변경,u3,3,2018-02-07,변경일 확정 안내,2 공지의 변경일 확정",
    ",수시변경,u4,4,2020-03-30,관리종목 지정 수시변경,",
    "2025-09-09,수시변경,u5,5,2025-07-23,회사분할 수시변경,취소 — 실제 미반영",
    ",수시변경(취소),u6,6,2025-07-30,수시변경 취소 안내,",
    "2019-03-12,수시변경,u7,7,2019-03-07,이벤트 없는 공지,",
]


class NoticeTest(unittest.TestCase):
    def write(self, text):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        path = Path(folder.name) / "notices.csv"
        path.write_text(text, encoding="utf-8-sig")
        return path

    def test_applied_flag(self):
        notices = load_notices(self.write(NOTICE_HEADER + "\n".join(NOTICE_ROWS) + "\n")).set_index("notice_id")
        self.assertTrue(notices.loc["1", "applied"])
        self.assertFalse(notices.loc["4", "applied"])  # 변경적용일 없음
        self.assertFalse(notices.loc["5", "applied"])  # 비고: 미반영
        self.assertFalse(notices.loc["6", "applied"])  # 변경유형에 (취소)

    def test_converts_to_announcements(self):
        notices = load_notices(self.write(NOTICE_HEADER + "\n".join(NOTICE_ROWS) + "\n"))
        events = pd.DataFrame(
            [
                ("001234", "정기변경", "편입", "2016-06-10"),
                ("005678", "정기변경", "편출", "2016-06-10"),
                ("068270", "수시변경", "편출", "2018-02-09"),
                ("065620", "수시변경", "편출", "2020-04-22"),
            ],
            columns=["ticker", "event_type", "direction", "effective_date"],
        ).assign(effective_date=lambda f: pd.to_datetime(f["effective_date"]))
        events["announce_date"] = events["effective_date"]  # 구성종목 이벤트의 임시 공지일 열이 있어도 됨
        rows, unmatched, unused = notices_to_announcements(events, notices, "crawled", "2026-11-02T00:00:00+09:00")
        self.assertEqual(rows["ticker"].tolist(), ["001234", "005678", "068270"])
        self.assertEqual(rows.loc[2, "announce_date"], "2018-02-06")  # 같은 적용일이면 가장 이른 공지
        self.assertEqual(unmatched["ticker"].tolist(), ["065620"])
        self.assertEqual(unused["notice_id"].tolist(), ["7"])
        path = self.write("")
        rows.to_csv(path, index=False)
        loaded = load_announcements(path)  # docs/EVENTS.md 형식 검사 통과
        self.assertEqual(set(loaded["source_type"]), {"crawled"})

    def test_rejects_bad_order(self):
        with self.assertRaisesRegex(EventDataError, "늦거나 같은"):
            load_notices(self.write(NOTICE_HEADER + "2016-06-10,정기변경,u,1,2016-06-11,잘못된 날짜,\n"))


if __name__ == "__main__":
    unittest.main()
