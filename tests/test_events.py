"""편입·편출 공지일 데이터 로딩·검증 테스트."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.events import EventDataError, load_announcements, reconcile_events


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


if __name__ == "__main__":
    unittest.main()
