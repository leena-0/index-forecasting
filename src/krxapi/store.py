"""수집 단위 파일 저장·진행 기록·MANIFEST (CLAUDE.md 9.3).

- 단위 하나 = parquet 파일 하나. 임시 파일에 쓴 뒤 os.replace로 이름을 바꾼다(atomic rename).
  중간에 끊기면 완료 파일이 생기지 않으므로, 완료 파일이 있으면 그 단위는 건너뛴다.
- data/_state/progress.jsonl: 단위마다 status, rows, sha256, collected_at, collector_id (로컬 전용)
- data/MANIFEST.csv: 완료 파일의 path, rows, sha256, collected_at, as_of, collector_id (git 추적)
- snapshot.json: 기준일(as_of)이 다른 스냅샷과 섞이지 않게 막는다.
(topic/krx-crawler의 src/crawl/store.py와 같은 코드)
"""
import csv
import datetime as dt
import hashlib
import json
import os
import threading
from pathlib import Path

import pandas as pd

MANIFEST_COLS = ["path", "rows", "sha256", "collected_at", "as_of", "collector_id"]


def now():
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class SnapshotMismatch(RuntimeError):
    pass


class Store:
    def __init__(self, data_dir, collector_id):
        self.data_dir = Path(data_dir)
        self.collector_id = collector_id
        self.progress_path = self.data_dir / "_state" / "progress.jsonl"
        self.manifest_path = self.data_dir / "MANIFEST.csv"
        self._log_lock = threading.Lock()  # 작업자 여러 개가 progress.jsonl에 동시에 쓰지 않게

    def path(self, rel):
        return self.data_dir / rel

    def exists(self, rel):
        return self.path(rel).exists()

    def read(self, rel):
        return pd.read_parquet(self.path(rel))

    def write(self, rel, rows):
        """rows(list[dict], KRX 원본 문자열 그대로)를 rel에 저장하고 진행 기록을 남긴다."""
        final = self.path(rel)
        final.parent.mkdir(parents=True, exist_ok=True)
        tmp = final.with_name(final.name + ".tmp")
        pd.DataFrame(rows).to_parquet(tmp, index=False)
        os.replace(tmp, final)
        self._log({"path": rel, "status": "done", "rows": len(rows), "sha256": sha256(final)})

    def log_failure(self, rel, err):
        tmp = self.path(rel).with_name(self.path(rel).name + ".tmp")
        if tmp.exists():
            tmp.unlink()
        self._log({"path": rel, "status": "failed", "error": f"{type(err).__name__}: {err}"})

    def _log(self, rec):
        rec = {**rec, "collected_at": now(), "collector_id": self.collector_id}
        self.progress_path.parent.mkdir(parents=True, exist_ok=True)
        with self._log_lock, open(self.progress_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ── 스냅샷 ──
    def check_snapshot(self, rel, meta):
        """처음이면 snapshot.json을 만들고, 있으면 as_of·start·index가 같은지 확인한다.
        이전 스냅샷 정보를 돌려준다(처음이면 새로 만든 것)."""
        p = self.path(rel)
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            snap = {**meta, "created_at": now(), "collector_id": self.collector_id}
            p.write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")
            return snap
        snap = json.loads(p.read_text(encoding="utf-8"))
        diff = {k: (snap.get(k), v) for k, v in meta.items() if snap.get(k) != v}
        if diff:
            raise SnapshotMismatch(
                f"{p}의 스냅샷 설정과 config가 다릅니다 (기존, config): {diff}. "
                "다른 기준일의 데이터를 섞지 않으려면 새 data 폴더에서 받으세요.")
        return snap

    # ── MANIFEST ──
    def write_manifest(self, as_of):
        """progress.jsonl에서 파일별 마지막 done 기록을 모아 MANIFEST.csv를 다시 쓴다."""
        done = {}
        if self.progress_path.exists():
            for line in self.progress_path.read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                if rec["status"] == "done":
                    done[rec["path"]] = rec
        rows = [{**{k: rec[k] for k in ("path", "rows", "sha256", "collected_at", "collector_id")},
                 "as_of": as_of}
                for path, rec in sorted(done.items()) if self.exists(path)]
        with open(self.manifest_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=MANIFEST_COLS)
            w.writeheader()
            w.writerows(rows)
        return len(rows)

    def verify(self):
        """MANIFEST.csv의 파일이 로컬에 있고 sha256이 같은지 확인한다.
        (missing, mismatched) 경로 목록을 돌려준다."""
        missing, mismatched = [], []
        with open(self.manifest_path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                p = self.path(row["path"])
                if not p.exists():
                    missing.append(row["path"])
                elif sha256(p) != row["sha256"]:
                    mismatched.append(row["path"])
        return missing, mismatched
