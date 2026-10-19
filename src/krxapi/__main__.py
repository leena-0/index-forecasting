"""KRX Open API 수집.

    python -m src.krxapi collect [--limit N]   # 수집 (끊기거나 하루 한도에 닿아도 다시 실행하면 이어 받음)
    python -m src.krxapi manifest              # progress.jsonl로 data/MANIFEST.csv 다시 쓰기
    python -m src.krxapi verify                # 로컬 파일 sha256을 MANIFEST.csv와 대조

--limit N: 처음 N 평일만 (시험용). 인증키는 .env의 KRX_OPENAPI_KEY, 수집자 구분은 COLLECTOR_ID.
"""
import argparse
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

from .client import ApiError, DailyLimit, OpenApiClient
from .collect import Collector
from .store import Store

REPO = Path(__file__).resolve().parents[2]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m src.krxapi")
    ap.add_argument("command", choices=["collect", "manifest", "verify"])
    ap.add_argument("--config", default=str(REPO / "configs" / "krxapi.yaml"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    load_dotenv(REPO / ".env", override=False)
    store = Store(REPO / cfg["data_dir"], os.environ.get("COLLECTOR_ID", "").strip())

    if args.command == "verify":
        if not store.manifest_path.exists():
            sys.exit(f"{store.manifest_path}이 없습니다.")
        missing, mismatched = store.verify()
        print(f"없음 {len(missing)}, sha256 불일치 {len(mismatched)}")
        for p in missing + mismatched:
            print("  " + p)
        return 1 if missing or mismatched else 0
    if not store.collector_id:
        sys.exit(".env에 COLLECTOR_ID를 적어 주세요.")
    if args.command == "manifest":
        print(f"MANIFEST {store.write_manifest(cfg['as_of'])}행")
        return 0

    key = os.environ.get("KRX_OPENAPI_KEY", "").strip()
    if not key:
        sys.exit(".env에 KRX_OPENAPI_KEY를 적어 주세요.")
    client = OpenApiClient(key, sleep_sec=cfg["sleep_sec"], max_calls=cfg["max_calls"])
    try:
        Collector(client, store, cfg).run(limit=args.limit)
    except (DailyLimit, ApiError) as e:
        sys.exit(str(e))
    return 0


if __name__ == "__main__":
    sys.exit(main())
