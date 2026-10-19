"""KRX Open API 수집.

    python -m src.krxapi collect [--limit N | --date YYYY-MM-DD]   # 수집 (끊기거나 하루 한도에 닿아도 다시 실행하면 이어 받음)
    python -m src.krxapi validate [--complete]                      # 저장된 파일 검사 (네트워크 없음)
    python -m src.krxapi manifest                                   # progress.jsonl로 data/MANIFEST.csv 다시 쓰기
    python -m src.krxapi verify                                     # 로컬 파일 sha256을 MANIFEST.csv와 대조

--limit N: 처음 N 평일만, --date: 그 하루만 (시험용). 인증키는 .env의 KRX_OPENAPI_KEY, 수집자 구분은 COLLECTOR_ID.
종료 코드(collect): 0 정상, 1 실패한 단위 있음, 2 하루 한도·요청 제한으로 멈춤.
"""
import argparse
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

from .client import ApiError, OpenApiClient, RequestBudget
from .collect import Collector, validate_files
from .store import Store

REPO = Path(__file__).resolve().parents[2]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m src.krxapi")
    ap.add_argument("command", choices=["collect", "validate", "manifest", "verify"])
    ap.add_argument("--config", default=str(REPO / "configs" / "krxapi.yaml"))
    ap.add_argument("--limit", type=int, default=None, help="처음 N 평일만")
    ap.add_argument("--date", default=None, help="한 날짜만 (YYYY-MM-DD)")
    ap.add_argument("--complete", action="store_true", help="validate: 모든 평일 파일이 있는지도 검사")
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
    if args.command == "validate":
        issues = validate_files(store, cfg, complete=args.complete)
        for issue in issues:
            print("  " + issue)
        print(f"검사 문제 {len(issues)}건" if issues else "검사 통과")
        return 1 if issues else 0
    if not store.collector_id:
        sys.exit(".env에 COLLECTOR_ID를 적어 주세요.")
    if args.command == "manifest":
        print(f"MANIFEST {store.write_manifest(cfg['as_of'])}행")
        return 0

    key = os.environ.get("KRX_OPENAPI_KEY", "").strip()
    if not key:
        sys.exit(".env에 KRX_OPENAPI_KEY를 적어 주세요.")
    budget = RequestBudget(store.path("_state"), limit=cfg["daily_limit"])
    client = OpenApiClient(key, budget, sleep_sec=cfg["sleep_sec"], retries=cfg["retries"])
    try:
        return Collector(client, store, cfg).run(limit=args.limit, only_date=args.date)
    except ApiError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    sys.exit(main())
