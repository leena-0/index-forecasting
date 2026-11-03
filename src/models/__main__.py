"""모델 실행: 공통 특성 패널 → 예측 테이블 (CLAUDE.md 6.2).

    python -m src.models --model ewma --h 5 [--lam 0.94] [--data-dir data/dev] [--out data/preds]
    python -m src.models --model lgbm --h 5 --params '{"num_leaves": 63}' [--smearing]   # 기본값에 덮어쓰기
    python -m src.models --model har  --h 22
    python -m src.models --model lgbm --h 5 [--seed 0]    # lightgbm 필요 (requirements-models.txt)
    python -m src.models --model lstm --h 5 [--seed 0]    # torch 필요
    python -m src.models --model gru|gnn --h 5            # 2단계 (torch), garch (scipy)
    python -m src.models --model tcn|transformer --h 5    # 3단계 (torch)

결과(out, git 제외):
  {model}_h{h}.parquet     ticker, origin_date, horizon, pred_log_var, model, seed, version
  {model}_h{h}.meta.json   모델 설정·계수, 예측 행 수, prepare meta의 reportable·hold_reasons
모든 분할(train·val·test)의 예측을 만든다. 테스트 구간 평가는 src.evaluate에서 --final로만 한다.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.__main__ import git_commit
from src.pipeline.features import load_panel

from . import registry

REPO = Path(__file__).resolve().parents[2]


def prediction_table(panel, pred_log_var, model, seed=0, version=None):
    """6.2 형식 예측 테이블. 예측이 없는(NaN) 행은 뺀다."""
    t = pd.DataFrame({"ticker": panel["ticker"], "origin_date": panel["origin_date"], "horizon": panel["horizon"],
                      "pred_log_var": pred_log_var, "model": model, "seed": seed, "version": version})
    return t[np.isfinite(t["pred_log_var"])].reset_index(drop=True)


def run(model, h, data_dir, out_dir, params=None, seed=0, name=None):
    """학습·예측 후 예측 테이블과 meta를 저장한다. params는 모델 기본값에 덮어쓴다 (registry.DEFAULTS)."""
    panel = load_panel(data_dir, h)
    pred, info = registry.fit_predict(model, panel, params, seed)
    if name is None:
        name = registry.DISPLAY[model]
        if model == "ewma" and info["lam"] != 0.94:
            name = f"EWMA_{info['lam']:g}"
    for pkg in ("lightgbm", "torch"):
        if pkg in sys.modules:
            info[pkg] = sys.modules[pkg].__version__
    if model == "har":
        print(f"HAR 계수 (학습 {info['n_fit']}행): {info['beta']}")
    elif "train_loss_by_epoch" in info:
        print(f"{name} 에포크별 학습 손실 {info['train_loss_by_epoch']}")
    table = prediction_table(panel, pred, name, seed=seed, version=git_commit())
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{name}_h{h}"
    table.to_parquet(out_dir / f"{stem}.parquet", index=False)
    prep_meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    meta = {"model": name, "horizon": h, "data_dir": str(data_dir), "params": info, "git_commit": git_commit(),
            "rows": len(table), "panel_rows": len(panel),
            "versions": {"python": sys.version.split()[0], "pandas": pd.__version__, "numpy": np.__version__},
            "reportable": bool(prep_meta.get("reportable", False)), "hold_reasons": prep_meta.get("hold_reasons")}
    (out_dir / f"{stem}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{name} h={h}: 예측 {len(table)}행 / 패널 {len(panel)}행 → {out_dir / stem}.parquet")
    return table


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m src.models")
    ap.add_argument("--model", choices=sorted(registry.DISPLAY), required=True)
    ap.add_argument("--h", type=int, choices=[5, 22], required=True)
    ap.add_argument("--lam", type=float, default=0.94, help="EWMA 감쇠 계수 λ")
    ap.add_argument("--params", default=None, help="모델 설정 JSON (기본값에 덮어씀, 예: 튜닝으로 고른 값)")
    ap.add_argument("--smearing", action="store_true", help="학습 구간 잔차로 로그 예측 편향 보정 (registry.py)")
    ap.add_argument("--seed", type=int, default=0, help="난수 seed (LightGBM·LSTM)")
    ap.add_argument("--data-dir", default="data/dev", help="src.pipeline prepare 결과 폴더")
    ap.add_argument("--out", default="data/preds")
    args = ap.parse_args(argv)
    params = json.loads(args.params) if args.params else {}
    if args.model == "ewma":
        params.setdefault("lam", args.lam)
    if args.smearing:
        params["smearing"] = True
    run(args.model, args.h, REPO / args.data_dir, REPO / args.out, params, args.seed)


if __name__ == "__main__":
    main()
