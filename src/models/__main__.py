"""모델 실행: 공통 특성 패널 → 예측 테이블 (CLAUDE.md 6.2).

    python -m src.models --model ewma --h 5 [--lam 0.94] [--data-dir data/dev] [--out data/preds]
    python -m src.models --model har  --h 22
    python -m src.models --model lgbm --h 5 [--seed 0]    # lightgbm 필요 (requirements-models.txt)
    python -m src.models --model lstm --h 5 [--seed 0]    # torch 필요

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

from . import ewma, har, lgbm, lstm

REPO = Path(__file__).resolve().parents[2]


def prediction_table(panel, pred_log_var, model, seed=0, version=None):
    """6.2 형식 예측 테이블. 예측이 없는(NaN) 행은 뺀다."""
    t = pd.DataFrame({"ticker": panel["ticker"], "origin_date": panel["origin_date"], "horizon": panel["horizon"],
                      "pred_log_var": pred_log_var, "model": model, "seed": seed, "version": version})
    return t[np.isfinite(t["pred_log_var"])].reset_index(drop=True)


def run(model, h, data_dir, out_dir, lam=0.94, seed=0):
    panel = load_panel(data_dir, h)
    info = {}
    if model == "ewma":
        name = "EWMA" if lam == 0.94 else f"EWMA_{lam:g}"
        var = ewma.predict(panel, lam)
        pred = np.log(var.where(var > 0))
        info = {"lam": lam, "warmup": 22}
    elif model == "har":
        name = "HAR"
        beta, n_fit = har.fit(panel)
        pred = har.predict(panel, beta)
        info = {"beta": dict(zip(["const"] + har.REGRESSORS, beta.tolist())), "n_fit": n_fit}
        print(f"HAR 계수 (학습 {n_fit}행): {info['beta']}")
    elif model == "lgbm":
        import lightgbm

        name = "LightGBM"
        fitted, n_fit = lgbm.fit(panel, seed)
        pred = lgbm.predict(panel, fitted)
        info = {**lgbm.PARAMS, "n_fit": n_fit, "lightgbm": lightgbm.__version__}
        print(f"LightGBM 학습 {n_fit}행")
    elif model == "lstm":
        import torch

        name = "LSTM"
        pred, info = lstm.fit_predict(panel, seed)
        info["torch"] = torch.__version__
        print(f"LSTM 학습 {info['n_train']}행, 에포크별 학습 손실 {info['train_loss_by_epoch']}")
    else:
        raise ValueError(f"지원하지 않는 모델: {model}")
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
    ap.add_argument("--model", choices=["ewma", "har", "lgbm", "lstm"], required=True)
    ap.add_argument("--h", type=int, choices=[5, 22], required=True)
    ap.add_argument("--lam", type=float, default=0.94, help="EWMA 감쇠 계수 λ")
    ap.add_argument("--seed", type=int, default=0, help="난수 seed (LightGBM·LSTM)")
    ap.add_argument("--data-dir", default="data/dev", help="src.pipeline prepare 결과 폴더")
    ap.add_argument("--out", default="data/preds")
    args = ap.parse_args(argv)
    run(args.model, args.h, REPO / args.data_dir, REPO / args.out, args.lam, args.seed)


if __name__ == "__main__":
    main()
