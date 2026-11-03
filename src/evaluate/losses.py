"""예측 테이블 → 손실 패널 (CLAUDE.md 6.2, 7.1).

예측 테이블 (CLAUDE.md 6.2): ticker, origin_date, horizon, pred_log_var, model, seed, version
타깃 (src.pipeline prepare의 targets_h{h}.parquet): ticker, origin_date, horizon, yz_var, split, embargo

손실은 분산 단위로 계산한다 (σ̂² = exp(pred_log_var), σ² = 실현 Yang–Zhang 분산):
    MSE   = (σ² − σ̂²)²
    QLIKE = σ²/σ̂² − ln(σ²/σ̂²) − 1          (주 지표. σ̂² = σ²일 때 0, 항상 ≥ 0)
Patton (2011): 둘 다 잡음 섞인 대리변수에서도 모델 순위가 유지된다.
"""
import numpy as np
import pandas as pd

PRED_COLS = ["ticker", "origin_date", "horizon", "pred_log_var", "model", "seed", "version"]
KEY = ["ticker", "origin_date", "horizon"]


def qlike(y, yhat):
    """QLIKE = y/ŷ − ln(y/ŷ) − 1. y, ŷ는 분산(> 0)."""
    r = np.asarray(y, dtype=float) / np.asarray(yhat, dtype=float)
    return r - np.log(r) - 1


def mse(y, yhat):
    """MSE = (y − ŷ)². y, ŷ는 분산."""
    return (np.asarray(y, dtype=float) - np.asarray(yhat, dtype=float)) ** 2


def check_predictions(pred):
    """예측 테이블이 6.2 형식인지 확인한다. 문제가 있으면 ValueError."""
    missing = [c for c in PRED_COLS if c not in pred]
    if missing:
        raise ValueError(f"예측 테이블에 열이 없습니다: {missing}")
    if not pred["horizon"].isin([5, 22]).all():
        raise ValueError(f"horizon은 5 또는 22여야 합니다: {sorted(pred['horizon'].unique())}")
    if not np.isfinite(pred["pred_log_var"].to_numpy(dtype=float)).all():
        raise ValueError("pred_log_var에 결측·무한값이 있습니다.")
    dup = pred.duplicated(KEY + ["model", "seed"])
    if dup.any():
        raise ValueError(f"(ticker, origin_date, horizon, model, seed)가 중복된 행 {int(dup.sum())}개")


def loss_panel(pred, targets, common=True):
    """예측과 타깃을 맞춰 행마다 손실을 계산한다.

    - 타깃은 embargo=False이고 yz_var > 0인 행만 쓴다 (0 이하면 QLIKE를 계산할 수 없음).
    - seed가 여러 개인 모델은 seed별 손실의 평균을 쓴다 (n_seeds 열).
    - common=True면 모든 모델이 예측한 (ticker, origin_date, horizon)만 남긴다 — 모델마다 평가 표본이
      달라지면 비교가 깨지기 때문 (CLAUDE.md 6.3).
    돌려주는 것: (panel[ticker, origin_date, horizon, split, model, qlike, mse, n_seeds], counts)
    """
    check_predictions(pred)
    t = targets[~targets["embargo"] & (targets["yz_var"] > 0)][KEY + ["split", "yz_var"]]
    p = pred.merge(t, on=KEY, how="left", validate="many_to_one")
    counts = {"pred_rows": len(pred), "no_usable_target": int(p["yz_var"].isna().sum())}
    p = p[p["yz_var"].notna()]
    vhat = np.exp(p["pred_log_var"].to_numpy(dtype=float))
    p = p.assign(qlike=qlike(p["yz_var"], vhat), mse=mse(p["yz_var"], vhat))
    panel = (p.groupby(KEY + ["split", "model"], as_index=False, observed=True)
              .agg(qlike=("qlike", "mean"), mse=("mse", "mean"), n_seeds=("seed", "nunique")))
    if common:
        n_models = panel["model"].nunique()
        covered = panel.groupby(KEY)["model"].transform("nunique") == n_models
        counts["dropped_not_common"] = int((~covered).sum())
        panel = panel[covered]
    counts["rows"] = len(panel)
    return panel.reset_index(drop=True), counts


def summary(panel):
    """분할·예측 기간·모델별 평균 손실. 열: split, horizon, model, n, qlike, mse."""
    return (panel.groupby(["split", "horizon", "model"], as_index=False)
                 .agg(n=("qlike", "size"), qlike=("qlike", "mean"), mse=("mse", "mean")))
