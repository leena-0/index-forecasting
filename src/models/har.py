"""HAR (Corsi, 2009) — 로그 형태, 전 종목 통합(pooled) OLS.

    log_var_t = β0 + β1·ln rv_d_t + β2·ln rv_w_t + β3·ln rv_m_t + ε_t
    log_var_t = ln(구간 t+1 ~ t+h의 Yang–Zhang 분산), rv_d·rv_w·rv_m = 1·5·22일 구성요소 (features.py)

- 추정: split == train, embargo == False이고 타깃·설명변수가 모두 유한한 행. 계수는 한 번만 추정하고
  검증·테스트 구간에도 그대로 쓴다 (검증·테스트 정보 미사용).
- rv ≤ 0(가격이 하루 종일 같은 날 등)인 행은 로그를 쓸 수 없어 추정·예측에서 빠진다.
- 로그에서 예측하므로 exp(ŷ)는 조건부 평균이 아니라 중앙값에 가깝다. 편향 보정은 하지 않는다.
"""
import numpy as np
import pandas as pd

REGRESSORS = ["rv_d", "rv_w", "rv_m"]


def design(panel):
    """설계행렬 [1, ln rv_d, ln rv_w, ln rv_m]. rv ≤ 0 또는 결측이면 그 행은 NaN."""
    x = panel[REGRESSORS].where(panel[REGRESSORS] > 0)
    return np.column_stack([np.ones(len(panel)), np.log(x.to_numpy(dtype=float))])


def fit(panel):
    """학습 구간 행으로 β = (β0, β1, β2, β3) 추정. 돌려주는 것: (beta, 사용한 행 수)."""
    tr = panel[(panel["split"] == "train") & ~panel["embargo"]]
    X, y = design(tr), tr["log_var"].to_numpy(dtype=float)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    beta, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
    return beta, int(ok.sum())


def predict(panel, beta):
    """예측 로그 분산 Series (panel과 같은 index). 설명변수가 없으면 NaN."""
    return pd.Series(design(panel) @ beta, index=panel.index)
