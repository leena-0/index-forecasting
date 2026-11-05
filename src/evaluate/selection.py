"""주 예측 기간 선택 (CLAUDE.md 7.4, 테스트 전에 확정).

    imp_h = (1/|M|) Σ_{m∈M} (1 − L_{m,h} / L_{EWMA,h})
    L = 검증 구간 편입 이벤트 윈도우의 평균 QLIKE (공통 표본), M = EWMA를 뺀 1단계 모델 (HAR, LightGBM, LSTM)
    imp_h가 더 큰 h를 주 결과로 쓰고, 다른 h는 보조 결과로 보고한다.
"""
import pandas as pd

STAGE1 = ["HAR", "LightGBM", "LSTM"]


def horizon_choice(event_summary, ref="EWMA", models=STAGE1):
    """event_summary: DataFrame[horizon, model, qlike] (한 분할). 돌려주는 것: DataFrame[horizon, n_models, improvement, chosen]."""
    rows = []
    for h, g in event_summary.groupby("horizon"):
        q = g.set_index("model")["qlike"]
        if ref not in q:
            continue
        ms = [m for m in models if m in q]
        imp = float((1 - q[ms] / q[ref]).mean()) if ms else float("nan")
        rows.append({"horizon": h, "n_models": len(ms), "models": ",".join(ms), "improvement": imp})
    out = pd.DataFrame(rows)
    if len(out):
        out["chosen"] = out["improvement"] == out["improvement"].max()
    return out
