"""3단계 모델(TCN·Transformer) 단위 테스트 (합성 데이터, 네트워크 없음).

    .venv/Scripts/python -m unittest
"""
import importlib.util
import unittest

import numpy as np

from src.models import lstm, registry
from tests.test_models import ml_panel

HAS_TORCH = importlib.util.find_spec("torch") is not None
TINY = {"lookback": 5, "hidden": 4, "epochs": 1, "batch_size": 16}


@unittest.skipUnless(HAS_TORCH, "torch 없음 (requirements-models.txt)")
class Stage3Test(unittest.TestCase):
    def test_fit_predict(self):
        p = ml_panel()
        _, rows = lstm.sequences(p.sort_values(["ticker", "origin_date"]), 5)
        for model in ("tcn", "transformer"):
            pred, info = registry.fit_predict(model, p, TINY, seed=0)
            pred2, _ = registry.fit_predict(model, p, TINY, seed=0)
            self.assertEqual(info["arch"], model)
            self.assertEqual(pred.notna().sum(), len(rows))  # 결측 없는 창마다 예측
            np.testing.assert_allclose(pred.dropna(), pred2.dropna())  # 같은 seed → 같은 결과

    def test_tcn_is_causal(self):
        import torch

        net = lstm._net(3, {**lstm.PARAMS, **TINY, "arch": "tcn", "layers": 3})
        x = torch.randn(2, 5, 3)
        y1 = net.blocks(x.transpose(1, 2))
        x2 = x.clone()
        x2[:, 3:] += 10.0  # 4번째 시점 이후를 바꿔도
        y2 = net.blocks(x2.transpose(1, 2))
        torch.testing.assert_close(y1[:, :, :3], y2[:, :, :3])  # 그 이전 시점의 출력은 같다


if __name__ == "__main__":
    unittest.main()
