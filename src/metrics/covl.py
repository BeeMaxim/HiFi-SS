import torch
from torch import Tensor
from src.metrics.base_metric import BaseMetric

from src.metrics.metric_denosing import composite_eval


class COVL(BaseMetric):
    def __init__(self, device="auto", *args, **kwargs):
        super().__init__(*args, **kwargs)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

    def __call__(self, reordered, audios, **kwargs) -> float:
        return (composite_eval(reordered[:, :1, :], audios[:, :1, :])["covl"] + composite_eval(reordered[:, 1:, :], audios[:, 1:, :])["covl"]) / 2

