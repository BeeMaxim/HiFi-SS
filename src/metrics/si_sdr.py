import torch
from torch import Tensor
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
from src.metrics.base_metric import BaseMetric
import numpy as np

class SiSDR(BaseMetric):
    def __init__(self, device="auto", *args, **kwargs):
        super().__init__(*args, **kwargs)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.si_sdr = ScaleInvariantSignalDistortionRatio().to(device)

    def __call__(self, clean_audio_predicted, clean_audio, noisy_audio, **kwargs) -> float:
        return self.si_sdr(clean_audio_predicted, clean_audio)
