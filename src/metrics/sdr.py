import torch
from torch import Tensor
from torchmetrics.audio import SignalDistortionRatio
from src.metrics.base_metric import BaseMetric


class SDR(BaseMetric):
    def __init__(self, device="auto", *args, **kwargs):
        super().__init__(*args, **kwargs)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sdr = SignalDistortionRatio().to(device)

    def __call__(self, clean_audio_predicted, clean_audio, noisy_audio, **kwargs) -> float:
        return self.sdr(clean_audio_predicted, clean_audio)
    

class BSSSDR(BaseMetric):
    def __init__(self, device="auto", *args, **kwargs):
        super().__init__(*args, **kwargs)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sdr = SignalDistortionRatio().to(device)

    def __call__(self, reordered, audios, **kwargs) -> float:
        return self.sdr(reordered, audios)