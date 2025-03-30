import torch
from torch import Tensor
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
from src.metrics.base_metric import BaseMetric


class STOI(BaseMetric):
    def __init__(self, device="auto", *args, **kwargs):
        super().__init__(*args, **kwargs)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.stoi = ShortTimeObjectiveIntelligibility(16000)

    def __call__(self, clean_audio_predicted, clean_audio, **kwargs) -> float:
        stoi_result = self.stoi(clean_audio_predicted, clean_audio)
        return stoi_result.item()
    

class BSSSTOI(BaseMetric):
    def __init__(self, device="auto", *args, **kwargs):
        super().__init__(*args, **kwargs)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.stoi = ShortTimeObjectiveIntelligibility(16000)

    def __call__(self, reordered, audios, **kwargs) -> float:
        stoi_result = self.stoi(reordered, audios)
        return stoi_result.item()
    