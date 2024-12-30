import torch
from torch import Tensor
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from src.metrics.base_metric import BaseMetric


class PESQ(BaseMetric):
    def __init__(self, device="auto", *args, **kwargs):
        super().__init__(*args, **kwargs)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pesq = PerceptualEvaluationSpeechQuality(16000, 'wb')

    def __call__(self, clean_audio_predicted, clean_audio, **kwargs) -> float:
        pesq_result = self.pesq(clean_audio_predicted, clean_audio)
        return pesq_result.item()
