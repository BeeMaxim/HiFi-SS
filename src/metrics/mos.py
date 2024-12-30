import torch
import torchaudio

import numpy as np

from src.metrics.base_metric import BaseMetric
from src.metrics.wav2vec2mos import Wav2Vec2MOS


class MOSNet(BaseMetric):
    name = "MOSNet"

    def __init__(self, sr=16000, device='cpu', **kwargs):
        super().__init__(**kwargs)

        self.mos_net = Wav2Vec2MOS("weights/wave2vec2mos.pth")
        self.sr = sr
        self.num_splits = 1 # ???
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    def better(self, first, second):
        return first > second

    def _compute_per_split(self, split):
        return self.mos_net.calculate(split)

    def _compute(self, samples, real_samples, epoch_num, epoch_info):
        required_sr = self.mos_net.sample_rate
        resample = torchaudio.transforms.Resample(
            orig_freq=self.sr, new_freq=required_sr
        ).to(self.device)

        samples /= samples.abs().max(-1, keepdim=True)[0]
        samples = [resample(s).squeeze() for s in samples]

        splits = [
            samples[i : i + self.num_splits]
            for i in range(0, len(samples), self.num_splits)
        ]
        fid_per_splits = [self._compute_per_split(split) for split in splits]
        return np.mean(fid_per_splits)
        self.result["mean"] = np.mean(fid_per_splits)
        self.result["std"] = np.std(fid_per_splits)
        print('metrics', self.result["mean"], self.result["std"])


    def __call__(self, clean_audio_predicted, **kwargs):
        """
        Metric calculation logic.

        Args:
            logits (Tensor): model output predictions.
            labels (Tensor): ground-truth labels.
        Returns:
            metric (float): calculated metric.
        """
        return self._compute(clean_audio_predicted, None, None, None)
