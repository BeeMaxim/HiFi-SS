from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm
import src.utils.nn_utils as nn_utils
from hydra.utils import instantiate

# from datasets import mel_spectrogram
# from .models_registry import generators
import src.utils.hifi_utils as utils

import numpy as np
import librosa
from librosa.filters import mel as librosa_mel_fn
from librosa.util import normalize


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def mel_spectrogram(
    y,
    n_fft,
    num_mels,
    sampling_rate,
    hop_size,
    win_size,
    fmin,
    fmax,
    center=False,
    return_mel_and_spec=False,
):
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y).unsqueeze(0)

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels,
                             fmin=fmin, fmax=fmax)
        mel_basis[str(fmax) + "_" + str(y.device)] = (
            torch.from_numpy(mel).float().to(y.device)
        )
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)
    
    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[str(y.device)],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)

    mel = torch.matmul(mel_basis[str(fmax) + "_" + str(y.device)], spec)
    mel = spectral_normalize_torch(mel)
    result = mel.squeeze()

    if return_mel_and_spec:
        spec = spectral_normalize_torch(spec)
        return result, spec
    else:
        return result


class HiFiPlusGenerator(torch.nn.Module):
    def __init__(
        self,
        generator,

        use_spectralunet=True,
        spectralunet_block_widths=(8, 16, 24, 32, 64),
        spectralunet_block_depth=5,
        spectralunet_positional_encoding=True,

        use_waveunet=True,
        waveunet_block_widths=(10, 20, 40, 80),
        waveunet_block_depth=4,

        use_spectralmasknet=True,
        spectralmasknet_block_widths=(8, 12, 24, 32),
        spectralmasknet_block_depth=4,

        norm_type: Literal["weight", "spectral"] = "weight",
        use_skip_connect=True,
        waveunet_before_spectralmasknet=True,
        out_ch=1
    ):
        super().__init__()
        self.norm = dict(weight=weight_norm, spectral=spectral_norm)[norm_type]
        self.norm_type = norm_type

        self.use_spectralunet = use_spectralunet
        self.use_waveunet = use_waveunet
        self.use_spectralmasknet = use_spectralmasknet

        self.use_skip_connect = use_skip_connect
        self.waveunet_before_spectralmasknet = waveunet_before_spectralmasknet
        
        self.hifi = generator
        ch = self.hifi.out_channels

        if self.use_spectralunet:
            self.spectralunet = nn_utils.SpectralUNet(
                block_widths=spectralunet_block_widths,
                block_depth=spectralunet_block_depth,
                positional_encoding=spectralunet_positional_encoding,
                norm_type=norm_type,
            )

        if self.use_waveunet:
            self.waveunet = nn_utils.MultiScaleResnet(
                waveunet_block_widths,
                waveunet_block_depth,
                mode="waveunet_k5",
                out_width=ch,
                in_width=ch,
                norm_type=norm_type
            )

        if self.use_spectralmasknet:
            self.spectralmasknet = nn_utils.SpectralMaskNet(
                in_ch=ch,
                block_widths=spectralmasknet_block_widths,
                block_depth=spectralmasknet_block_depth,
                norm_type=norm_type
            )

        self.waveunet_skip_connect = None
        self.spectralmasknet_skip_connect = None
        if self.use_skip_connect:
            self.make_waveunet_skip_connect(ch)
            self.make_spectralmasknet_skip_connect(ch)

        self.conv_post = None
        self.make_conv_post(ch, out_ch=out_ch)

    def make_waveunet_skip_connect(self, ch):
        self.waveunet_skip_connect = self.norm(nn.Conv1d(ch, ch, 1, 1))
        self.waveunet_skip_connect.weight.data = torch.eye(ch, ch).unsqueeze(-1)
        self.waveunet_skip_connect.bias.data.fill_(0.0)

    def make_spectralmasknet_skip_connect(self, ch):
        self.spectralmasknet_skip_connect = self.norm(nn.Conv1d(ch, ch, 1, 1))
        self.spectralmasknet_skip_connect.weight.data = torch.eye(ch, ch).unsqueeze(-1)
        self.spectralmasknet_skip_connect.bias.data.fill_(0.0)

    def make_conv_post(self, ch, out_ch=1):
        self.conv_post = self.norm(nn.Conv1d(ch, out_ch, 7, 1, padding=3))
        self.conv_post.apply(nn_utils.init_weights)

    def apply_spectralunet(self, x_orig):
        if self.use_spectralunet:
            pad_size = (
                utils.closest_power_of_two(x_orig.shape[-1]) - x_orig.shape[-1]
            )
            x = torch.nn.functional.pad(x_orig, (0, pad_size))
            x = self.spectralunet(x)
            x = x[..., : x_orig.shape[-1]]
        else:
            x = x_orig.squeeze(1)
        return x

    def apply_waveunet(self, x):
        x_a = x
        x = self.waveunet(x_a)
        if self.use_skip_connect:
            x += self.waveunet_skip_connect(x_a)
        return x

    def apply_spectralmasknet(self, x):
        x_a = x
        x = self.spectralmasknet(x)
        if self.use_skip_connect:
            x += self.spectralmasknet_skip_connect(x_a)
        return x

    def forward(self, x_orig):
        x = self.apply_spectralunet(x_orig)
        x = self.hifi(x)
        if self.use_waveunet and self.waveunet_before_spectralmasknet:
            x = self.apply_waveunet(x)
        if self.use_spectralmasknet:
            x = self.apply_spectralmasknet(x)
        if self.use_waveunet and not self.waveunet_before_spectralmasknet:
            x = self.apply_waveunet(x)

        x = self.conv_post(x)
        x = torch.tanh(x)
        
        return x


class A2AHiFiPlusGeneratorV2(HiFiPlusGenerator):
    def __init__(
        self,
        generator,

        use_spectralunet=True,
        spectralunet_block_widths=(8, 16, 24, 32, 64),
        spectralunet_block_depth=5,
        spectralunet_positional_encoding=True,

        use_waveunet=True,
        waveunet_block_widths=(10, 20, 40, 80),
        waveunet_block_depth=4,

        use_spectralmasknet=True,
        spectralmasknet_block_widths=(8, 12, 24, 32),
        spectralmasknet_block_depth=4,

        norm_type: Literal["weight", "spectral"] = "weight",
        use_skip_connect=False,
        waveunet_before_spectralmasknet=True,

        waveunet_input: Literal["waveform", "hifi", "both"] = "hifi",
        out_channels=1
    ):
        super().__init__(
            generator,

            use_spectralunet=use_spectralunet,
            spectralunet_block_widths=spectralunet_block_widths,
            spectralunet_block_depth=spectralunet_block_depth,
            spectralunet_positional_encoding=spectralunet_positional_encoding,

            use_waveunet=use_waveunet,
            waveunet_block_widths=waveunet_block_widths,
            waveunet_block_depth=waveunet_block_depth,

            use_spectralmasknet=use_spectralmasknet,
            spectralmasknet_block_widths=spectralmasknet_block_widths,
            spectralmasknet_block_depth=spectralmasknet_block_depth,

            norm_type=norm_type,
            use_skip_connect=use_skip_connect,
            waveunet_before_spectralmasknet=waveunet_before_spectralmasknet,
            out_ch=out_channels
        )
        self.waveunet_input = waveunet_input

        self.waveunet_conv_pre = None
        if self.waveunet_input == "waveform":
            self.waveunet_conv_pre = weight_norm(
                nn.Conv1d(
                    1, self.hifi.out_channels, 1
                )
            )
        elif self.waveunet_input == "both":
            self.waveunet_conv_pre = weight_norm(
                nn.Conv1d(
                    1 + self.hifi.out_channels, self.hifi.out_channels, 1
                )
            )
        
    @staticmethod
    def get_melspec(x):
        shape = x.shape
        x = x.view(shape[0] * shape[1], shape[2])
        x = mel_spectrogram(x, 1024, 80, 16000, 256, 1024, 0, 8000)
        x = x.view(shape[0], -1, x.shape[-1])
        return x
    
    @staticmethod
    def get_spec(x):
        shape = x.shape
        x = x.view(shape[0] * shape[1], shape[2])
        x = mel_spectrogram(x, 1024, 80, 16000, 256,
                            1024, 0, 8000, return_mel_and_spec=True)[1]
        x = x.view(shape[0], -1, x.shape[-1])
        return x

    def apply_waveunet_a2a(self, x, x_orig):
        if self.waveunet_input == "waveform":
            x_a = self.waveunet_conv_pre(x_orig)
        elif self.waveunet_input == "both":
            x_a = torch.cat([x, x_orig], 1)
            x_a = self.waveunet_conv_pre(x_a)
        elif self.waveunet_input == "hifi":
            x_a = x
        else:
            raise ValueError
        x = self.waveunet(x_a)
        if self.use_skip_connect:
            x += self.waveunet_skip_connect(x_a)
        return x

    def forward(self, noisy_audio, **batch):
        print("orig", noisy_audio.shape)
        x = noisy_audio
        if len(x.shape) == 2:
            x = x.unsqueeze(1) # adding channel dimension

        x_orig = x.clone()

        x_orig = x_orig[:, :, : x_orig.shape[2] // 1024 * 1024]
        x = self.get_melspec(x_orig)

        mel_spec_before = x.clone()
        print("mel spec", x.shape)
        x = self.apply_spectralunet(x)

        x = self.hifi(x)
        print("after hifi", x.shape)
        if self.use_waveunet and self.waveunet_before_spectralmasknet and not self.hifi.return_stft:
            x = self.apply_waveunet_a2a(x, x_orig)
            print("after waveunet", x.shape)
        if self.use_spectralmasknet:
            x = self.apply_spectralmasknet(x)
            print("after spectral", x.shape)
        if self.use_waveunet and not self.waveunet_before_spectralmasknet:
            x = self.apply_waveunet_a2a(x, x_orig)

        x = self.conv_post(x)

        x = torch.tanh(x)
        mel_spec_after = self.get_melspec(x)
        print('X', x.isnan().any())
        return {"clean_audio_predicted": x, "mel_spec_before": mel_spec_before, "mel_spec_after": mel_spec_after}
    