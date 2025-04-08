import torch

from src.model.hifi import A2AHiFiPlusGeneratorV2, mel_spectrogram
import src.utils.nn_utils as nn_utils
from torch.nn.utils import weight_norm, spectral_norm
import torch.nn as nn

import copy


class A2AHiFiPlusGeneratorBSS(A2AHiFiPlusGeneratorV2):
    @staticmethod
    def get_melspec(x):
        shape = x.shape
        x = x.reshape(shape[0] * shape[1], shape[2])
        x = mel_spectrogram(x, 1024, 80, 16000, 256, 1024, 0, 8000)
        if shape[1] > 1:
            x = x.view(shape[0], shape[1], -1, x.shape[-1])
        else:
            x = x.view(shape[0], -1, x.shape[-1])
        return x
    
    def forward(self, mix_audio, **batch):
        x = mix_audio
        x_orig = x.clone()

        # x_orig = x_orig[:, :, : x_orig.shape[2] // 1024 * 1024]
        x = self.get_melspec(x)

        mel_spec_before = x.clone()
        x = self.apply_spectralunet(x)

        x = self.hifi(x)

        if self.use_waveunet and self.waveunet_before_spectralmasknet and not self.hifi.return_stft:
            x = self.apply_waveunet_a2a(x, x_orig)

        if self.use_spectralmasknet:
            x = self.apply_spectralmasknet(x)

        if self.use_waveunet and not self.waveunet_before_spectralmasknet:
            x = self.apply_waveunet_a2a(x, x_orig)
        x = self.conv_post(x)

        x = torch.tanh(x)

        mel_spec_after = self.get_melspec(x)

        return {"separated_audios": x, "fake_melspec": mel_spec_after}
    

class A2AHiFiPlusGeneratorBSSV2(A2AHiFiPlusGeneratorV2):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        ch = self.hifi.out_channels
        self.ch = ch

        self.mask1 = nn_utils.SpectralMaskNet(
                in_ch=ch // 2,
                block_widths=(8, 12, 24, 32),
                block_depth=4,
                norm_type='weight'
            )
        self.mask2 = nn_utils.SpectralMaskNet(
                in_ch=ch // 2,
                block_widths=(8, 12, 24, 32),
                block_depth=4,
                norm_type='weight'
            )
        
        self.sep1 = nn_utils.MultiScaleResnet(
                (10, 20, 40),
                3,
                mode="waveunet_k5",
                out_width=64 * 2,
                in_width=64,
                norm_type='weight'
            )
        
        self.sep2 = nn_utils.MultiScaleResnet(
                (10, 20, 40),
                3,
                mode="waveunet_k5",
                out_width=64 * 2,
                in_width=64,
                norm_type='weight'
            )
        
        self.waveunet1 = nn_utils.MultiScaleResnet(
                (10, 20, 40, 80),
                4,
                mode="waveunet_k5",
                out_width=ch // 2,
                in_width=ch // 2,
                norm_type='weight'
            )
        self.waveunet2 = nn_utils.MultiScaleResnet(
                (10, 20, 40, 80),
                4,
                mode="waveunet_k5",
                out_width=ch // 2,
                in_width=ch // 2,
                norm_type='weight'
            )
        
        self.conv_post1 = self.norm(nn.Conv1d(ch // 2, 1, 7, 1, padding=3))
        self.conv_post1.apply(nn_utils.init_weights)
        self.conv_post2 = self.norm(nn.Conv1d(ch // 2, 1, 7, 1, padding=3))
        self.conv_post2.apply(nn_utils.init_weights)

    @staticmethod
    def get_melspec(x):
        shape = x.shape
        x = x.reshape(shape[0] * shape[1], shape[2])
        x = mel_spectrogram(x, 1024, 80, 16000, 256, 1024, 0, 8000)
        if shape[1] > 1:
            x = x.view(shape[0], shape[1], -1, x.shape[-1])
        else:
            x = x.view(shape[0], -1, x.shape[-1])
        return x
    
    def forward(self, mix_audio, **batch):
        target_norm = mix_audio.pow(2).mean(dim=-1, keepdim=True).sqrt().detach()
        x = mix_audio
        x_orig = x.clone()

        # x_orig = x_orig[:, :, : x_orig.shape[2] // 1024 * 1024]
        x = self.get_melspec(x)

        mel_spec_before = x.clone()
        # x = self.apply_spectralunet(x)

        #sep1 = x[:, :64, :].clone()
        #sep2 = x[:, 64:, :].clone()

        #sep1 = self.sep1(sep1)
        #sep2 = self.sep2(sep2)

        #sep1 = self.hifi(sep1)
        #sep2 = self.hifi(sep2)
        #x = torch.cat([sep1, sep2], dim=1)
        
        x = self.hifi(x)
        
        if self.use_waveunet and self.waveunet_before_spectralmasknet and not self.hifi.return_stft:
            x = self.apply_waveunet_a2a(x, x_orig)

        masked_1 = self.mask1(x[:, :self.ch // 2, :])
        masked_2 = self.mask2(x[:, self.ch // 2:, :])
        #masked_1 = self.mask1(x)
        #masked_2 = self.mask2(x)

        masked_1 = self.waveunet1(masked_1)
        masked_2 = self.waveunet2(masked_2)

        #masked_1 += self.spectralmasknet_skip_connect(x)
        #masked_2 += self.spectralmasknet_skip_connect(x)
        '''
        if self.use_waveunet and not self.waveunet_before_spectralmasknet:
            x = self.apply_waveunet_a2a(x, x_orig)'''

        # x = self.conv_post(x)

        masked_1 = self.conv_post1(masked_1)
        masked_2 = self.conv_post2(masked_2)


        x = torch.cat([masked_1, masked_2], dim=1)

        x = torch.tanh(x)
        current_norm = x.pow(2).mean(dim=-1, keepdim=True).sqrt()

        # 
        # x = x * (target_norm / (current_norm + 1e-12))
        '''
        print(target_norm, current_norm)
        print(target_norm / current_norm)
        print('---------------------------')'''

        mel_spec_after = self.get_melspec(x)


        return {"separated_audios": x, "fake_melspec": mel_spec_after}
    

class A2AHiFiPlusGeneratorBSSV3(nn.Module):
    def __init__(self, generator, generator2, **kwargs):
        super().__init__()

        self.hifi1 = generator
        self.hifi2 = generator2
        ch = 8
        
        self.waveunet1 = nn_utils.MultiScaleResnet(
                (10, 20, 40, 80),
                4,
                mode="waveunet_k5",
                out_width=16,
                in_width=1,
                norm_type='weight'
            )

        self.waveunet2 = nn_utils.MultiScaleResnet(
                (10, 20, 40, 80),
                4,
                mode="waveunet_k5",
                out_width=16,
                in_width=17,
                norm_type='weight'
            )

        self.waveunet3 = nn_utils.MultiScaleResnet(
                (10, 20, 40, 80),
                4,
                mode="waveunet_k5",
                out_width=16,
                in_width=17,
                norm_type='weight'
            )
        
        self.waveunet4 = nn_utils.MultiScaleResnet(
                (10, 20, 40, 80),
                4,
                mode="waveunet_k5",
                out_width=2,
                in_width=17,
                norm_type='weight'
            )
        
        self.conv_post1 = weight_norm(nn.Conv1d(ch, 1, 7, 1, padding=3))
        self.conv_post1.apply(nn_utils.init_weights)
        self.conv_post2 = weight_norm(nn.Conv1d(ch, 1, 7, 1, padding=3))
        self.conv_post2.apply(nn_utils.init_weights)

    def forward(self, mix_audio, **batch):
        x_orig = mix_audio.clone()
        x = self.waveunet1(mix_audio)
        x = torch.cat([x, x_orig], dim=1)

        x = self.waveunet2(x)
        x = torch.cat([x, x_orig], dim=1)

        x = self.waveunet3(x)
        x = torch.cat([x, x_orig], dim=1)
        
        x = self.waveunet4(x)
        x = torch.tanh(x)

        mel1 = self.get_melspec(x[:, 0:1, :])
        mel2 = self.get_melspec(x[:, 1:2, :])

        x1 = self.hifi1(mel1)
        x2 = self.hifi2(mel2)

        x1 = self.conv_post1(x1)
        x2 = self.conv_post2(x2)

        x = torch.cat([x1, x2], dim=1)

        mel_spec_after = self.get_melspec(x)


        return {"separated_audios": x, "fake_melspec": mel_spec_after}
    
    @staticmethod
    def get_melspec(x):
        shape = x.shape
        x = x.reshape(shape[0] * shape[1], shape[2])
        x = mel_spectrogram(x, 1024, 80, 16000, 256, 1024, 0, 8000)
        if shape[1] > 1:
            x = x.view(shape[0], shape[1], -1, x.shape[-1])
        else:
            x = x.view(shape[0], -1, x.shape[-1])
        return x
