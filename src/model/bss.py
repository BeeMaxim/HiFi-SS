import torch

from src.model.hifi import A2AHiFiPlusGeneratorV2, mel_spectrogram
import src.utils.nn_utils as nn_utils
from torch.nn.utils import weight_norm, spectral_norm
import torch.nn as nn

import src.utils.hifi_utils as utils

import copy

LRELU_SLOPE = 0.1


class ResBlock(nn.Module):
    def __init__(self, kernel_size, channels, D_r): # D_r[n]
        super().__init__()

        self.convs = nn.ModuleList()

        for m in range(len(D_r)):
            layer = nn.ModuleList()
            for l in range(len(D_r[m])):
                layer.append(nn.Sequential(
                    nn.LeakyReLU(LRELU_SLOPE),
                    nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, dilation=D_r[m][l], padding="same")
                ))
            self.convs.append(layer)

    def forward(self, x):
        for conv in self.convs:
            for layer in conv:
                y = layer(x)
            x = x + y
        return x


class MRF(nn.Module):
    def __init__(self, channels, k_r, D_r):
        super().__init__()

        self.res_blocks = nn.ModuleList()

        for n in range(len(k_r)):
            self.res_blocks.append(ResBlock(k_r[n], channels, D_r[n]))

    def forward(self, x):
        res = None
        for res_block in self.res_blocks:
            if res is None:
                res = res_block(x)
            else:
                res = res + res_block(x)
        return res / len(self.res_blocks)


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

        ch = 8
        self.ch = ch


        self.mask1 = nn_utils.SpectralMaskNet(
                in_ch=4,
                block_widths=(8, 12, 24, 32),
                block_depth=4,
                norm_type='weight'
            )
        self.mask2 = nn_utils.SpectralMaskNet(
                in_ch=4,
                block_widths=(8, 12, 24, 32),
                block_depth=4,
                norm_type='weight'
            )
           
        self.waveunet1 = nn_utils.MultiScaleResnet(
                (10, 20, 40, 80),
                4,
                mode="waveunet_k5",
                out_width=4,
                in_width=5,
                norm_type='weight'
            )
        self.waveunet2 = nn_utils.MultiScaleResnet(
                (10, 20, 40, 80),
                4,
                mode="waveunet_k5",
                out_width=4,
                in_width=5,
                norm_type='weight'
            )
        
        #self.mrf1 = MRF(4, [3, 7, 11], [[[1, 1], [3, 1], [5, 1]], [[1, 1], [3, 1], [5, 1]], [[1, 1], [3, 1], [5, 1]]])
        #self.mrf2 = MRF(4, [3, 7, 11], [[[1, 1], [3, 1], [5, 1]], [[1, 1], [3, 1], [5, 1]], [[1, 1], [3, 1], [5, 1]]])

        self.ups_post1 = self.norm(nn.Conv1d(ch // 2, 1, 7, 1, padding=3))
        self.ups_post1.apply(nn_utils.init_weights)
        self.ups_post2 = self.norm(nn.Conv1d(ch // 2, 1, 7, 1, padding=3))
        self.ups_post2.apply(nn_utils.init_weights)
        
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
        '''
        masked_1 = self.conv_post1(x[:, :self.ch // 2, :])
        masked_2 = self.conv_post2(x[:, self.ch // 2:, :])

        y = torch.cat([masked_1, masked_2], dim=1)
        y = torch.tanh(y)'''

        '''
        y = self.ups_post(x)
        y = torch.tanh(y)

        y_mel = self.get_melspec(y)'''

        '''
        x = self.hi1(x)
        x = self.hi2(x)
        x = self.hi3(x)'''

        '''
        if self.training:
            dropout_rate = 0.9
            drop_mask = (torch.rand(x_orig.shape[0]) < dropout_rate).float()
            first_orig = x_orig * drop_mask[:, None, None].to(x_orig.device) / dropout_rate
        else:
            first_orig = x_orig'''
        
        
        if self.use_waveunet and self.waveunet_before_spectralmasknet:
            x = self.apply_waveunet_a2a(x, x_orig)
        y = x.detach()
        if self.use_waveunet and self.waveunet_before_spectralmasknet:
            y = self.apply_waveunet_a2a(y, x_orig)

        
        masked_1 = self.mask1(x[:, :self.ch // 2, :])
        masked_2 = self.mask2(x[:, self.ch // 2:, :])
        y_1 = self.mask1(y[:, :self.ch // 2, :])
        y_2 = self.mask2(y[:, self.ch // 2:, :])

        #masked_1 = self.mask1(masked_1.detach())
        #masked_2 = self.mask1(masked_2.detach())

        #masked_1 = x[:, :self.ch // 2, :]
        #masked_2 = x[:, self.ch // 2:, :]

        masked_1 = self.waveunet1(torch.cat([masked_1, x_orig], dim=1))
        masked_2 = self.waveunet2(torch.cat([masked_2, x_orig], dim=1))
        y_1 = self.waveunet1(torch.cat([y_1, x_orig], dim=1))
        y_2 = self.waveunet2(torch.cat([y_2, x_orig], dim=1))
        #masked_1 = self.waveunet1(masked_1)
        #masked_2 = self.waveunet2(masked_2)

        #masked_1 = self.mrf1(masked_1)
        #masked_2 = self.mrf2(masked_2)

        #masked_11 = self.waveunet11(torch.cat([masked_1, masked_2[:, :self.ch // 4, :]], dim=1))
        #masked_12 = self.waveunet12(torch.cat([masked_2, masked_1[:, :self.ch // 4, :]], dim=1))

        #masked_1 += self.spectralmasknet_skip_connect(x)
        #masked_2 += self.spectralmasknet_skip_connect(x)
        '''
        if self.use_waveunet and not self.waveunet_before_spectralmasknet:
            x = self.apply_waveunet_a2a(x, x_orig)'''

        # x = self.conv_post(x)

        masked_1 = self.conv_post1(masked_1)
        masked_2 = self.conv_post2(masked_2)

        y_1 = self.conv_post1(y_1)
        y_2 = self.conv_post2(y_2)


        x = torch.cat([masked_1, masked_2], dim=1)
        x = torch.tanh(x)

        y = torch.cat([y_1, y_2], dim=1)
        y = torch.tanh(y)

        #x[:, :, :] = 0.1
        current_norm = x.pow(2).mean(dim=-1, keepdim=True).sqrt()

        # 
        # x = x * (target_norm / (current_norm + 1e-12))
        '''
        print(target_norm, current_norm)
        print(target_norm / current_norm)
        print('---------------------------')'''

        mel_spec_after = self.get_melspec(x)

        return {"separated_audios": y, "upsampler_audios": x, "fake_melspec": mel_spec_after}
    

class A2AHiFiPlusGeneratorBSSV3(nn.Module):
    def __init__(self, generator, generator2, **kwargs):
        super().__init__()

        self.hifi1 = generator
        self.hifi2 = generator2
        ch = 8

        self.spectralunet = nn_utils.SpectralUNet(
                block_widths=(8, 16, 24, 32, 64),
                block_depth=5,
                positional_encoding=True,
                norm_type='weight',
            )
        
        self.waveunet1 = nn_utils.MultiScaleResnet(
                (10, 20, 40, 80),
                4,
                mode="waveunet_k5",
                out_width=8,
                in_width=9,
                norm_type='weight'
            )

        self.waveunet2 = nn_utils.MultiScaleResnet(
                (10, 20, 40, 80),
                4,
                mode="waveunet_k5",
                out_width=8,
                in_width=9,
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
        
        self.post1 = nn_utils.MultiScaleResnet(
                (10, 20, 40, 80),
                4,
                mode="waveunet_k5",
                out_width=8,
                in_width=10,
                norm_type='weight'
            )
        self.post2 = nn_utils.MultiScaleResnet(
                (10, 20, 40, 80),
                4,
                mode="waveunet_k5",
                out_width=8,
                in_width=10,
                norm_type='weight'
            )
        
        self.conv_post1 = weight_norm(nn.Conv1d(ch, 1, 7, 1, padding=3))
        self.conv_post1.apply(nn_utils.init_weights)
        self.conv_post2 = weight_norm(nn.Conv1d(ch, 1, 7, 1, padding=3))
        self.conv_post2.apply(nn_utils.init_weights)

    def forward(self, mix_audio, **batch):
        x = mix_audio
        x_orig = mix_audio.clone()

        mel = self.get_melspec(x)

        mel = self.apply_spectralunet(mel)

        x1 = self.hifi1(mel[:, :64, :])
        x2 = self.hifi2(mel[:, 64:, :])
        '''
        y1 = self.post1(torch.cat([x1, x_orig, x2[:, 0:1, :]], dim=1))
        y2 = self.post2(torch.cat([x2, x_orig, x1[:, 0:1, :]], dim=1))

        y1 = self.conv_post1(y1)
        y2 = self.conv_post2(y2)

        x = torch.cat([y1, y2], dim=1)
        x = torch.tanh(x)'''

        x1 = self.waveunet1(torch.cat([x1, x_orig], dim=1))
        x2 = self.waveunet2(torch.cat([x2, x_orig], dim=1))
        #x1 = self.waveunet1(x1)
        #x2 = self.waveunet2(x2)

        x1 = self.conv_post1(x1)
        x2 = self.conv_post2(x2)

        x = torch.cat([x1, x2], dim=1)
        x = torch.tanh(x)

        mel_spec_after = self.get_melspec(x)


        return {"separated_audios": x, "fake_melspec": mel_spec_after}
    
    def apply_spectralunet(self, x_orig):
        pad_size = (
            utils.closest_power_of_two(x_orig.shape[-1]) - x_orig.shape[-1]
        )
        x = torch.nn.functional.pad(x_orig, (0, pad_size))
        x = self.spectralunet(x)
        x = x[..., : x_orig.shape[-1]]

        return x

    
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
    

class A2AHiFiPlusGeneratorBSSV4(nn.Module):
    def __init__(self, generator, **kwargs):
        super().__init__()
        self.hifi = generator

        self.conv_post = weight_norm(nn.Conv1d(8, 1, 7, 1, padding=3))
        self.conv_post.apply(nn_utils.init_weights)

    def forward(self, mix_audio, audios, **batch):
        x = self.get_melspec(audios[:, :1, :])

        x = self.hifi(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        mel = self.get_melspec(x)

        return {"separated_audios": x, "fake_melspec": mel}

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
    

class TestModel(A2AHiFiPlusGeneratorBSSV2):
    def forward(self, mix_audio, **batch):
        target_norm = mix_audio.pow(2).mean(dim=-1, keepdim=True).sqrt().detach()
        x = mix_audio
        x_orig = x.clone()

        x = self.get_melspec(x)


        x = self.hifi(x)
        after_hifi = x.detach().cpu()
        
        if self.use_waveunet and self.waveunet_before_spectralmasknet:
            x = self.apply_waveunet_a2a(x, x_orig)

        after_waveunet = x.detach().cpu()

        masked_1 = self.mask1(x[:, :self.ch // 2, :])
        masked_2 = self.mask2(x[:, self.ch // 2:, :])

        after_mask1 = masked_1.detach().cpu()
        after_mask2 = masked_2.detach().cpu()

        masked_1 = self.waveunet1(torch.cat([masked_1, x_orig], dim=1))
        masked_2 = self.waveunet2(torch.cat([masked_2, x_orig], dim=1))


        masked_1 = self.conv_post1(masked_1)
        masked_2 = self.conv_post2(masked_2)


        x = torch.cat([masked_1, masked_2], dim=1)

        x = torch.tanh(x)

        mel_spec_after = self.get_melspec(x)


        return {"separated_audios": x, "fake_melspec": mel_spec_after, "after_hifi": after_hifi,
                "after_waveunet": after_waveunet,
                "after_mask1": after_mask1,
                "after_mask2": after_mask2}
