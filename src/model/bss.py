import torch

from src.model.hifi import A2AHiFiPlusGeneratorV2, mel_spectrogram
import src.utils.nn_utils as nn_utils
import torch.nn as nn


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

        self.mask1 = nn_utils.SpectralMaskNet(
                in_ch=ch,
                block_widths=(8, 12, 24, 32),
                block_depth=4,
                norm_type='weight'
            )
        self.mask2 = nn_utils.SpectralMaskNet(
                in_ch=ch,
                block_widths=(8, 12, 24, 32),
                block_depth=4,
                norm_type='weight'
            )
        
        self.conv_post1 = self.norm(nn.Conv1d(ch, 1, 7, 1, padding=3))
        self.conv_post1.apply(nn_utils.init_weights)
        self.conv_post2 = self.norm(nn.Conv1d(ch, 1, 7, 1, padding=3))
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
        x = mix_audio
        x_orig = x.clone()

        # x_orig = x_orig[:, :, : x_orig.shape[2] // 1024 * 1024]
        x = self.get_melspec(x)

        mel_spec_before = x.clone()
        x = self.apply_spectralunet(x)

        x = self.hifi(x)

        if self.use_waveunet and self.waveunet_before_spectralmasknet and not self.hifi.return_stft:
            x = self.apply_waveunet_a2a(x, x_orig)

        masked_1 = self.mask1(x)
        masked_2 = self.mask2(x)

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
        mel_spec_after = self.get_melspec(x)

        return {"separated_audios": x, "fake_melspec": mel_spec_after}
