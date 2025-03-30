import torch

from src.model.hifi import A2AHiFiPlusGeneratorV2, mel_spectrogram


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
