import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

LRELU_SLOPE = 0.1


class DiscriminatorP(torch.nn.Module):
    def __init__(self,
                 period, 
                 kernel_size=5, 
                 stride=3, 
                 use_spectral_norm=False, 
                 use_id_channel=False,
                 embedding_dim=4,
                 embedding_count=0,
                 channel_count=1):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        init_channels = 1 + embedding_dim if use_id_channel else 1
        if channel_count != 1:
            init_channels = channel_count
        kernel = 1
        self.convs = nn.ModuleList([
            norm_f(Conv2d(init_channels, 32, (kernel_size, kernel), (stride, 1), padding=(2, kernel // 2))),
            norm_f(Conv2d(32, 128, (kernel_size, kernel), (stride, 1), padding=(2, kernel // 2))),
            norm_f(Conv2d(128, 512, (kernel_size, kernel), (stride, 1), padding=(2, kernel // 2))),
            norm_f(Conv2d(512, 1024, (kernel_size, kernel), (stride, 1), padding=(2, kernel // 2))),
            norm_f(Conv2d(1024, 1024, (kernel_size, kernel), 1, padding=(2, kernel // 2))),
        ])
        
        if use_id_channel:
            self.id_embedding = nn.Embedding(embedding_count, embedding_dim)

        self.use_id_channel = use_id_channel
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x, speaker_id=None):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        if self.use_id_channel:
            embed = self.id_embedding(speaker_id)
            embed = embed.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.shape[2], x.shape[3])
            x = torch.cat([x, embed], dim=1)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, **kwargs):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(2, **kwargs),
            DiscriminatorP(3, **kwargs),
            DiscriminatorP(5, **kwargs),
            DiscriminatorP(7, **kwargs),
            DiscriminatorP(11, **kwargs),
        ])

    def forward(self, audio, ids=None, **batch):
        y = audio
        y_ds = []
        fmaps = []
        for i, d in enumerate(self.discriminators):
            y_d, fmap = d(y, ids)
            y_ds.append(y_d)
            fmaps.append(fmap)

        return {"estimation": y_ds, "fmap": fmaps}


class DiscriminatorS(torch.nn.Module):
    def __init__(self, 
                 use_spectral_norm=False,
                 use_id_channel=False,
                 embedding_dim=4,
                 embedding_count=0,
                 channel_count=1):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        init_channels = 1 + embedding_dim if use_id_channel else 1
        if channel_count != 1:
            init_channels = channel_count
        self.convs = nn.ModuleList([
            norm_f(Conv1d(init_channels, 128, 15, 1, padding=7)),
            norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])

        if use_id_channel:
            self.id_embedding = nn.Embedding(embedding_count, embedding_dim)

        self.use_id_channel = use_id_channel
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x, speaker_id=None):
        fmap = []
        if self.use_id_channel:
            embed = self.id_embedding(speaker_id)
            embed = embed.unsqueeze(-1).expand(-1, -1, x.shape[2])
            x = torch.cat([x, embed], dim=1)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap
    

class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self, **kwargs):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True, **kwargs),
            DiscriminatorS(**kwargs),
            DiscriminatorS(**kwargs),
        ])
        self.meanpools = nn.ModuleList([
            AvgPool1d(4, 2, padding=2),
            AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, audio, ids=None, **batch):
        y = audio
        y_ds = []
        fmaps = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i-1](y)
            y_d, fmap = d(y, ids)
            y_ds.append(y_d)
            fmaps.append(fmap)

        return {"estimation": y_ds, "fmap": fmaps}
    

class BSSDiscriminator(nn.Module):
    def __init__(self, use_id_channel=False, embedding_dim=4, embedding_count=0, channel_count=1):
        super().__init__()

        self.channels = channel_count

        self.mpd = MultiPeriodDiscriminator(use_id_channel=use_id_channel, 
                                            embedding_dim=embedding_dim, 
                                            embedding_count=embedding_count,
                                            channel_count=channel_count)
        self.msd = MultiScaleDiscriminator(use_id_channel=use_id_channel, 
                                            embedding_dim=embedding_dim, 
                                            embedding_count=embedding_count,
                                            channel_count=channel_count)

    def forward(self, audios, mix_audio, ids=None, **batch):
        #audios = audios[:, :1, :]
        B, C, _ = audios.shape
        '''
        first = torch.cat([audios[:, :1, :], mix_audio], dim=1)
        second = torch.cat([audios[:, 1:, :], mix_audio], dim=1)
        audios = torch.cat([first, second], dim=0)'''
        
        if self.channels != C:
            audios = audios.reshape(B * C, 1, -1)

        if ids is not None:
            ids = ids.flatten()
        res = self.mpd(audios, ids)

        for key, value in self.msd(audios, ids).items():
            res[key].extend(value)

        if self.channels != C:
            res["estimation"] = [x.reshape(B, C, -1) for x in res["estimation"]]
            for i in range(len(res["fmap"])):
                res["fmap"][i] = [x.reshape(B, C, *x.shape[1:]) for x in res["fmap"][i]]

        return res
    

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.mpd = MultiPeriodDiscriminator()
        self.msd = MultiScaleDiscriminator()

    def forward(self, clean_audio, clean_audio_predicted, **batch):
        res = self.mpd(clean_audio, clean_audio_predicted)

        for key, value in self.msd(clean_audio, clean_audio_predicted).items():
            res[key].extend(value)
        return res