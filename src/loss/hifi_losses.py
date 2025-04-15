import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio

import itertools

from src.model.hifi import mel_spectrogram


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())
    ''' 
    for dm in mix_estimation:
        loss += torch.mean(dm**2)'''

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1 - dg)**2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses


class DiscriminatorLoss(nn.Module):
    """
    Loss for discriminator
    """

    def __init__(self):
        super().__init__()

    def forward(self, real_estimation, fake_estimation, **batch):
        return {"discriminator_loss": discriminator_loss(real_estimation, fake_estimation)[0]}


class GeneratorLoss(nn.Module):
    """
    Weighted sum of losses for generator
    """

    def __init__(self):
        super().__init__()

    def forward(self, real_estimation, fake_estimation, mel_spec_before, mel_spec_after, real_fmap, fake_fmap, **batch):
        losses = {}
        losses["feature_loss"] = feature_loss(real_fmap, fake_fmap)
        losses["g_loss"] = generator_loss(fake_estimation)[0]
        losses["l1_loss"] = F.l1_loss(mel_spec_before, mel_spec_after)
        losses["generator_loss"] = losses["feature_loss"] + losses["g_loss"] + losses["l1_loss"] * 45

        return losses
    
import time
class BSSGeneratorLoss(nn.Module):
    """
    Weighted sum of losses for generator
    """

    def __init__(self):
        super().__init__()
        self.si_snr_loss = ScaleInvariantSignalNoiseRatio()

    def forward(self, real_melspec, fake_melspec, separated_audios, audios, ids, discriminator, **batch):
        B, N, _ = separated_audios.shape
        permutations = list(itertools.permutations(range(N)))
        order = torch.zeros((B, N), dtype=torch.int32)
        losses = [None] * B
        st = time.time()
        
        with torch.no_grad():
            real_estimations = discriminator(audios, batch["mix_audio"], ids)

        for p in permutations:
        
            bef = time.time()
            
            with torch.no_grad():
                discriminator_estimations = discriminator(separated_audios[:, p, :], batch["mix_audio"], ids)

            cur_loss = [0] * B
            for b in range(B):
                sp = time.time()
                cur_loss[b] += -self.si_snr_loss(separated_audios[b, p, :], audios[b]) * 1
                # print(p)
                
                # print('SI-SNR LOSS:', cur_loss[b] / 45)
                #print('HA')
                for i in range(2):
                    for j in range(2):
                        pass
                        # print(self.si_snr_loss(separated_audios[b, i, :], audios[b, j, :]))
                # mix_audio = (audios[b, 0, :] + audios[b, 1, :]) / 2
                #print('MIXES', self.si_snr_loss(audios[b, 0, :], mix_audio), self.si_snr_loss(audios[b, 1, :], mix_audio))
                #print('MIXES', self.si_snr_loss(separated_audios[b, 0, :], mix_audio), self.si_snr_loss(separated_audios[b, 1, :], mix_audio))
                #print('DIFF', self.si_snr_loss(audios[b, 0, :], audios[b, 1, :]))
                # cur_loss[b] += F.l1_loss(fake_melspec[b, p], real_melspec[b]) * 45
                # print("si-snr", p, time.time() - sp)
                '''
                lss = time.time()
                
                ch = [x[b:b+1] for x in discriminator_estimations["estimation"]]
                cur_loss[b] += generator_loss(ch)[0]
                fmap_f, fmap_r = [], []

                for d in discriminator_estimations["fmap"]:
                    for fmap in d:
                        fmap_f.append(fmap[b:b+1, ...])
                for d in real_estimations["fmap"]:
                    for fmap in d:
                        fmap_r.append(fmap[b:b+1, ...])
                cur_loss[b] += feature_loss(fmap_f, fmap_r)'''

                if losses[b] is None or cur_loss[b] < losses[b]:
                    losses[b] = cur_loss[b]
                    order[b, :] = torch.tensor(p, dtype=torch.int32)
                # print("LOSS CALC", time.time() - lss)
        '''
        print("FIRST", time.time() - st)
        st = time.time()
        for b in range(B):
            loss = None
            with torch.no_grad():
                real_estimations = discriminator(audios[b:b+1], ids[b:b+1])

            for p in permutations:
                l1_loss = F.l1_loss(fake_melspec[b, p, ...], real_melspec[b, ...])
                snr_loss = -self.si_snr_loss(separated_audios[b, p, :], audios[b])
                with torch.no_grad():
                    discriminator_estimations = discriminator(separated_audios[b:b+1, p, :], ids[b:b+1])

                g_loss = generator_loss(discriminator_estimations["estimation"])[0]
                feat_loss = feature_loss(real_estimations["fmap"], discriminator_estimations["fmap"])
                cur_loss = g_loss + feat_loss + snr_loss * 4.5
                # cur_loss = l1_loss * 45

                if loss is None or cur_loss < loss:
                    loss = cur_loss
                    order[b, :] = torch.tensor(p, dtype=torch.int32)
        print("SECOND", time.time() - st)
        st = time.time()'''
        reordered = separated_audios[torch.arange(separated_audios.shape[0])[:, None], order]
        mel_reordered = fake_melspec[torch.arange(fake_melspec.shape[0])[:, None], order]

        fake_estimations = discriminator(reordered, batch["mix_audio"], ids)
        real_estimations = discriminator(audios, batch["mix_audio"], ids)

        losses = {}
        losses["feature_loss"] = feature_loss(real_estimations["fmap"], fake_estimations["fmap"])
        losses["g_loss"] = generator_loss(fake_estimations["estimation"])[0]
        losses["l1_loss"] = F.l1_loss(mel_reordered, real_melspec)
        losses["snr_loss"] = -self.si_snr_loss(reordered, audios)# + self.si_snr_loss(reordered[:, 0, :], reordered[:, 1, :]) / 4
        # print('total', losses['snr_loss'])
        losses["generator_loss"] = losses["feature_loss"] + losses["g_loss"] + losses["snr_loss"]
        # losses["generator_loss"] = losses["snr_loss"] * 1
        # losses["generator_loss"] = losses["snr_loss"]
        # print("OTHER", time.time() - st)
        # losses["generator_loss"] = losses["l1_loss"] * 45

        return losses, order
