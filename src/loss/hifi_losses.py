import torch
from torch import nn
import torch.nn.functional as F

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
    

class BSSGeneratorLoss(nn.Module):
    """
    Weighted sum of losses for generator
    """

    def __init__(self):
        super().__init__()

    def forward(self, real_melspec, fake_melspec, separated_audios, audios, ids, discriminator, **batch):
        B, N, _ = separated_audios.shape
        permutations = list(itertools.permutations(range(N)))
        order = torch.zeros((B, N), dtype=torch.int32)

        for b in range(B):
            loss = None
            with torch.no_grad():
                real_estimations = discriminator(audios[b:b + 1], ids)

            for p in permutations:
                l1_loss = F.l1_loss(fake_melspec[b, p, ...], real_melspec[b, ...])
                with torch.no_grad():
                    discriminator_estimations = discriminator(separated_audios[b:b+1, p, :], ids)

                g_loss = generator_loss(discriminator_estimations["estimation"])[0]
                feat_loss = feature_loss(real_estimations["fmap"], discriminator_estimations["fmap"])
                cur_loss = g_loss + feat_loss + l1_loss * 45
                # cur_loss = l1_loss * 45

                '''
                for i in range(2):
                    for j in range(2):
                        print(F.l1_loss(real_melspec[0, i], fake_melspec[0, j]))'''
                if loss is None or cur_loss < loss:
                    loss = cur_loss
                    order[b, :] = torch.tensor(p, dtype=torch.int32)

        reordered = separated_audios[torch.arange(separated_audios.shape[0])[:, None], order]
        mel_reordered = fake_melspec[torch.arange(fake_melspec.shape[0])[:, None], order]

        fake_estimations = discriminator(reordered, ids)
        real_estimations = discriminator(audios, ids)

        losses = {}
        losses["feature_loss"] = feature_loss(real_estimations["fmap"], fake_estimations["fmap"])
        losses["g_loss"] = generator_loss(fake_estimations["estimation"])[0]
        losses["l1_loss"] = F.l1_loss(mel_reordered, real_melspec)
        losses["generator_loss"] = losses["feature_loss"] + losses["g_loss"] + losses["l1_loss"] * 45
        # losses["generator_loss"] = losses["l1_loss"] * 45

        return losses, order
