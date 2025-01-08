import torch
from torch import nn
import torch.nn.functional as F

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
