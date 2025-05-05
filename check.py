import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import BSSTrainer, Inferencer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

from src.model.disriminators import BSSDiscriminator

from src.model.hifi import A2AHiFiPlusGeneratorV2, mel_spectrogram
import src.utils.nn_utils as nn_utils
from torch.nn.utils import weight_norm, spectral_norm
import torch.nn as nn
import torchaudio
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio

from src.utils.io_utils import ROOT_PATH

warnings.filterwarnings("ignore", category=UserWarning)




@hydra.main(version_base=None, config_path="src/configs", config_name="check")
def main(config):
    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device
    pretrained_path = str('D:\HiFi-SS\checkpoint-epoch15.pth')
    epoch = pretrained_path.split('.')[0].split('-')[2][5:]
    print(epoch)

    checkpoint = torch.load(pretrained_path, device)

    save_path = ROOT_PATH / "data" / "saved" / config.trainer.save_path / epoch
    save_path.mkdir(exist_ok=True, parents=True)

    dataloaders, batch_transforms, _ = get_dataloaders(config, device)


    generator = instantiate(config.model).to(device)
    
    if checkpoint.get("generator_state_dict") is not None:
        generator.load_state_dict(checkpoint["generator_state_dict"])
    else:
        generator.load_state_dict(checkpoint)

    weights = generator.waveunet_conv_pre.weight
    channel_importance = torch.norm(weights, p=2, dim=(0, 2))
    print(weights.shape)
    print(channel_importance)
    
    for i, d in enumerate(dataloaders["train"]):
        item_path = save_path / f"{i}"
        item_path.mkdir(exist_ok=True, parents=True)

        torchaudio.save(item_path / "gt_mix.wav", d["mix_audio"][0], sample_rate=16000)
        torchaudio.save(item_path / "gt_1.wav", d["audios"][0, :1, :], sample_rate=16000)
        torchaudio.save(item_path / "gt_2.wav", d["audios"][0, 1:, :], sample_rate=16000)

        hifi_path = item_path / "after_hifi"
        hifi_path.mkdir(exist_ok=True, parents=True)
        waveunet_path = item_path / "after_waveunet"
        waveunet_path.mkdir(exist_ok=True, parents=True)

        d = {k: v.to(device) for k, v in d.items() if k != 'sr'}
        with torch.no_grad():
            pred = generator(**d)

        print('GT mix:', d["mix_audio"][0, 0, :].pow(2).mean())

        si_snr = ScaleInvariantSignalNoiseRatio().to(device)

        for channel in range(8):

            print(f'channel: {channel}', pred["after_hifi"][0, channel, :].pow(2).mean())
            print('mix-snr:', si_snr(pred["after_hifi"][0, channel, :], d["mix_audio"][0, 0, :].cpu()))
            torchaudio.save(hifi_path / f"channel_{channel}.wav", pred["after_hifi"][0, channel:channel+1, :], sample_rate=16000)
            torchaudio.save(waveunet_path / f"channel_{channel}.wav", pred["after_waveunet"][0, channel:channel+1, :], sample_rate=16000)

        mask1_path = item_path / "after_mask1"
        mask1_path.mkdir(exist_ok=True, parents=True)
        mask2_path = item_path / "after_mask2"
        mask2_path.mkdir(exist_ok=True, parents=True)

        for channel in range(4):
            torchaudio.save(mask1_path / f"channel_{channel}.wav", pred["after_mask1"][0, channel:channel+1, :], sample_rate=16000)
            torchaudio.save(mask2_path / f"channel_{channel}.wav", pred["after_mask2"][0, channel:channel+1, :], sample_rate=16000)

        print('-------------------------------------------')



if __name__ == "__main__":
    main()
