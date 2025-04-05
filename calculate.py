import warnings

import hydra
import torch
from thop import profile
from thop import clever_format
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import Trainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)

@hydra.main(version_base=None, config_path="src/configs", config_name="calculate")
def main(config):
    set_random_seed(config.trainer.seed)

    project_config = OmegaConf.to_container(config)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    model = instantiate(config.model).to(device)
    input = torch.randn(1, 1, 1024 * 16).to(device)
    macs, params = profile(model, inputs=(input, ), verbose=False)
    macs, params = clever_format([macs, params], "%.3f")
    if 'hifi' in model.__dict__:
        print('generator params:', sum(p.numel() for p in model.hifi.parameters()))
    print('total:')
    print(macs, params)


if __name__ == "__main__":
    main()
