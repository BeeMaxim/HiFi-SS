import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import BSSTrainer, Inferencer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

from src.model.disriminators import BSSDiscriminator

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="baseline_bss")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.trainer.seed)

    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device
    print(torch.cuda.get_device_name())

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms, speaker_ids = get_dataloaders(config, device)

    generator = instantiate(config.model).to(device)
    discriminator = BSSDiscriminator(use_id_channel=False, embedding_count=speaker_ids, channel_count=1).to(device)

    logger.info(generator)
    logger.info(discriminator)

    # get function handles of loss and metrics
    generator_loss_function = instantiate(config.generator_loss_function).to(device)
    discriminator_loss_function = instantiate(config.discriminator_loss_function).to(device)
    metrics = instantiate(config.metrics)

    # build optimizer, learning rate scheduler
    generator_params = filter(lambda p: p.requires_grad, generator.parameters())
    discriminator_params = filter(lambda p: p.requires_grad, discriminator.parameters())

    epoch_len = config.trainer.get("epoch_len")
    if epoch_len is None:
        config.lr_scheduler['step_size'] = len(dataloaders['train'])

    # print(trainable_params)
    g_optimizer = instantiate(config.generator_optimizer, params=generator_params)
    g_lr_scheduler = instantiate(config.lr_scheduler, optimizer=g_optimizer)

    d_optimizer = instantiate(config.discriminator_optimizer, params=discriminator_params)
    d_lr_scheduler = instantiate(config.lr_scheduler, optimizer=d_optimizer)

    # epoch_len = number of iterations for iteration-based training
    # epoch_len = None or len(dataloader) for epoch-based training

    trainer = BSSTrainer(
        generator=generator,
        discriminator=discriminator,
        generator_criterion=generator_loss_function,
        discriminator_criterion=discriminator_loss_function,
        metrics=metrics,
        g_optimizer=g_optimizer,
        d_optimizer=d_optimizer,
        g_lr_scheduler=g_lr_scheduler,
        d_lr_scheduler=d_lr_scheduler,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", True),
    )
    trainer.train()
    
    if config.trainer.inference:
        inferencer = Inferencer(
            generator=generator,
            config=config,
            device=device,
            dataloaders=dataloaders,
            batch_transforms=batch_transforms,
            save_path=None,
            metrics=metrics,
            skip_model_load=True,
        )

        logs = inferencer.run_inference()

        for part in logs.keys():
            for key, value in logs[part].items():
                full_key = part + "_" + key
                print(f"    {full_key:15s}: {value}")


if __name__ == "__main__":
    main()
