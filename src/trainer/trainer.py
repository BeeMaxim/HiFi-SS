from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.g_optimizer.zero_grad()
            self.d_optimizer.zero_grad()

        clean_audio_hat = self.generator(**batch)

        # Discriminator step
        self.d_optimizer.zero_grad()
        discriminator_estimations = self.discriminator(clean_audio_predicted=clean_audio_hat["clean_audio_predicted"].detach(), **batch)
        # discriminator_estimations = self.discriminator(**batch)
        batch.update(discriminator_estimations)

        discriminator_loss = self.discriminator_criterion(**batch)
        batch.update(discriminator_loss)
        # print('DISCRIMINATOR LOSS:', discriminator_loss["discriminator_loss"])

        if self.is_train:
            batch["discriminator_loss"].backward()
            self._clip_grad_norm()
            self.d_optimizer.step()
            if self.d_lr_scheduler is not None:
                self.d_lr_scheduler.step()


        # Generator step
        self.g_optimizer.zero_grad()

        batch.update(clean_audio_hat)

        discriminator_estimations = self.discriminator(**batch)
        batch.update(discriminator_estimations)

        generator_loss  = self.generator_criterion(**batch)
        batch.update(generator_loss)
        # print('GENERATOR LOSS:', generator_loss["generator_loss"])

        if self.is_train:
            batch["generator_loss"].backward()  # sum of all losses is always called loss
            self._clip_grad_norm()
            self.g_optimizer.step()
            if self.g_lr_scheduler is not None:
                self.g_lr_scheduler.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            # Log Stuff
            pass
        else:
            # Log Stuff
            pass
