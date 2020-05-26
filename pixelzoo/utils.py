import torch
import logging

log = logging.getLogger('logger')
log.setLevel(logging.INFO)


class EarlyStopping():
    """
    Early stopping callback to speedup convergence and prevent overfitting
    Implementation inspired from https://github.com/PyTorchLightning/PyTorch-Lightning/blob/master/pytorch_lightning/callbacks/early_st    opping.py#L19-L141
    """

    mode_dict = {
        'min': torch.lt,
        'max': torch.gt,
    }

    def __init__(self,
                 monitor='loss',
                 min_delta=0.01,
                 patience=3,
                 mode='auto'):
        """
        Args:
            monitor(str): quantity to be monitored. Possible values: 'loss' or 'acc'.
            min_delta(float): minimum change in the monitored quantity to qualify as an improvement,
                        i.e. an absolute change of less than `min_delta`, will count as no
                        improvement. Default: ``0``.
            patience(int): number of validation epochs with no improvement after which training will be stopped.
            mode: one of {auto, min, max}. In `min` mode, training will stop when the quantity
                monitored has stopped decreasing; in `max` mode it will stop when the quantity monitored has 
                stopped increasing; in `auto` mode, the direction is automatically inferred from the name of the
                monitored quantity.

        """

        self.wait = 0
        self.patience = patience
        self.stopped_epoch = 0
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode

        if mode not in self.mode_dict:
            log.info(
                f'EarlyStopping mode {mode} is unknown, fallback to auto mode.'
            )
            self.mode = 'auto'

        if self.mode == 'auto':
            if self.monitor == 'acc':
                self.mode = 'max'
            else:
                self.mode = 'min'
            log.info(
                f'EarlyStopping mode set to {self.mode} for monitoring {self.monitor}.'
            )

        self.min_delta *= 1 if self.monitor_op == torch.gt else -1
        self.best = torch_inf if self.monitor_op == torch.lt else -torch_inf

    @property
    def monitor_op(self):
        return self.mode_dict[self.mode]

    def early_stop(self, current_epoch, current_val):
        """
        Args:
            current_epoch(int): the current epoch value
            current_val(float): the current loss or accuracy depending on the monitor value.
        Returns:
            stop_training(bool): True if the training should be stopped early; otherwise, false.
        """
        log.info(f'Epoch {self.stopped_epoch + 1:05d}: early stopping')
        stop_training = False

        if not isinstance(current_val, torch.Tensor):
            current_val = torch.tensor(current_val)

        if self.monitor_op(current_val - self.min_delta, self.best):
            self.best = current_val
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = current_epoch
                stop_training = True

        return stop_training
