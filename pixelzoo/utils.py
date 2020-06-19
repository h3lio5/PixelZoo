import time
from colorama import Fore

import torch
import logging
import numpy as np

log = logging.getLogger('logger')
log.setLevel(logging.INFO)

torch_inf = torch.tensor(np.Inf)


class EarlyStopping():
    """
    Early stopping callback to speedup convergence and prevent overfitting
    Implementation inspired from https://github.com/PyTorchLightning/PyTorch-Lightning/blob/master/pytorch_lightning/callbacks/early_stopping.py#L19-L141
    """

    mode_dict = {
        'min': torch.lt,
        'max': torch.gt,
    }

    def __init__(self,
                 monitor='loss',
                 min_delta=0.0001,
                 patience=5,
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


class ProgressBar(object):
    '''
    custom progress bar
    Example:
        >>> pbar = ProgressBar(n_total=30,desc='training')
        >>> step = 2
        >>> pbar(step=step)
    '''

    def __init__(self, n_total, width=30, desc='Training'):
        self.width = width
        self.n_total = n_total
        self.start_time = time.time()
        self.desc = desc

    def __call__(self, step, info={}):
        now = time.time()
        current = step + 1
        recv_per = current / self.n_total
        bar = Fore.BLUE + f'\r[{self.desc}] {current}/{self.n_total} ' + Fore.GREEN + '[' 
        if recv_per >= 1:
            recv_per = 1
        prog_width = int(self.width * recv_per)
        if prog_width > 0:
            bar += '=' * (prog_width - 1)
            if current < self.n_total:
                bar += ">"
            else:
                bar += '='
        bar += '.' * (self.width - prog_width)
        bar += ']'
        show_bar = f"{bar}"
        time_per_unit = (now - self.start_time) / current
        if current < self.n_total:
            eta = time_per_unit * (self.n_total - current)
            if eta > 3600:
                eta_format = ('%d:%02d:%02d hr' %
                              (eta // 3600, (eta % 3600) // 60, eta % 60))
            elif eta > 60:
                eta_format = '%d:%02d min' % (eta // 60, eta % 60)
            else:
                eta_format = '%d s' % eta
            time_info = f' - ETA: {eta_format}'
        else:
            if time_per_unit >= 1:
                time_info = f' {time_per_unit:.1f} s/step'
            elif time_per_unit >= 1e-3:
                time_info = f' {time_per_unit * 1e3:.1f} ms/step'
            else:
                time_info = f' {time_per_unit * 1e6:.1f} us/step'

        show_bar += time_info
        if len(info) != 0:
            show_info = f'{show_bar} ' + Fore.RED + \
                        "-".join([f' {key}: {value:.4f} bits/dim ' for key, 
                                  value in info.items()]) + Fore.GREEN
            print(show_info, end='')
            # Move to the next line after the last iteration of the training epoch
            # so that the progress bar for the testing loop doesn't replace that of
            # the training loop
            if current == self.n_total:
                print('')
        else:
            print(show_bar, end='')
