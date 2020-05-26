











class EarlyStopping():
    """
    Early stopping callback to speedup convergence and prevent overfitting.
    Implementation inspired from https://github.com/PyTorchLightning/PyTorch-Lightning/blob/master/pytorch_lightning/callbacks/early_stopping.py#L19-L141
    """

    def __init__(self, monitor='val_loss', min_delta=0.1, patience=3):
        """
        Parameters:
            monitor(str): quantity to be monitored. Possible values: 'val_loss' or 'val_acc'.
            min_delta(float): minimum change in the monitored quantity to qualify as an improvement,
                       i.e. an absolute change of less than `min_delta`, will count as no
                       improvement. Default: ``0``.
            patience(int): number of validation epochs with no improvement after which training will be stopped.
        """

        self.wait = 0
        self.patience = patience

