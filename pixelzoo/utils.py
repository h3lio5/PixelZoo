











class EarlyStopping():
    """
    Early stopping callback to speedup convergence and prevent overfitting
    """

    def __init__(self,monitor):
        """
        """

        self.wait = 0
        self.patience = patience

