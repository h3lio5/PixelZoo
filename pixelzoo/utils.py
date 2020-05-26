











class EarlyStopping():
    """
    Early stopping callback to speedup convergence and prevent overfitting
    """

    def __init__(self, monitor, patience):
        """
        Parameters:
            monitor(str): 
            patience(int): 
        """

        self.wait = 0
        self.patience = patience

