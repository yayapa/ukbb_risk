import torch
import wandb


class EarlyStopping:
    """
    Early stops the training if validation accuracy doesn't improve after a given patience.
    """

    def __init__(self, path, patience=3, verbose=False, logger=None, save_results=True):
        """
        :param patience: how long to wait after last time validation accuracy improved.
        :param verbose: If True, prints a message for each validation accuracy improvement.
        :param path: Path for the checkpoint to be saved to
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.early_stop = False
        self.val_param_best = 0.0  # can be accuracy, loss, etc.
        self.path = path
        self.logger = logger
        self.save_results = save_results

    def __call__(self, val_param, model, epoch):
        if val_param <= self.val_param_best:
            self.counter += 1
            self.logger.info(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.save_results:
                wandb.log({"early_stop_counter": self.counter}, step=epoch)
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if self.save_results:
                self.save_checkpoint(val_param, model)
                self.counter = 0

    def save_checkpoint(self, val_param, model):
        """
        Saves model when validation parameter improves.
        :param val_param: validation parameter (accuracy, loss, etc.)
        :param model: model
        """
        if self.verbose:
            self.logger.info(
                f"Validation accuracy increased ({self.val_param_best:.6f} --> {val_param:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path + "checkpoint.pth")
        self.val_param_best = val_param
