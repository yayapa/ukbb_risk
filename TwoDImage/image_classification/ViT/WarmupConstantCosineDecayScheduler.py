import torch
import math


class WarmupConstantCosineDecayScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.initial_lr = optimizer.param_groups[0]["lr"]

        def lr_lambda(current_epoch):
            if current_epoch < self.warmup_epochs:
                return 1.0  # Warmup phase (constant learning rate at initial_lr)
            else:
                # Cosine decay phase
                decay_epochs = current_epoch - self.warmup_epochs
                total_decay_epochs = self.total_epochs - self.warmup_epochs
                cosine_decay = 0.5 * (
                    1 + math.cos(math.pi * decay_epochs / total_decay_epochs)
                )
                return (
                    cosine_decay * (self.initial_lr - self.min_lr) / self.initial_lr
                    + self.min_lr / self.initial_lr
                )

        super(WarmupConstantCosineDecayScheduler, self).__init__(
            optimizer, lr_lambda, last_epoch
        )
