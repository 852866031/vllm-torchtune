"""Schedulers wrapped so torchtune's `_setup_lr_scheduler` can build them.

torchtune passes `num_training_steps=<int>` into the scheduler ctor for any
scheduler it instantiates. Stock torch schedulers like ExponentialLR don't
accept it, so we shim them here.
"""
from torch.optim.lr_scheduler import ExponentialLR


def exponential_lr(optimizer, gamma: float, num_training_steps: int = None,
                   last_epoch: int = -1, **_ignored):
    return ExponentialLR(optimizer, gamma=gamma, last_epoch=last_epoch)
