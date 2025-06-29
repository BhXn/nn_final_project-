"""Scheduler module"""
import needle as ndl
import numpy as np

class Scheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def step(self):
        raise NotImplementedError()

    def get_lr(self):
        return self.optimizer.lr


class StepDecay(Scheduler):
    def __init__(self, optimizer, step_size, gamma=0.1):
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma
        self.last_epoch = 0

    def step(self):
        self.last_epoch += 1
        if self.last_epoch % self.step_size == 0:
            self.optimizer.lr *= self.gamma


class LinearWarmUp(Scheduler):
    def __init__(self, optimizer, warmup_epochs, initial_lr, target_lr):
        super().__init__(optimizer)
        self.warmup_epochs = warmup_epochs
        self.initial_lr = initial_lr
        self.target_lr = target_lr
        self.current_epoch = 0

        # Set the initial learning rate
        self.optimizer.lr = initial_lr

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            lr = self.initial_lr + (self.target_lr - self.initial_lr) * (self.current_epoch / self.warmup_epochs)
            self.optimizer.lr = lr
        else:
            self.optimizer.lr = self.target_lr
        ### END YOUR SOLUTION



class CosineDecayWithWarmRestarts(Scheduler):
    def __init__(self, optimizer, initial_lr=0.001, T_0=10, T_mult=2):
        super().__init__(optimizer)
        self.initial_lr = initial_lr
        self.T_0 = T_0  # Initial number of epochs for the first restart
        self.T_mult = T_mult  # Multiplicative factor to increase T_i after a restart
        self.current_epoch = 0
        self.T_cur = T_0  # Current epoch count within the current cycle

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.current_epoch += 1
        # Compute the number of epochs since the last restart
        if self.current_epoch > self.T_cur:
            self.current_epoch = 1
            self.T_cur *= self.T_mult

        # Cosine annealing formula
        lr = self.initial_lr * 0.5 * (1 + np.cos(np.pi * (self.current_epoch - 1) / self.T_cur))
        self.optimizer.lr = lr
        ### END YOUR SOLUTION
