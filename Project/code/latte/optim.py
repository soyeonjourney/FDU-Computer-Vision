import numpy as np
from typing import List
from latte.tensor import Tensor


class Optimizer:
    def __init__(
        self, params: List['Tensor'], lr: float = 1e-3, weight_decay: float = 0.0
    ) -> None:
        self.params = params
        self.lr = lr
        self.weight_decay = weight_decay

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = 0

    def step(self) -> None:
        raise NotImplementedError

    def lr_decay(self, lr_decay: float) -> None:
        # Self-designed interface for learning rate scheduler
        self.lr *= lr_decay


class SGD(Optimizer):
    """SGD with momentum."""

    def __init__(
        self,
        params: List['Tensor'],
        lr: float = 1e-3,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params, lr, weight_decay)
        self.momentum = momentum
        self.v = [np.zeros(param.shape) for param in self.params]

    def step(self) -> None:
        for param, v in zip(self.params, self.v):
            v = self.momentum * v + self.lr * param.grad

            if not param.is_bias:
                param.data = (
                    param.data
                    - v
                    - self.weight_decay
                    * param.data
                    / np.linalg.norm(param.data, ord='fro')
                )
            # Bias is not regularized
            else:
                param.data = param.data - v


class Adam(Optimizer):
    def __init__(
        self,
        params: List['Tensor'],
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(params, lr, weight_decay)
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = [np.zeros(param.shape) for param in self.params]
        self.v = [np.zeros(param.shape) for param in self.params]
        self.t = 0  # number of steps
        self.eps = eps

    def step(self) -> None:
        self.t += 1
        lr_t = self.lr * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
        eps = self.eps * self.t ** 0.5
        for param, m, v in zip(self.params, self.m, self.v):
            m = self.beta1 * m + (1 - self.beta1) * param.grad
            v = self.beta2 * v + (1 - self.beta2) * param.grad ** 2

            if not param.is_bias:
                param.data = (
                    param.data
                    - lr_t * m / (np.sqrt(v) + eps)
                    - self.weight_decay
                    * param.data
                    / np.linalg.norm(param.data, ord='fro')
                )
            # Bias is not regularized
            else:
                param.data = param.data - lr_t * m / (np.sqrt(v) + eps)
