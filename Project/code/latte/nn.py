import numpy as np
from typing import List
from latte.tensor import Tensor, Function
import latte.functional as F


#########################################################################################
#                                         Layer                                         #
#########################################################################################


class Module:
    def __init__(self) -> None:
        self._modules = []
        self._params = []
        self.training = True  # Training mode or test mode

    def __repr__(self) -> str:
        return 'Module()'

    def __call__(self, input: 'Tensor') -> None:
        return self.forward(input)

    def forward(self, input: 'Tensor') -> None:
        # Overwritten by subclass
        raise NotImplementedError

    def parameters(self) -> List['Tensor']:
        return self._params

    def __setattr__(self, __name: str, __value) -> None:
        if isinstance(__value, Module):
            self._modules.append(__value)
            self._params.extend(__value.parameters())
        # object.__setattr__(self, __name, __value)
        super().__setattr__(__name, __value)

    def eval(self) -> None:
        self.training = False

    def train(self) -> None:
        self.training = True

    def load(self, filename: str) -> None:
        with open(filename, 'rb') as f:
            for param in self.parameters():
                param.data = np.load(f)


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.weight = Tensor(
            np.random.randn(in_features, out_features) * 0.01, requires_grad=True
        )
        if bias:
            self.bias = Tensor(
                np.zeros((1, out_features)), requires_grad=True, is_bias=True
            )
            self._params.extend([self.weight, self.bias])
        else:
            self.bias = None
            self._params.append(self.weight)

    def __repr__(self) -> str:
        return f'Linear({self.weight.shape[0]}, {self.weight.shape[1]}, bias={self.bias is not None})'

    def forward(self, input: 'Tensor') -> 'Tensor':
        if self.bias is not None:
            return input.dot(self.weight) + self.bias  # x @ W + b
        else:
            return input.dot(self.weight)  # x @ W


#########################################################################################
#                                      Activation                                       #
#########################################################################################


class ReLU(Module):
    def __repr__(self) -> str:
        return 'Activation(ReLU)'

    def forward(self, input: 'Tensor') -> 'Tensor':
        return F.relu(input)


class Sigmoid(Module):
    def __repr__(self) -> str:
        return 'Activation(Sigmoid)'

    def forward(self, input: 'Tensor') -> 'Tensor':
        return F.sigmoid(input)


#########################################################################################
#                                     Loss Function                                     #
#########################################################################################


class MSELoss(Function):
    def __repr__(self) -> str:
        return 'LossFunction(MSELoss)'

    def forward(self, input: 'Tensor', target: 'Tensor') -> 'Tensor':
        self.save_backward_node([input, target])
        input_data = input.data
        target_data = target.data
        loss = (input_data - target_data) ** 2 / 2

        return Tensor(np.mean(loss), grad_fn=self, requires_grad=input.requires_grad)

    def backward(self, out: np.ndarray) -> None:
        a, b = self.prev
        m = a.shape[0]
        input_data = a.data
        target_data = b.data
        a.grad = (input_data - target_data) / m


class BCELoss(Function):
    def __repr__(self) -> str:
        return 'LossFunction(BCELoss)'

    def forward(self, input: 'Tensor', target: 'Tensor') -> 'Tensor':
        self.save_backward_node([input, target])
        input_data = input.data
        target_data = target.data
        loss = -(
            target_data * np.log(input_data)
            + (1 - target_data) * np.log(1 - input_data)
        )

        return Tensor(np.mean(loss), grad_fn=self, requires_grad=input.requires_grad)

    def backward(self, out: np.ndarray) -> None:
        a, b = self.prev
        m = a.size
        input_data = a.data
        target_data = b.data
        a.grad = (-target_data / input_data + (1 - target_data) / (1 - input_data)) / m


class CrossEntropyLoss(Function):
    """Combines log_softmax and nll_loss in a single function."""

    def __repr__(self) -> str:
        return 'LossFunction(CrossEntropyLoss)'

    def forward(self, input: 'Tensor', target: 'Tensor') -> 'Tensor':
        self.save_backward_node([input, target])
        m = input.shape[0]
        input_data = input.data
        target_data = target.data

        # softmax = exp(x[target]) / sum(exp(x[i]), axis=1)
        neg_log_softmax = -input_data[np.arange(m), target_data] + np.log(
            np.sum(np.exp(input_data), axis=1)
        )

        return Tensor(
            np.mean(neg_log_softmax), grad_fn=self, requires_grad=input.requires_grad
        )

    def backward(self, out: np.ndarray) -> None:
        a, b = self.prev
        m = a.shape[0]
        input_data = a.data
        target_data = b.data

        # neg_log_softmax' = softmax - 1 * (i == target)
        softmax = np.exp(input_data) / np.sum(np.exp(input_data), axis=1).reshape(-1, 1)
        softmax[np.arange(m), target_data] -= 1
        a.grad = softmax / m
