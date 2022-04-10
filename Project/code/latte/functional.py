import numpy as np
from typing import Tuple
from latte.tensor import Tensor, Function


#########################################################################################
#                                        Function                                       #
#########################################################################################


def relu(input: 'Tensor') -> 'Tensor':
    relu_fn = ReLU()
    return relu_fn(input)


def sigmoid(input: 'Tensor') -> 'Tensor':
    sigmoid_fn = Sigmoid()
    return sigmoid_fn(input)


def dropout(input: 'Tensor', p: float = 0.5, training: bool = True) -> 'Tensor':
    dropout_fn = Dropout(p, training)
    return dropout_fn(input)


#########################################################################################
#                                         Class                                         #
#########################################################################################


class ReLU(Function):
    def __repr__(self) -> str:
        return 'Function(ReLU)'

    def forward(self, x: 'Tensor') -> 'Tensor':
        self.save_backward_node([x])
        out = x.data
        out[out < 0] = 0
        return Tensor(out, grad_fn=self, requires_grad=x.requires_grad)

    def backward(self, out: np.ndarray):
        x = self.prev[0]
        x.data[x.data <= 0] = 0
        x.data[x.data > 0] = 1
        x.grad = x.data * out


class Sigmoid(Function):
    def __repr__(self) -> str:
        return 'Function(Sigmoid)'

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def forward(self, x: 'Tensor') -> 'Tensor':
        self.save_backward_node([x])
        out = self.sigmoid(x.data)
        return Tensor(out, grad_fn=self, requires_grad=x.requires_grad)

    def backward(self, out: np.ndarray) -> None:
        x = self.prev[0]
        d_sigmoid = self.sigmoid(x.data) * (1 - self.sigmoid(x.data))
        x.grad = d_sigmoid * out


class Dropout(Function):
    def __init__(self, p: float = 0.5, training: bool = True) -> None:
        super().__init__()
        self.p = p
        self.training = training
        self.mask = None

    def __repr__(self) -> str:
        return 'Function(Dropout)'

    def forward(self, x: 'Tensor') -> 'Tensor':
        if self.training:
            self.save_backward_node([x])
            out = x.data
            self.mask = np.random.binomial(1, self.p, size=x.shape) / (1 - self.p)
            out = out * self.mask
            return Tensor(out, grad_fn=self, requires_grad=x.requires_grad)

        else:
            return Tensor(x.data, grad_fn=self, requires_grad=x.requires_grad)

    def backward(self, out: np.ndarray) -> None:
        x = self.prev[0]
        x.grad = self.mask * out
