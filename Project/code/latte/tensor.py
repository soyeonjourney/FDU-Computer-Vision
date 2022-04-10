import numpy as np
from typing import List, Tuple


#########################################################################################
#                                        Tensor                                         #
#########################################################################################


class Tensor:
    def __init__(
        self,
        data: np.ndarray,
        grad_fn: 'Function' = None,
        requires_grad: bool = False,
        is_bias: bool = False,  # Notation for bias, not official
    ) -> None:
        self.data = data
        self.grad = None
        self.grad_fn = grad_fn
        self.requires_grad = requires_grad
        self.is_bias = is_bias

    def __repr__(self) -> str:
        if self.data is None:
            return 'Tensor()'
        else:
            return f'Tensor(data={self.data}, grad={self.grad}, \
                grad_fn={self.grad_fn}, requires_grad={self.requires_grad})'

    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def T(self) -> 'Tensor':
        def _backward():
            self.grad = out.grad.T

        out = Tensor(self.data.T, grad_fn=_backward, requires_grad=self.requires_grad)
        return out

    def __len__(self):
        return len(self.data)

    def __add__(self, tensor: 'Tensor') -> 'Tensor':
        add_fn = Add()
        return add_fn(self, tensor)

    def __neg__(self) -> 'Tensor':
        neg_fn = Neg()
        return neg_fn(self)

    def __sub__(self, tensor: 'Tensor') -> 'Tensor':
        sub_fn = Sub()
        return sub_fn(self, tensor)

    def dot(self, tensor: 'Tensor') -> 'Tensor':
        dot_fn = Dot()
        return dot_fn(self, tensor)

    def reshape(self, *shape):
        reshape_fn = Reshape()
        return reshape_fn(self, *shape)

    def backward(self) -> None:
        # Build computational graph
        graph = []
        visited = set()

        def build_graph(node: 'Tensor'):
            if node.requires_grad is True and node not in visited:
                visited.add(node)

                # Post-order traversal
                if node.grad_fn is not None:
                    for prev_node in node.grad_fn.prev:
                        build_graph(prev_node)

                graph.append(node)

        build_graph(self)

        # Backpropagate gradients
        self.grad = np.array([1.0]).reshape(1, 1)  # Create implicit gradient
        for node in reversed(graph):
            if node.grad_fn is not None:
                node.grad_fn.backward(node.grad)


class Function:
    def __init__(self) -> None:
        self.prev = []

    def __call__(self, *inputs: 'Tensor') -> None:
        return self.forward(*inputs)

    def forward(self, *inputs: 'Tensor') -> None:
        raise NotImplementedError

    def backward(self, *inputs: np.ndarray) -> None:
        raise NotImplementedError

    def save_backward_node(self, tensors: List['Tensor']) -> None:
        self.prev = tensors


class Add(Function):
    def __repr__(self) -> str:
        return 'Function(Add)'

    def forward(self, a: 'Tensor', b: 'Tensor') -> 'Tensor':
        self.save_backward_node([a, b])
        return Tensor(
            a.data + b.data,
            grad_fn=self,
            requires_grad=(a.requires_grad or b.requires_grad),
        )

    def backward(self, out: np.ndarray) -> None:
        a, b = self.prev
        a_grad, b_grad = out, out
        a.grad = sum_to_shape(a_grad, a.shape)
        b.grad = sum_to_shape(b_grad, b.shape)


class Neg(Function):
    def __repr__(self) -> str:
        return 'Function(Neg)'

    def forward(self, a: 'Tensor') -> 'Tensor':
        self.save_backward_node([a])
        return Tensor(-a.data, grad_fn=self, requires_grad=a.requires_grad)

    def backward(self, out: np.ndarray) -> None:
        a = self.prev[0]
        a.grad = -out


class Sub(Function):
    def __repr__(self) -> str:
        return 'Function(Sub)'

    def forward(self, a: 'Tensor', b: 'Tensor') -> 'Tensor':
        self.save_backward_node([a, b])
        return Tensor(
            a.data - b.data,
            grad_fn=self,
            requires_grad=(a.requires_grad or b.requires_grad),
        )

    def backward(self, out: np.ndarray) -> None:
        a, b = self.prev
        a_grad, b_grad = out, -out
        a.grad = sum_to_shape(a_grad, a.shape)
        b.grad = sum_to_shape(b_grad, b.shape)


class Dot(Function):
    def __repr__(self) -> str:
        return 'Function(Dot)'

    def forward(self, a: 'Tensor', b: 'Tensor') -> 'Tensor':
        self.save_backward_node([a, b])
        return Tensor(
            a.data.dot(b.data),
            grad_fn=self,
            requires_grad=(a.requires_grad or b.requires_grad),
        )

    def backward(self, out: np.ndarray) -> None:
        a, b = self.prev
        a.grad = out.dot(b.T.data)
        b.grad = a.T.data.dot(out)


class Reshape(Function):
    def __repr__(self) -> str:
        return 'Function(Reshape)'

    def forward(self, a: 'Tensor', *shape: Tuple[int, ...]) -> 'Tensor':
        self.save_backward_node([a])

        def _backward():
            self.grad = a.grad.reshape(*shape)

        return Tensor(
            a.data.reshape(*shape), grad_fn=_backward, requires_grad=a.requires_grad
        )

    def backward(self, out: np.ndarray) -> None:
        a = self.prev[0]
        a.grad = out.reshape(a.shape)


#########################################################################################
#                                        Utility                                        #
#########################################################################################


# Avoid bias.grad broadcast unexpectedly
def sum_to_shape(array: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """
    Sums the elements of an array to the given shape.
    """
    dim_diff = array.ndim - len(shape)
    if dim_diff < 0:
        raise ValueError(f'Shape {shape} is smaller than array {array}')

    else:
        sum_axis = tuple(
            [
                dim_idx + dim_diff
                for dim_idx, dim_val in enumerate(shape)
                if dim_val == 1
            ]
        )
        sqz_axis = tuple(range(dim_diff))
        output = np.sum(array, axis=sum_axis + sqz_axis, keepdims=True)

        if dim_diff > 0:
            output = output.squeeze(axis=sqz_axis)

        return output


#########################################################################################
#                                         Save                                          #
#########################################################################################


def save(param_list: List['Tensor'], filename: str) -> None:
    """
    Saves the parameters of a list of tensors to a file.
    """
    with open(filename, 'wb') as f:
        for param in param_list:
            np.save(f, param.data)
