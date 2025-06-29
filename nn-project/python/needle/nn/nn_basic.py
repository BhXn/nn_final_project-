"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
from functools import reduce
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # TODO
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(
          init.kaiming_uniform(
            fan_in=in_features, 
            fan_out=out_features, 
            device=device, 
            dtype=dtype,
          )
        )
        self.bias = Parameter(
          init.kaiming_uniform(
            fan_in=out_features,
            fan_out=1,
            device=device, 
            dtype=dtype,
          ).reshape((1, out_features))
        ) if bias else None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        # TODO
        ### BEGIN YOUR SOLUTION
        y = X @ self.weight # (n, out_features)

        if self.bias:
          y += ops.broadcast_to(self.bias, (*X.shape[:-1], self.out_features))
        
        return y
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        # TODO
        ### BEGIN YOUR SOLUTION
        flattened_dim = reduce(lambda a, b: a * b, X.shape[1:])
        return X.reshape((X.shape[0], flattened_dim))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        # TODO
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules: List["Module"]):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        # TODO
        ### BEGIN YOUR SOLUTION
        for module in self.modules: 
            x = module(x)
        return x 
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        # logits: (batch, num_classes)
        # y: (batch,) with class indices
        ### BEGIN YOUR SOLUTION
        # Compute log-sum-exp for numerical stability
        logsumexp = ops.logsumexp(logits, axes=(1,))
        # Gather the logits corresponding to the correct class
        batch_indices = ops.arange(logits.shape[0], device=logits.device)
        correct_class_logits = logits[batch_indices, y]
        # Negative log likelihood
        loss = logsumexp - correct_class_logits
        return loss.mean()
        ### END YOUR SOLUTION
        
class CrossEntrophyLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        # logits: (batch, num_classes)
        # y: (batch,) with class indices
        ### BEGIN YOUR SOLUTION
        logsumexp = ops.logsumexp(logits, axes=(1,))
        batch_indices = ops.arange(logits.shape[0], device=logits.device)
        correct_class_logits = logits[batch_indices, y]
        loss = logsumexp - correct_class_logits
        return loss.mean()
        ### END YOUR SOLUTION
        
class BinaryCrossEntrophyLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        # logits: (batch,) or (batch, 1)
        # y: (batch,) or (batch, 1), values in {0, 1}
        ### BEGIN YOUR SOLUTION
        # Sigmoid for logits
        probs = ops.sigmoid(logits)
        # BCE loss: -y*log(p) - (1-y)*log(1-p)
        loss = - (y * ops.log(probs + 1e-8) + (1 - y) * ops.log(1 - probs + 1e-8))
        return loss.mean()
        ### END YOUR SOLUTION
        
class MSELoss(Module):
    def forward(self, input: Tensor, target: Tensor):
        # TODO
        ### BEGIN YOUR SOLUTION
        return ((input - target) ** 2).mean()
        ### END YOUR SOLUTION
class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        # Learnable parameters
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        # Running statistics (not parameters)
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            mean = x.mean(axes=(0,))
            var = ((x - mean) ** 2).mean(axes=(0,))
            # Update running stats
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.data
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.data
        else:
            mean = Tensor(self.running_mean, device=x.device, dtype=x.dtype)
            var = Tensor(self.running_var, device=x.device, dtype=x.dtype)
        x_hat = (x - mean) / (var + self.eps) ** 0.5
        return self.weight * x_hat + self.bias


class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def init(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().init()
        self.dim = dim
        self.eps = eps
        # Learnable parameters for scaling and shifting
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        # x shape: (batch_size, dim)
        mean = x.mean(axes=(1,), keepdims=True)  # shape: (batch_size, 1)
        var = ((x - mean) ** 2).mean(axes=(1,), keepdims=True)  # shape: (batch_size, 1)
        x_hat = (x - mean) / (var + self.eps) ** 0.5
        # Broadcast weight and bias to (batch_size, dim)
        return self.weight * x_hat + self.bias

class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if not self.training or self.p == 0.0:
            return x
        mask = (np.random.rand(*x.shape) > self.p).astype(x.dtype)
        mask = Tensor(mask, device=x.device, dtype=x.dtype)
        return x * mask / (1.0 - self.p)
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION

