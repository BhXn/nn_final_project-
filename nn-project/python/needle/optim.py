"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        # TODO
        ### BEGIN YOUR SOLUTION
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
                
            # 获取梯度
            grad = param.grad.data
            
            # 添加权重衰减 (L2正则化)
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param.data
            
            # 动量更新
            if self.momentum > 0:
                if i not in self.u:
                    self.u[i] = np.zeros_like(param.data)
                self.u[i] = self.momentum * self.u[i] + grad
                grad = self.u[i]
            
            # 参数更新
            param.data = param.data - self.lr * grad
            ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        # 计算所有参数梯度的范数
        total_norm = 0.0
        for p in self.params:
            if p.grad is not None:
                grad = p.grad.data
                total_norm += np.sum(grad ** 2)
        total_norm = np.sqrt(total_norm)

        # 如果超过max_norm，则缩放所有梯度
        if total_norm > max_norm and total_norm > 0:
            scale = max_norm / total_norm
            for p in self.params:
                if p.grad is not None:
                    p.grad.data = p.grad.data * scale


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.u = {}
        self.v = {}

    def step(self):
        # TODO
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            grad = param.grad.data

            # Add weight decay (L2 regularization)
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param.data

            # Initialize moment estimates if not present
            if i not in self.u:
                self.u[i] = np.zeros_like(param.data)
                self.v[i] = np.zeros_like(param.data)

            # Update biased first moment estimate
            self.u[i] = self.beta1 * self.u[i] + (1 - self.beta1) * grad
            # Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)

            # Compute bias-corrected first and second moment estimates
            u_hat = self.u[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # Update parameters
            param.data = param.data - self.lr * u_hat / (np.sqrt(v_hat) + self.eps)
        ### END YOUR SOLUTION
