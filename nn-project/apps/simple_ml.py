"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl

import needle.nn as nn
from apps.models import *
import time
device = ndl.cuda()





def parse_mnist(image_filesname, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    # TODO
    ### BEGIN YOUR SOLUTION
    # 读取图像文件
    with gzip.open(image_filesname, 'rb') as f:
        # 读取文件头 (16字节)
        magic_num = struct.unpack('>I', f.read(4))[0]  # magic number
        num_images = struct.unpack('>I', f.read(4))[0]  # 图像数量
        num_rows = struct.unpack('>I', f.read(4))[0]    # 行数 (28)
        num_cols = struct.unpack('>I', f.read(4))[0]    # 列数 (28)
        
        # 读取图像数据
        image_data = f.read()
        images = np.frombuffer(image_data, dtype=np.uint8)
        images = images.reshape(num_images, num_rows * num_cols)
        
        # 归一化到 [0, 1] 并转换为 float32
        X = images.astype(np.float32) / 255.0
    
    # 读取标签文件
    with gzip.open(label_filename, 'rb') as f:
        # 读取文件头 (8字节)
        magic_num = struct.unpack('>I', f.read(4))[0]  # magic number
        num_labels = struct.unpack('>I', f.read(4))[0]  # 标签数量
        
        # 读取标签数据
        label_data = f.read()
        y = np.frombuffer(label_data, dtype=np.uint8)
        y = y.astype(np.int8)
    
    return X, y
    ### END YOUR SOLUTION


def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    # TODO
    ### BEGIN YOUR SOLUTION
    # 计算 log_softmax = logits - log_sum_exp(logits)
    log_sum_exp = ndl.ops.logsumexp(Z, axes=(1,))  # shape: (batch_size,)
    log_sum_exp = log_sum_exp.reshape((-1, 1))     # shape: (batch_size, 1)
    log_softmax = Z - log_sum_exp                  # shape: (batch_size, num_classes)
    
    # 计算交叉熵损失: -sum(y_one_hot * log_softmax)
    loss = -ndl.ops.summation(y_one_hot * log_softmax, axes=(1,))  # shape: (batch_size,)
    
    # 返回平均损失
    return ndl.ops.summation(loss) / Z.shape[0]
    ### END YOUR SOLUTION

def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """
    # TODO
    ### BEGIN YOUR SOLUTION
    num_examples = X.shape[0]
    num_classes = W2.shape[1]
    
    # 遍历所有mini-batch
    for i in range(0, num_examples, batch):
        # 获取当前batch的数据
        end_idx = min(i + batch, num_examples)
        X_batch = ndl.Tensor(X[i:end_idx])  # (batch_size, input_dim)
        y_batch = y[i:end_idx]              # (batch_size,)
        
        # 将标签转换为one-hot编码
        y_one_hot = np.zeros((y_batch.shape[0], num_classes))
        y_one_hot[np.arange(y_batch.size), y_batch] = 1
        y_one_hot = ndl.Tensor(y_one_hot)  # (batch_size, num_classes)
        
        # 前向传播: logits = ReLU(X * W1) * W2
        Z1 = X_batch @ W1                    # (batch_size, hidden_dim)
        A1 = ndl.ops.relu(Z1)                # (batch_size, hidden_dim) 
        Z2 = A1 @ W2                         # (batch_size, num_classes)
        
        # 计算损失
        loss = softmax_loss(Z2, y_one_hot)
        
        # 反向传播
        loss.backward()
        
        # 更新权重 (SGD)
        W1 = ndl.Tensor(W1.numpy() - lr * W1.grad.numpy())
        W2 = ndl.Tensor(W2.numpy() - lr * W2.grad.numpy())
        
        # 清零梯度
        W1.grad = None
        W2.grad = None
    
    return W1, W2
    ### END YOUR SOLUTION

def epoch_general_cifar10(dataloader, model, epoch, loss_fn=nn.SoftmaxLoss(), opt=None):
    """
    Iterates over the dataloader. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    # TODO
    ### BEGIN YOUR SOLUTION
    # 设置模型模式
    if opt is not None:
        model.train()  # 训练模式
    else:
        model.eval()   # 评估模式
    
    total_loss = 0.0
    total_accuracy = 0.0
    total_samples = 0
    
    # 遍历数据批次
    for batch_idx, (X, y) in enumerate(dataloader):
        # 将数据转换为Tensor
        X = ndl.Tensor(X, device=device)
        y = ndl.Tensor(y, device=device)
        
        # 前向传播
        logits = model(X)
        
        # 计算损失
        loss = loss_fn(logits, y)
        
        # 计算准确率
        predictions = logits.numpy().argmax(axis=1)
        accuracy = (predictions == y.numpy()).mean()
        
        # 累计统计
        batch_size = X.shape[0]
        total_loss += loss.numpy() * batch_size
        total_accuracy += accuracy * batch_size
        total_samples += batch_size
        
        # 如果有优化器，进行反向传播和参数更新
        if opt is not None:
            # 清零梯度
            opt.reset_grad()
            
            # 反向传播
            loss.backward()
            
            # 更新参数
            opt.step()
    
    # 计算平均损失和准确率
    avg_loss = total_loss / total_samples
    avg_acc = total_accuracy / total_samples
    
    return avg_acc, avg_loss
    ### END YOUR SOLUTION

def train_cifar10(model, dataloader, n_epochs=1, optimizer=ndl.optim.Adam,
          lr=0.001, weight_decay=0.001, loss_fn=nn.SoftmaxLoss()):
    """
    Performs {n_epochs} epochs of training.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    # TODO
    ### BEGIN YOUR SOLUTION
    # 初始化优化器
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 训练多个epoch
    for epoch in range(n_epochs):
        avg_acc, avg_loss = epoch_general_cifar10(dataloader, model, epoch, loss_fn, opt)
        
        # 可选：打印训练进度
        print(f"Epoch {epoch+1}/{n_epochs}: Loss = {avg_loss:.4f}, Accuracy = {avg_acc:.4f}")
    
    # 返回最后一个epoch的结果
    return avg_acc, avg_loss
    ### END YOUR SOLUTION


def evaluate_cifar10(model, dataloader, loss_fn=nn.SoftmaxLoss()):
    """
    Computes the test accuracy and loss of the model.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    # TODO
    ### BEGIN YOUR SOLUTION
    # 调用epoch_general_cifar10，不传入优化器（opt=None）进行评估
    avg_acc, avg_loss = epoch_general_cifar10(dataloader, model, 0, loss_fn, opt=None)
    return avg_acc, avg_loss
    ### END YOUR SOLUTION

### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)



import pickle
import os

def save_model(model, filepath):
    """保存模型参数"""
    # 获取所有参数
    params = {}
    for name, param in model.named_parameters():
        params[name] = param.numpy()
    
    # 保存到文件
    with open(filepath, 'wb') as f:
        pickle.dump(params, f)
    print(f"Model saved to {filepath}")

def load_model(model, filepath):
    """加载模型参数"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file {filepath} not found")
    
    # 从文件加载
    with open(filepath, 'rb') as f:
        params = pickle.load(f)
    
    # 设置参数
    for name, param in model.named_parameters():
        if name in params:
            param.data = ndl.Tensor(params[name])
    
    print(f"Model loaded from {filepath}")