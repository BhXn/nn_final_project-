import numpy as np
import gzip
import struct
import os

def create_dummy_mnist_images(filename, num_images=1000, height=28, width=28):
    """
    创建虚拟MNIST图像文件，格式完全符合原始MNIST格式
    """
    os.makedirs("data", exist_ok=True)
    
    with gzip.open(filename, 'wb') as f:
        # 写入文件头 (按照MNIST格式)
        magic_number = 2051  # MNIST图像文件的magic number
        f.write(struct.pack('>I', magic_number))  # magic number (4 bytes)
        f.write(struct.pack('>I', num_images))    # number of images (4 bytes)
        f.write(struct.pack('>I', height))        # height (4 bytes)
        f.write(struct.pack('>I', width))         # width (4 bytes)
        
        # 生成随机图像数据 (0-255的像素值)
        for i in range(num_images):
            # 生成类似手写数字的简单模式
            image = np.random.randint(0, 256, (height, width), dtype=np.uint8)
            
            # 添加一些结构化模式，使其更像真实数据
            center_x, center_y = width//2, height//2
            
            # 随机选择一个"数字类型"来生成不同模式
            digit_type = i % 10
            
            if digit_type == 0:  # 类似数字0的环形
                for y in range(height):
                    for x in range(width):
                        dist = ((x-center_x)**2 + (y-center_y)**2)**0.5
                        if 8 < dist < 12:
                            image[y, x] = min(255, image[y, x] + 100)
            elif digit_type == 1:  # 类似数字1的竖线
                for y in range(height):
                    if 5 < y < 23:
                        image[y, center_x] = min(255, image[y, center_x] + 100)
                        image[y, center_x+1] = min(255, image[y, center_x+1] + 80)
            # 可以继续添加其他数字的模式...
            
            f.write(image.tobytes())
    
    print(f"Created dummy MNIST images: {filename} ({num_images} images)")

def create_dummy_mnist_labels(filename, num_labels=1000):
    """
    创建虚拟MNIST标签文件，格式完全符合原始MNIST格式
    """
    with gzip.open(filename, 'wb') as f:
        # 写入文件头
        magic_number = 2049  # MNIST标签文件的magic number
        f.write(struct.pack('>I', magic_number))  # magic number (4 bytes)
        f.write(struct.pack('>I', num_labels))    # number of labels (4 bytes)
        
        # 生成随机标签 (0-9)
        labels = np.random.randint(0, 10, num_labels, dtype=np.uint8)
        f.write(labels.tobytes())
    
    print(f"Created dummy MNIST labels: {filename} ({num_labels} labels)")

def create_dummy_cifar10_data():
    """
    创建虚拟CIFAR-10数据（如果需要的话）
    注意：这里假设您有parse_cifar10函数，我们创建对应格式的数据
    """
    os.makedirs("data", exist_ok=True)
    
    # CIFAR-10: 32x32x3的彩色图像
    num_samples = 1000
    
    # 生成虚拟图像数据
    images = np.random.randint(0, 256, (num_samples, 32, 32, 3), dtype=np.uint8)
    labels = np.random.randint(0, 10, num_samples, dtype=np.int8)
    
    # 添加一些结构化模式
    for i in range(num_samples):
        # 随机生成一些简单的几何形状
        if labels[i] % 3 == 0:  # 生成红色矩形
            images[i, 10:22, 10:22, 0] = 255  # 红色通道
            images[i, 10:22, 10:22, 1:] = 50   # 其他通道
        elif labels[i] % 3 == 1:  # 生成绿色圆形
            center = (16, 16)
            for y in range(32):
                for x in range(32):
                    if ((x-center[0])**2 + (y-center[1])**2) < 64:
                        images[i, y, x, 1] = 255  # 绿色通道
                        images[i, y, x, [0, 2]] = 50
        # 可以继续添加更多模式...
    
    # 保存为numpy格式（简化版，您可能需要根据实际的parse_cifar10调整）
    np.save("data/cifar10_train_images.npy", images)
    np.save("data/cifar10_train_labels.npy", labels)
    
    # 生成测试数据
    test_images = np.random.randint(0, 256, (200, 32, 32, 3), dtype=np.uint8)
    test_labels = np.random.randint(0, 10, 200, dtype=np.int8)
    
    np.save("data/cifar10_test_images.npy", test_images)
    np.save("data/cifar10_test_labels.npy", test_labels)
    
    print("Created dummy CIFAR-10 data")

def generate_all_dummy_data():
    """生成所有需要的虚拟数据"""
    print("=== Generating Dummy Datasets ===")
    
    # 生成MNIST训练数据
    create_dummy_mnist_images("data/train-images-idx3-ubyte.gz", num_images=5000)
    create_dummy_mnist_labels("data/train-labels-idx1-ubyte.gz", num_labels=5000)
    
    # 生成MNIST测试数据
    create_dummy_mnist_images("data/t10k-images-idx3-ubyte.gz", num_images=1000)
    create_dummy_mnist_labels("data/t10k-labels-idx1-ubyte.gz", num_labels=1000)
    
    # 生成CIFAR-10数据
    create_dummy_cifar10_data()
    
    print("\n=== All Dummy Data Generated ===")
    print("Files created:")
    print("  - data/train-images-idx3-ubyte.gz (5000 images)")
    print("  - data/train-labels-idx1-ubyte.gz (5000 labels)")
    print("  - data/t10k-images-idx3-ubyte.gz (1000 images)")
    print("  - data/t10k-labels-idx1-ubyte.gz (1000 labels)")
    print("  - data/cifar10_*.npy (CIFAR-10 data)")

if __name__ == "__main__":
    generate_all_dummy_data()