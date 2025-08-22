#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ReLU神经网络训练样例
ReLU Neural Network Training Example

实现一个使用ReLU激活函数的多层神经网络，演示从Sigmoid到ReLU的改进
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import time

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ReLUNeuralNetwork:
    """
    使用ReLU激活函数的多层神经网络
    
    网络结构：
    - 输入层 → 隐藏层1 (ReLU) → 隐藏层2 (ReLU) → 输出层
    - 支持任意隐藏层大小和层数
    """
    
    def __init__(self, layer_sizes: List[int], learning_rate: float = 0.01):
        """
        初始化神经网络
        
        Args:
            layer_sizes: 每层的神经元数量，例如 [2, 64, 32, 1]
            learning_rate: 学习率
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.num_layers = len(layer_sizes)
        
        # 初始化权重和偏置
        self.weights = []
        self.biases = []
        
        # He初始化，适合ReLU激活函数
        for i in range(self.num_layers - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
        
        # 记录训练过程
        self.loss_history = []
        self.accuracy_history = []
        
        print(f"神经网络初始化完成:")
        print(f"网络结构: {' → '.join(map(str, layer_sizes))}")
        print(f"学习率: {learning_rate}")
        print(f"权重初始化: He初始化 (适合ReLU)")
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU激活函数: f(x) = max(0, x)"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """ReLU激活函数的导数"""
        return (x > 0).astype(float)
    
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid激活函数（用于输出层）"""
        # 防止数值溢出
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def forward_pass(self, X: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """
        前向传播
        
        Args:
            X: 输入数据 (batch_size, input_features)
            
        Returns:
            output: 网络输出
            activations: 每层的激活值
            z_values: 每层的线性变换结果（激活前）
        """
        activations = [X]  # 第0层是输入
        z_values = []
        
        current_input = X
        
        # 通过所有隐藏层
        for i in range(self.num_layers - 2):
            # 线性变换: z = X * W + b
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            z_values.append(z)
            
            # ReLU激活
            a = self.relu(z)
            activations.append(a)
            current_input = a
        
        # 输出层（使用sigmoid进行二分类）
        z_output = np.dot(current_input, self.weights[-1]) + self.biases[-1]
        z_values.append(z_output)
        output = self.sigmoid(z_output)
        activations.append(output)
        
        return output, activations, z_values
    
    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """计算二元交叉熵损失"""
        # 防止log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
    
    def backward_pass(self, X: np.ndarray, y: np.ndarray, 
                     activations: List[np.ndarray], z_values: List[np.ndarray]) -> None:
        """
        反向传播算法
        
        Args:
            X: 输入数据
            y: 真实标签
            activations: 前向传播的激活值
            z_values: 前向传播的线性变换结果
        """
        m = X.shape[0]  # batch size
        
        # 计算梯度
        dW = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]
        
        # 输出层误差（sigmoid + 交叉熵损失的梯度）
        delta = activations[-1] - y
        
        # 从输出层向输入层反向传播
        for i in range(self.num_layers - 2, -1, -1):
            # 计算当前层的权重和偏置梯度
            dW[i] = np.dot(activations[i].T, delta) / m
            db[i] = np.mean(delta, axis=0, keepdims=True)
            
            # 如果不是输入层，计算下一层的误差
            if i > 0:
                # 误差反向传播
                delta = np.dot(delta, self.weights[i].T)
                # 应用ReLU导数
                delta = delta * self.relu_derivative(z_values[i-1])
        
        # 更新权重和偏置
        for i in range(self.num_layers - 1):
            self.weights[i] -= self.learning_rate * dW[i]
            self.biases[i] -= self.learning_rate * db[i]
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000, 
              batch_size: int = 32, verbose: bool = True) -> None:
        """
        训练神经网络
        
        Args:
            X: 训练数据
            y: 训练标签
            epochs: 训练轮数
            batch_size: 批次大小
            verbose: 是否打印训练进度
        """
        n_samples = X.shape[0]
        n_batches = n_samples // batch_size
        
        print(f"\n开始训练神经网络...")
        print(f"训练样本: {n_samples}, 批次大小: {batch_size}, 训练轮数: {epochs}")
        print("=" * 60)
        
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_accuracy = 0
            
            # 随机打乱数据
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # 批次训练
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # 前向传播
                y_pred, activations, z_values = self.forward_pass(X_batch)
                
                # 计算损失
                batch_loss = self.compute_loss(y_pred, y_batch)
                epoch_loss += batch_loss
                
                # 计算准确率
                predictions = (y_pred > 0.5).astype(int)
                batch_accuracy = np.mean(predictions == y_batch)
                epoch_accuracy += batch_accuracy
                
                # 反向传播
                self.backward_pass(X_batch, y_batch, activations, z_values)
            
            # 记录平均损失和准确率
            avg_loss = epoch_loss / n_batches
            avg_accuracy = epoch_accuracy / n_batches
            
            self.loss_history.append(avg_loss)
            self.accuracy_history.append(avg_accuracy)
            
            # 打印训练进度
            if verbose and (epoch + 1) % 100 == 0:
                elapsed_time = time.time() - start_time
                print(f"Epoch {epoch+1:4d}/{epochs}: "
                      f"Loss = {avg_loss:.6f}, "
                      f"Accuracy = {avg_accuracy:.4f}, "
                      f"Time = {elapsed_time:.2f}s")
        
        total_time = time.time() - start_time
        print("=" * 60)
        print(f"训练完成！总用时: {total_time:.2f}秒")
        print(f"最终损失: {self.loss_history[-1]:.6f}")
        print(f"最终准确率: {self.accuracy_history[-1]:.4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        output, _, _ = self.forward_pass(X)
        return output
    
    def predict_classes(self, X: np.ndarray) -> np.ndarray:
        """预测类别"""
        predictions = self.predict(X)
        return (predictions > 0.5).astype(int)
    
    def plot_training_history(self):
        """可视化训练过程"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 损失曲线
        ax1.plot(self.loss_history, 'b-', linewidth=2)
        ax1.set_title('训练损失变化')
        ax1.set_xlabel('训练轮数')
        ax1.set_ylabel('损失值')
        ax1.grid(True, alpha=0.3)
        
        # 准确率曲线
        ax2.plot(self.accuracy_history, 'r-', linewidth=2)
        ax2.set_title('训练准确率变化')
        ax2.set_xlabel('训练轮数')
        ax2.set_ylabel('准确率')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def generate_classification_data(n_samples: int = 10000, random_seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成二分类数据集
    
    Args:
        n_samples: 样本数量
        random_seed: 随机种子
        
    Returns:
        X: 特征数据 (n_samples, 2)
        y: 标签数据 (n_samples, 1)
    """
    np.random.seed(random_seed)
    
    # 生成两个类别的数据
    n_class0 = n_samples // 2
    n_class1 = n_samples - n_class0
    
    # 类别0：以(-1, -1)为中心的高斯分布
    X_class0 = np.random.multivariate_normal([-1, -1], [[0.8, 0.2], [0.2, 0.8]], n_class0)
    y_class0 = np.zeros((n_class0, 1))
    
    # 类别1：以(1, 1)为中心的高斯分布
    X_class1 = np.random.multivariate_normal([1, 1], [[0.8, -0.2], [-0.2, 0.8]], n_class1)
    y_class1 = np.ones((n_class1, 1))
    
    # 合并数据
    X = np.vstack([X_class0, X_class1])
    y = np.vstack([y_class0, y_class1])
    
    # 随机打乱
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]
    
    print(f"生成分类数据集: {n_samples} 个样本, 2个特征, 2个类别")
    print(f"类别0: {n_class0} 个样本")
    print(f"类别1: {n_class1} 个样本")
    
    return X, y


def visualize_data_and_results(X: np.ndarray, y: np.ndarray, model: ReLUNeuralNetwork):
    """可视化数据和模型结果"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('ReLU神经网络训练结果', fontsize=16)
    
    # 1. 原始数据分布
    ax = axes[0, 0]
    class0_mask = y.flatten() == 0
    class1_mask = y.flatten() == 1
    
    ax.scatter(X[class0_mask, 0], X[class0_mask, 1], c='red', alpha=0.6, label='类别 0')
    ax.scatter(X[class1_mask, 0], X[class1_mask, 1], c='blue', alpha=0.6, label='类别 1')
    ax.set_title('原始数据分布')
    ax.set_xlabel('特征 1')
    ax.set_ylabel('特征 2')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 决策边界
    ax = axes[0, 1]
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(mesh_points)
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, levels=50, alpha=0.8, cmap='RdYlBu')
    ax.scatter(X[class0_mask, 0], X[class0_mask, 1], c='red', alpha=0.8, label='类别 0')
    ax.scatter(X[class1_mask, 0], X[class1_mask, 1], c='blue', alpha=0.8, label='类别 1')
    ax.set_title('决策边界')
    ax.set_xlabel('特征 1')
    ax.set_ylabel('特征 2')
    ax.legend()
    
    # 3. 训练损失
    ax = axes[1, 0]
    ax.plot(model.loss_history, 'b-', linewidth=2)
    ax.set_title('训练损失变化')
    ax.set_xlabel('训练轮数')
    ax.set_ylabel('损失值')
    ax.grid(True, alpha=0.3)
    
    # 4. 训练准确率
    ax = axes[1, 1]
    ax.plot(model.accuracy_history, 'r-', linewidth=2)
    ax.set_title('训练准确率变化')
    ax.set_xlabel('训练轮数')
    ax.set_ylabel('准确率')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def compare_activation_functions():
    """比较Sigmoid和ReLU激活函数"""
    x = np.linspace(-5, 5, 100)
    
    # Sigmoid函数
    sigmoid = 1 / (1 + np.exp(-x))
    sigmoid_derivative = sigmoid * (1 - sigmoid)
    
    # ReLU函数
    relu = np.maximum(0, x)
    relu_derivative = (x > 0).astype(float)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('激活函数比较: Sigmoid vs ReLU', fontsize=16)
    
    # Sigmoid函数
    axes[0, 0].plot(x, sigmoid, 'b-', linewidth=2, label='Sigmoid')
    axes[0, 0].set_title('Sigmoid激活函数')
    axes[0, 0].set_xlabel('输入值')
    axes[0, 0].set_ylabel('输出值')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # ReLU函数
    axes[0, 1].plot(x, relu, 'r-', linewidth=2, label='ReLU')
    axes[0, 1].set_title('ReLU激活函数')
    axes[0, 1].set_xlabel('输入值')
    axes[0, 1].set_ylabel('输出值')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Sigmoid导数
    axes[1, 0].plot(x, sigmoid_derivative, 'b--', linewidth=2, label='Sigmoid导数')
    axes[1, 0].set_title('Sigmoid导数')
    axes[1, 0].set_xlabel('输入值')
    axes[1, 0].set_ylabel('导数值')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # ReLU导数
    axes[1, 1].plot(x, relu_derivative, 'r--', linewidth=2, label='ReLU导数')
    axes[1, 1].set_title('ReLU导数')
    axes[1, 1].set_xlabel('输入值')
    axes[1, 1].set_ylabel('导数值')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()
    
    print("激活函数特点比较:")
    print("Sigmoid:")
    print("  优点: 输出范围(0,1), 平滑可导")
    print("  缺点: 梯度消失问题, 计算复杂")
    print("ReLU:")
    print("  优点: 计算简单, 缓解梯度消失, 稀疏激活")
    print("  缺点: 神经元死亡问题")


def main():
    """主函数：完整的ReLU神经网络训练演示"""
    print("🎯 ReLU神经网络训练演示")
    print("=" * 60)
    
    # 1. 比较激活函数
    print("\n📊 第一步：比较Sigmoid和ReLU激活函数")
    compare_activation_functions()
    
    # 2. 生成训练数据
    print("\n📊 第二步：生成训练数据")
    X_train, y_train = generate_classification_data(n_samples=10000, random_seed=42)
    
    # 3. 创建和训练模型
    print("\n🚀 第三步：创建和训练ReLU神经网络")
    # 网络结构: 2个输入 → 64个隐藏单元 → 32个隐藏单元 → 1个输出
    model = ReLUNeuralNetwork(layer_sizes=[2, 64, 32, 1], learning_rate=0.01)
    
    # 训练模型
    model.train(X_train, y_train, epochs=500, batch_size=64, verbose=True)
    
    # 4. 评估模型
    print("\n📋 第四步：模型评估")
    train_predictions = model.predict_classes(X_train)
    train_accuracy = np.mean(train_predictions == y_train)
    print(f"训练集准确率: {train_accuracy:.4f}")
    
    # 5. 可视化结果
    print("\n📈 第五步：可视化训练结果")
    visualize_data_and_results(X_train, y_train, model)
    
    # 6. 网络结构信息
    print("\n🏗️ 网络结构详情:")
    total_params = 0
    for i, (w, b) in enumerate(zip(model.weights, model.biases)):
        layer_params = w.size + b.size
        total_params += layer_params
        print(f"  第{i+1}层: 权重{w.shape}, 偏置{b.shape}, 参数数量: {layer_params}")
    
    print(f"  总参数数量: {total_params}")
    
    print("\n✅ ReLU神经网络训练演示完成！")
    print("\n💡 关键改进点:")
    print("   1. 使用ReLU激活函数缓解梯度消失问题")
    print("   2. He初始化适合ReLU网络")
    print("   3. 批次训练提高训练效率")
    print("   4. 深层网络可以学习复杂的非线性关系")


if __name__ == "__main__":
    main()
