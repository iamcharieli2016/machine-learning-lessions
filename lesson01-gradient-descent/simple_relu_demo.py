#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版ReLU神经网络演示
Simple ReLU Neural Network Demo

快速演示ReLU神经网络的基本概念和训练过程
"""

import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def relu(x):
    """ReLU激活函数"""
    return np.maximum(0, x)

def relu_derivative(x):
    """ReLU导数"""
    return (x > 0).astype(float)

def sigmoid(x):
    """Sigmoid激活函数"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def simple_relu_network_demo():
    """简单的ReLU网络演示"""
    print("🎯 简单ReLU神经网络演示")
    print("=" * 50)
    
    # 生成简单的二分类数据
    np.random.seed(42)
    n_samples = 1000
    
    # 类别0: 圆形区域内的点
    angles = np.random.uniform(0, 2*np.pi, n_samples//2)
    radius = np.random.uniform(0, 1, n_samples//2)
    X_class0 = np.column_stack([
        radius * np.cos(angles),
        radius * np.sin(angles)
    ])
    y_class0 = np.zeros((n_samples//2, 1))
    
    # 类别1: 圆环区域的点
    angles = np.random.uniform(0, 2*np.pi, n_samples//2)
    radius = np.random.uniform(1.5, 2.5, n_samples//2)
    X_class1 = np.column_stack([
        radius * np.cos(angles),
        radius * np.sin(angles)
    ])
    y_class1 = np.ones((n_samples//2, 1))
    
    # 合并数据
    X = np.vstack([X_class0, X_class1])
    y = np.vstack([y_class0, y_class1])
    
    # 随机打乱
    indices = np.random.permutation(n_samples)
    X, y = X[indices], y[indices]
    
    print(f"生成数据: {n_samples} 个样本，2个特征，2个类别")
    
    # 网络参数
    input_size = 2
    hidden_size = 10
    output_size = 1
    learning_rate = 0.1
    epochs = 1000
    
    # 初始化权重（He初始化）
    W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
    b2 = np.zeros((1, output_size))
    
    print(f"网络结构: {input_size} → {hidden_size} (ReLU) → {output_size} (Sigmoid)")
    print(f"学习率: {learning_rate}, 训练轮数: {epochs}")
    
    # 训练过程记录
    losses = []
    accuracies = []
    
    print("\n开始训练...")
    print("-" * 40)
    
    for epoch in range(epochs):
        # 前向传播
        # 隐藏层
        z1 = np.dot(X, W1) + b1
        a1 = relu(z1)
        
        # 输出层
        z2 = np.dot(a1, W2) + b2
        a2 = sigmoid(z2)
        
        # 计算损失（二元交叉熵）
        loss = -np.mean(y * np.log(np.clip(a2, 1e-15, 1-1e-15)) + 
                       (1 - y) * np.log(np.clip(1 - a2, 1e-15, 1-1e-15)))
        losses.append(loss)
        
        # 计算准确率
        predictions = (a2 > 0.5).astype(int)
        accuracy = np.mean(predictions == y)
        accuracies.append(accuracy)
        
        # 反向传播
        # 输出层梯度
        dz2 = a2 - y
        dW2 = np.dot(a1.T, dz2) / n_samples
        db2 = np.mean(dz2, axis=0, keepdims=True)
        
        # 隐藏层梯度
        da1 = np.dot(dz2, W2.T)
        dz1 = da1 * relu_derivative(z1)
        dW1 = np.dot(X.T, dz1) / n_samples
        db1 = np.mean(dz1, axis=0, keepdims=True)
        
        # 更新参数
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        
        # 打印进度
        if (epoch + 1) % 200 == 0:
            print(f"Epoch {epoch+1:4d}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")
    
    print("-" * 40)
    print(f"训练完成！最终损失: {losses[-1]:.4f}, 最终准确率: {accuracies[-1]:.4f}")
    
    # 可视化结果
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('简单ReLU神经网络训练结果', fontsize=16)
    
    # 1. 原始数据
    ax = axes[0, 0]
    class0_mask = y.flatten() == 0
    class1_mask = y.flatten() == 1
    ax.scatter(X[class0_mask, 0], X[class0_mask, 1], c='red', alpha=0.6, label='类别 0', s=20)
    ax.scatter(X[class1_mask, 0], X[class1_mask, 1], c='blue', alpha=0.6, label='类别 1', s=20)
    ax.set_title('原始数据分布')
    ax.set_xlabel('特征 1')
    ax.set_ylabel('特征 2')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 决策边界
    ax = axes[0, 1]
    h = 0.1
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # 预测网格点
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    z1_mesh = relu(np.dot(mesh_points, W1) + b1)
    z2_mesh = sigmoid(np.dot(z1_mesh, W2) + b2)
    Z = z2_mesh.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, levels=50, alpha=0.8, cmap='RdYlBu')
    ax.scatter(X[class0_mask, 0], X[class0_mask, 1], c='red', alpha=0.8, label='类别 0', s=20)
    ax.scatter(X[class1_mask, 0], X[class1_mask, 1], c='blue', alpha=0.8, label='类别 1', s=20)
    ax.set_title('学习到的决策边界')
    ax.set_xlabel('特征 1')
    ax.set_ylabel('特征 2')
    ax.legend()
    
    # 3. 损失曲线
    ax = axes[1, 0]
    ax.plot(losses, 'b-', linewidth=2)
    ax.set_title('训练损失变化')
    ax.set_xlabel('训练轮数')
    ax.set_ylabel('损失值')
    ax.grid(True, alpha=0.3)
    
    # 4. 准确率曲线
    ax = axes[1, 1]
    ax.plot(accuracies, 'r-', linewidth=2)
    ax.set_title('训练准确率变化')
    ax.set_xlabel('训练轮数')
    ax.set_ylabel('准确率')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return W1, b1, W2, b2

def compare_activations_simple():
    """简单比较激活函数"""
    print("\n🔍 激活函数比较")
    print("-" * 30)
    
    x = np.linspace(-3, 3, 100)
    
    # 计算函数值
    sigmoid_vals = sigmoid(x)
    relu_vals = relu(x)
    
    # 计算导数
    sigmoid_derivs = sigmoid_vals * (1 - sigmoid_vals)
    relu_derivs = relu_derivative(x)
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 函数比较
    axes[0].plot(x, sigmoid_vals, 'b-', linewidth=2, label='Sigmoid')
    axes[0].plot(x, relu_vals, 'r-', linewidth=2, label='ReLU')
    axes[0].set_title('激活函数比较')
    axes[0].set_xlabel('输入值')
    axes[0].set_ylabel('输出值')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 导数比较
    axes[1].plot(x, sigmoid_derivs, 'b--', linewidth=2, label='Sigmoid导数')
    axes[1].plot(x, relu_derivs, 'r--', linewidth=2, label='ReLU导数')
    axes[1].set_title('导数比较')
    axes[1].set_xlabel('输入值')
    axes[1].set_ylabel('导数值')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("ReLU相比Sigmoid的优势:")
    print("1. 计算简单：max(0, x)")
    print("2. 缓解梯度消失：正区间导数为1")
    print("3. 稀疏激活：负值被抑制为0")
    print("4. 更好的收敛性能")

if __name__ == "__main__":
    # 运行激活函数比较
    compare_activations_simple()
    
    # 运行简单演示
    simple_relu_network_demo()
    
    print("\n✅ 简单ReLU神经网络演示完成！")
    print("\n💡 要点总结:")
    print("   • ReLU激活函数: f(x) = max(0, x)")
    print("   • 解决了梯度消失问题")
    print("   • 计算效率高，训练速度快")
    print("   • 适合深层神经网络")
