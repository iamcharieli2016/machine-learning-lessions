#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
李宏毅机器学习2021 - 第一讲：梯度下降法训练线性回归模型
Gradient Descent for Linear Regression

本文件实现了梯度下降算法的核心概念：
1. 定义损失函数 (Loss Function)
2. 计算梯度 (Gradient Calculation)
3. 参数更新 (Parameter Update)
4. 模型训练和可视化
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

class LinearRegressionGD:
    """
    使用梯度下降法实现的线性回归模型
    
    模型形式: y = w * x + b
    损失函数: MSE = (1/2n) * Σ(y_pred - y_true)²
    """
    
    def __init__(self, learning_rate: float = 0.01):
        """
        初始化模型参数
        
        Args:
            learning_rate: 学习率，控制参数更新的步长
        """
        self.learning_rate = learning_rate
        self.w = 0.0  # 权重参数
        self.b = 0.0  # 偏置参数
        self.loss_history = []  # 记录损失函数值的变化
        self.w_history = []     # 记录权重参数的变化
        self.b_history = []     # 记录偏置参数的变化
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播：计算预测值
        
        Args:
            x: 输入特征
            
        Returns:
            预测值 y_pred = w * x + b
        """
        return self.w * x + self.b
    
    def compute_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        计算均方误差损失函数
        
        Args:
            x: 输入特征
            y: 真实标签
            
        Returns:
            损失函数值
        """
        y_pred = self.forward(x)
        mse_loss = np.mean((y_pred - y) ** 2) / 2
        return mse_loss
    
    def compute_gradients(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """
        计算损失函数关于参数w和b的梯度
        
        梯度计算公式：
        ∂L/∂w = (1/n) * Σ(y_pred - y_true) * x
        ∂L/∂b = (1/n) * Σ(y_pred - y_true)
        
        Args:
            x: 输入特征
            y: 真实标签
            
        Returns:
            (dw, db): 权重和偏置的梯度
        """
        n = len(x)
        y_pred = self.forward(x)
        error = y_pred - y
        
        # 计算梯度
        dw = np.mean(error * x)
        db = np.mean(error)
        
        return dw, db
    
    def update_parameters(self, dw: float, db: float):
        """
        使用梯度下降法更新参数
        
        更新公式：
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        Args:
            dw: 权重的梯度
            db: 偏置的梯度
        """
        self.w -= self.learning_rate * dw
        self.b -= self.learning_rate * db
    
    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int = 1000, verbose: bool = True):
        """
        训练模型
        
        Args:
            x: 训练数据的输入特征
            y: 训练数据的真实标签
            epochs: 训练轮数
            verbose: 是否打印训练过程
        """
        print(f"开始训练模型，学习率: {self.learning_rate}, 训练轮数: {epochs}")
        print("=" * 50)
        
        for epoch in range(epochs):
            # 1. 计算当前损失
            current_loss = self.compute_loss(x, y)
            
            # 2. 计算梯度
            dw, db = self.compute_gradients(x, y)
            
            # 3. 更新参数
            self.update_parameters(dw, db)
            
            # 4. 记录训练过程
            self.loss_history.append(current_loss)
            self.w_history.append(self.w)
            self.b_history.append(self.b)
            
            # 5. 打印训练进度
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1:4d}: Loss = {current_loss:.6f}, w = {self.w:.4f}, b = {self.b:.4f}")
        
        print("=" * 50)
        print(f"训练完成！最终参数: w = {self.w:.4f}, b = {self.b:.4f}")
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        使用训练好的模型进行预测
        
        Args:
            x: 输入特征
            
        Returns:
            预测结果
        """
        return self.forward(x)
    
    def plot_training_process(self):
        """
        可视化训练过程
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('梯度下降训练过程可视化', fontsize=16)
        
        # 1. 损失函数变化
        axes[0, 0].plot(self.loss_history, 'b-', linewidth=2)
        axes[0, 0].set_title('损失函数变化')
        axes[0, 0].set_xlabel('训练轮数')
        axes[0, 0].set_ylabel('损失值')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 权重参数变化
        axes[0, 1].plot(self.w_history, 'r-', linewidth=2)
        axes[0, 1].set_title('权重参数w的变化')
        axes[0, 1].set_xlabel('训练轮数')
        axes[0, 1].set_ylabel('权重值')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 偏置参数变化
        axes[1, 0].plot(self.b_history, 'g-', linewidth=2)
        axes[1, 0].set_title('偏置参数b的变化')
        axes[1, 0].set_xlabel('训练轮数')
        axes[1, 0].set_ylabel('偏置值')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 参数轨迹图
        axes[1, 1].plot(self.w_history, self.b_history, 'purple', linewidth=2, alpha=0.7)
        axes[1, 1].scatter(self.w_history[0], self.b_history[0], color='red', s=100, label='起始点')
        axes[1, 1].scatter(self.w_history[-1], self.b_history[-1], color='green', s=100, label='终点')
        axes[1, 1].set_title('参数空间中的梯度下降轨迹')
        axes[1, 1].set_xlabel('权重w')
        axes[1, 1].set_ylabel('偏置b')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_regression_result(self, x: np.ndarray, y: np.ndarray):
        """
        可视化回归结果
        
        Args:
            x: 输入特征
            y: 真实标签
        """
        plt.figure(figsize=(10, 6))
        
        # 绘制原始数据点
        plt.scatter(x, y, color='blue', alpha=0.6, s=50, label='原始数据')
        
        # 绘制回归线
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = self.predict(x_line)
        plt.plot(x_line, y_line, 'r-', linewidth=2, label=f'回归线: y = {self.w:.2f}x + {self.b:.2f}')
        
        plt.title('线性回归结果', fontsize=14)
        plt.xlabel('输入特征 x')
        plt.ylabel('输出标签 y')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


def generate_sample_data(n_samples: int = 100, noise_std: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成用于训练的样本数据
    
    Args:
        n_samples: 样本数量
        noise_std: 噪声标准差
        
    Returns:
        (x, y): 输入特征和对应的标签
    """
    # 设置随机种子以保证结果可重复
    np.random.seed(42)
    
    # 生成输入特征
    x = np.random.uniform(-2, 2, n_samples)
    
    # 真实的线性关系: y = 2.5x + 1.0 + noise
    true_w = 2.5
    true_b = 1.0
    noise = np.random.normal(0, noise_std, n_samples)
    y = true_w * x + true_b + noise
    
    print(f"生成数据: 样本数量={n_samples}, 真实参数: w={true_w}, b={true_b}")
    return x, y


def compare_learning_rates(x: np.ndarray, y: np.ndarray):
    """
    比较不同学习率对训练过程的影响
    
    Args:
        x: 输入特征
        y: 真实标签
    """
    learning_rates = [0.001, 0.01, 0.1, 0.5]
    colors = ['blue', 'red', 'green', 'orange']
    
    plt.figure(figsize=(12, 8))
    
    for i, lr in enumerate(learning_rates):
        model = LinearRegressionGD(learning_rate=lr)
        model.fit(x, y, epochs=200, verbose=False)
        
        plt.subplot(2, 2, i+1)
        plt.plot(model.loss_history, color=colors[i], linewidth=2)
        plt.title(f'学习率 = {lr}')
        plt.xlabel('训练轮数')
        plt.ylabel('损失值')
        plt.grid(True, alpha=0.3)
        
        # 添加最终参数信息
        final_loss = model.loss_history[-1]
        plt.text(0.7, 0.8, f'最终损失: {final_loss:.4f}\nw: {model.w:.3f}, b: {model.b:.3f}', 
                transform=plt.gca().transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat"))
    
    plt.suptitle('不同学习率对训练过程的影响', fontsize=16)
    plt.tight_layout()
    plt.show()


def main():
    """
    主函数：演示梯度下降法训练线性回归模型
    """
    print("🎯 李宏毅机器学习2021 - 第一讲：梯度下降法实践")
    print("=" * 60)
    
    # 1. 生成训练数据
    print("\n📊 第一步：生成训练数据")
    x_train, y_train = generate_sample_data(n_samples=100, noise_std=0.3)
    
    # 2. 创建和训练模型
    print("\n🚀 第二步：创建模型并开始训练")
    model = LinearRegressionGD(learning_rate=0.01)
    model.fit(x_train, y_train, epochs=1000, verbose=True)
    
    # 3. 可视化训练过程
    print("\n📈 第三步：可视化训练过程")
    model.plot_training_process()
    
    # 4. 可视化回归结果
    print("\n📊 第四步：可视化回归结果")
    model.plot_regression_result(x_train, y_train)
    
    # 5. 模型评估
    print("\n📋 第五步：模型评估")
    final_loss = model.compute_loss(x_train, y_train)
    print(f"最终训练损失: {final_loss:.6f}")
    
    # 测试预测
    test_x = np.array([0.5, 1.0, 1.5])
    test_pred = model.predict(test_x)
    print(f"测试预测:")
    for i, (x_val, pred) in enumerate(zip(test_x, test_pred)):
        print(f"  x={x_val:.1f} -> 预测值={pred:.3f}")
    
    # 6. 比较不同学习率
    print("\n🔍 第六步：比较不同学习率的效果")
    compare_learning_rates(x_train, y_train)
    
    print("\n✅ 梯度下降法训练完成！")
    print("\n💡 关键概念总结:")
    print("   1. 损失函数：衡量模型预测与真实值的差异")
    print("   2. 梯度：指示参数更新的方向")
    print("   3. 学习率：控制参数更新的步长")
    print("   4. 迭代优化：通过多轮训练逐步优化参数")


if __name__ == "__main__":
    main()
