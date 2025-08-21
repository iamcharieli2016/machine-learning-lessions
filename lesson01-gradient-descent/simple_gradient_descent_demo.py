#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
梯度下降法简化演示
Simple Gradient Descent Demo

这是一个最简化的梯度下降实现，用于理解核心概念
"""

import numpy as np
import matplotlib.pyplot as plt

def simple_gradient_descent_demo():
    """
    最简单的梯度下降演示
    目标：找到函数 f(x) = (x-3)² 的最小值点
    """
    print("🎯 简单梯度下降演示：寻找函数 f(x) = (x-3)² 的最小值")
    print("=" * 50)
    
    # 定义目标函数和其导数
    def f(x):
        return (x - 3) ** 2
    
    def df_dx(x):  # 导数
        return 2 * (x - 3)
    
    # 初始化参数
    x = 0.0  # 起始点
    learning_rate = 0.1
    epochs = 20
    
    # 记录优化过程
    x_history = [x]
    f_history = [f(x)]
    
    print(f"起始点: x = {x:.3f}, f(x) = {f(x):.3f}")
    print("开始梯度下降...")
    print("-" * 30)
    
    # 梯度下降迭代
    for epoch in range(epochs):
        # 计算梯度
        gradient = df_dx(x)
        
        # 更新参数
        x = x - learning_rate * gradient
        
        # 记录过程
        x_history.append(x)
        f_history.append(f(x))
        
        print(f"第{epoch+1:2d}轮: x = {x:.3f}, f(x) = {f(x):.3f}, 梯度 = {gradient:.3f}")
    
    print("-" * 30)
    print(f"最终结果: x = {x:.3f}, f(x) = {f(x):.6f}")
    print(f"理论最优解: x = 3.000, f(x) = 0.000000")
    
    # 可视化优化过程
    plt.figure(figsize=(12, 5))
    
    # 左图：函数曲线和优化路径
    plt.subplot(1, 2, 1)
    x_range = np.linspace(-1, 6, 100)
    y_range = f(x_range)
    plt.plot(x_range, y_range, 'b-', linewidth=2, label='f(x) = (x-3)²')
    
    # 绘制优化路径
    for i in range(len(x_history)-1):
        plt.arrow(x_history[i], f_history[i], 
                 x_history[i+1] - x_history[i], 
                 f_history[i+1] - f_history[i],
                 head_width=0.1, head_length=0.1, fc='red', ec='red', alpha=0.7)
    
    plt.scatter(x_history, f_history, color='red', s=50, zorder=5)
    plt.scatter([3], [0], color='green', s=100, marker='*', label='全局最优解')
    plt.title('梯度下降优化路径')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 右图：参数和函数值的变化
    plt.subplot(1, 2, 2)
    epochs_range = range(len(x_history))
    plt.plot(epochs_range, x_history, 'r-o', label='参数 x', markersize=4)
    plt.plot(epochs_range, f_history, 'b-s', label='函数值 f(x)', markersize=4)
    plt.title('优化过程中参数和函数值的变化')
    plt.xlabel('迭代次数')
    plt.ylabel('数值')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def linear_regression_step_by_step():
    """
    逐步演示线性回归的梯度下降过程
    """
    print("\n🎯 线性回归梯度下降逐步演示")
    print("=" * 50)
    
    # 生成简单数据
    np.random.seed(42)
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 6, 8, 10])  # 完美的线性关系 y = 2x
    
    print(f"训练数据: ")
    for i in range(len(x)):
        print(f"  x[{i}] = {x[i]}, y[{i}] = {y[i]}")
    
    # 初始化参数
    w = 0.0  # 权重
    b = 0.0  # 偏置
    learning_rate = 0.1
    
    print(f"\n初始参数: w = {w}, b = {b}")
    print(f"学习率: {learning_rate}")
    print("\n开始训练...")
    print("-" * 60)
    
    # 训练几步，详细展示每一步
    for step in range(5):
        print(f"\n第 {step+1} 步:")
        
        # 1. 前向传播
        y_pred = w * x + b
        print(f"  1. 前向传播: y_pred = {w:.3f} * x + {b:.3f}")
        print(f"     预测值: {y_pred}")
        
        # 2. 计算损失
        loss = np.mean((y_pred - y) ** 2) / 2
        print(f"  2. 计算损失: MSE = {loss:.6f}")
        
        # 3. 计算梯度
        n = len(x)
        dw = np.mean((y_pred - y) * x)
        db = np.mean(y_pred - y)
        print(f"  3. 计算梯度: dw = {dw:.6f}, db = {db:.6f}")
        
        # 4. 更新参数
        w_new = w - learning_rate * dw
        b_new = b - learning_rate * db
        print(f"  4. 更新参数: w = {w:.3f} - {learning_rate} * {dw:.6f} = {w_new:.6f}")
        print(f"                b = {b:.3f} - {learning_rate} * {db:.6f} = {b_new:.6f}")
        
        w, b = w_new, b_new
        print(f"  更新后参数: w = {w:.6f}, b = {b:.6f}")
    
    print(f"\n最终参数: w = {w:.6f}, b = {b:.6f}")
    print(f"理论最优解: w = 2.000000, b = 0.000000")

if __name__ == "__main__":
    # 运行简单演示
    simple_gradient_descent_demo()
    
    # 运行线性回归逐步演示
    linear_regression_step_by_step()
