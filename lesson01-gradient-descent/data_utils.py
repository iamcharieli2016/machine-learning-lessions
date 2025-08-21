#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据生成和处理工具
Data Generation and Processing Utilities

提供各种用于机器学习实验的数据生成函数
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def generate_linear_data(n_samples: int = 100, 
                        true_w: float = 2.0, 
                        true_b: float = 1.0,
                        noise_std: float = 0.1,
                        x_range: Tuple[float, float] = (-2, 2),
                        random_seed: Optional[int] = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成线性回归数据
    
    Args:
        n_samples: 样本数量
        true_w: 真实的权重参数
        true_b: 真实的偏置参数
        noise_std: 噪声标准差
        x_range: 输入特征的范围
        random_seed: 随机种子
        
    Returns:
        (x, y): 输入特征和对应的标签
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # 生成输入特征
    x = np.random.uniform(x_range[0], x_range[1], n_samples)
    
    # 生成噪声
    noise = np.random.normal(0, noise_std, n_samples)
    
    # 生成标签
    y = true_w * x + true_b + noise
    
    print(f"生成线性数据: 样本数={n_samples}, 真实参数 w={true_w}, b={true_b}, 噪声std={noise_std}")
    
    return x, y

def generate_polynomial_data(n_samples: int = 100,
                           degree: int = 2,
                           coefficients: Optional[list] = None,
                           noise_std: float = 0.1,
                           x_range: Tuple[float, float] = (-2, 2),
                           random_seed: Optional[int] = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成多项式回归数据
    
    Args:
        n_samples: 样本数量
        degree: 多项式次数
        coefficients: 多项式系数，如果为None则随机生成
        noise_std: 噪声标准差
        x_range: 输入特征的范围
        random_seed: 随机种子
        
    Returns:
        (x, y): 输入特征和对应的标签
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # 生成输入特征
    x = np.random.uniform(x_range[0], x_range[1], n_samples)
    
    # 设置多项式系数
    if coefficients is None:
        coefficients = np.random.uniform(-2, 2, degree + 1)
    
    # 计算多项式值
    y = np.zeros(n_samples)
    for i, coef in enumerate(coefficients):
        y += coef * (x ** i)
    
    # 添加噪声
    noise = np.random.normal(0, noise_std, n_samples)
    y += noise
    
    print(f"生成多项式数据: 样本数={n_samples}, 次数={degree}, 系数={coefficients}")
    
    return x, y

def generate_classification_data(n_samples: int = 200,
                               n_features: int = 2,
                               n_classes: int = 2,
                               noise_std: float = 0.1,
                               random_seed: Optional[int] = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成分类数据
    
    Args:
        n_samples: 样本数量
        n_features: 特征数量
        n_classes: 类别数量
        noise_std: 噪声标准差
        random_seed: 随机种子
        
    Returns:
        (X, y): 特征矩阵和类别标签
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # 为每个类别生成中心点
    centers = np.random.uniform(-3, 3, (n_classes, n_features))
    
    # 为每个类别生成样本
    samples_per_class = n_samples // n_classes
    X = []
    y = []
    
    for class_idx in range(n_classes):
        # 在类别中心周围生成样本
        class_samples = np.random.multivariate_normal(
            centers[class_idx], 
            np.eye(n_features) * noise_std, 
            samples_per_class
        )
        X.append(class_samples)
        y.extend([class_idx] * samples_per_class)
    
    X = np.vstack(X)
    y = np.array(y)
    
    # 打乱数据
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    print(f"生成分类数据: 样本数={len(X)}, 特征数={n_features}, 类别数={n_classes}")
    
    return X, y

def visualize_data(x, y, title: str = "数据可视化", save_path: Optional[str] = None):
    """
    可视化一维数据
    
    Args:
        x: 输入特征
        y: 标签
        title: 图表标题
        save_path: 保存路径，如果为None则不保存
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.6, s=50)
    plt.title(title)
    plt.xlabel('输入特征 x')
    plt.ylabel('输出标签 y')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
    
    plt.show()

def visualize_classification_data(X, y, title: str = "分类数据可视化", save_path: Optional[str] = None):
    """
    可视化二维分类数据
    
    Args:
        X: 特征矩阵 (n_samples, 2)
        y: 类别标签
        title: 图表标题
        save_path: 保存路径
    """
    if X.shape[1] != 2:
        print("只支持二维特征的可视化")
        return
    
    plt.figure(figsize=(10, 8))
    
    # 为不同类别使用不同颜色
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    unique_classes = np.unique(y)
    
    for i, class_label in enumerate(unique_classes):
        mask = y == class_label
        plt.scatter(X[mask, 0], X[mask, 1], 
                   c=colors[i % len(colors)], 
                   alpha=0.6, s=50, 
                   label=f'类别 {class_label}')
    
    plt.title(title)
    plt.xlabel('特征 1')
    plt.ylabel('特征 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
    
    plt.show()

def split_data(X, y, train_ratio: float = 0.8, random_seed: Optional[int] = 42):
    """
    分割数据为训练集和测试集
    
    Args:
        X: 特征数据
        y: 标签数据
        train_ratio: 训练集比例
        random_seed: 随机种子
        
    Returns:
        (X_train, X_test, y_train, y_test): 分割后的数据
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    n_samples = len(X)
    n_train = int(n_samples * train_ratio)
    
    # 随机打乱索引
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    if X.ndim == 1:
        X_train, X_test = X[train_indices], X[test_indices]
    else:
        X_train, X_test = X[train_indices], X[test_indices]
    
    y_train, y_test = y[train_indices], y[test_indices]
    
    print(f"数据分割完成: 训练集 {len(X_train)} 样本, 测试集 {len(X_test)} 样本")
    
    return X_train, X_test, y_train, y_test

def save_data(X, y, filename: str, data_dir: str = "data"):
    """
    保存数据到文件
    
    Args:
        X: 特征数据
        y: 标签数据
        filename: 文件名
        data_dir: 数据目录
    """
    # 创建数据目录
    os.makedirs(data_dir, exist_ok=True)
    
    # 保存数据
    filepath = os.path.join(data_dir, filename)
    if X.ndim == 1:
        data = np.column_stack([X, y])
    else:
        data = np.column_stack([X, y])
    
    np.savetxt(filepath, data, delimiter=',', 
               header='features,label' if X.ndim == 1 else f'feature1,feature2,label')
    
    print(f"数据已保存到: {filepath}")

def load_data(filename: str, data_dir: str = "data"):
    """
    从文件加载数据
    
    Args:
        filename: 文件名
        data_dir: 数据目录
        
    Returns:
        (X, y): 特征和标签
    """
    filepath = os.path.join(data_dir, filename)
    data = np.loadtxt(filepath, delimiter=',', skiprows=1)
    
    X = data[:, :-1]
    y = data[:, -1]
    
    if X.shape[1] == 1:
        X = X.flatten()
    
    print(f"数据已从 {filepath} 加载: {len(X)} 样本")
    
    return X, y

def demo_data_generation():
    """
    演示数据生成功能
    """
    print("🎯 数据生成工具演示")
    print("=" * 50)
    
    # 1. 生成线性数据
    print("\n1. 生成线性回归数据")
    x_linear, y_linear = generate_linear_data(n_samples=100, true_w=2.5, true_b=1.0, noise_std=0.3)
    visualize_data(x_linear, y_linear, "线性回归数据")
    
    # 2. 生成多项式数据
    print("\n2. 生成多项式回归数据")
    x_poly, y_poly = generate_polynomial_data(n_samples=100, degree=2, coefficients=[1, -2, 0.5], noise_std=0.2)
    visualize_data(x_poly, y_poly, "多项式回归数据")
    
    # 3. 生成分类数据
    print("\n3. 生成分类数据")
    X_class, y_class = generate_classification_data(n_samples=200, n_classes=3, noise_std=0.5)
    visualize_classification_data(X_class, y_class, "三分类数据")
    
    # 4. 数据分割演示
    print("\n4. 数据分割演示")
    X_train, X_test, y_train, y_test = split_data(x_linear, y_linear, train_ratio=0.8)

if __name__ == "__main__":
    demo_data_generation()
