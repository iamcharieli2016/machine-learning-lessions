# 李宏毅机器学习2021课程练习项目

本项目是基于李宏毅教授机器学习2021课程的实践练习代码，旨在通过具体的代码实现来加深对机器学习核心概念的理解。

## 🎯 项目目标

- 通过代码实践理解机器学习的核心概念
- 提供完整的训练样例和可视化
- 逐步实现课程中的各个主题
- 为机器学习初学者提供实用的代码参考

## 📚 课程内容结构

```
machine-learning-lessions/
├── lesson01-gradient-descent/     # 第1讲：梯度下降法
├── lesson02-classification/       # 第2讲：分类问题
├── lesson03-cnn/                 # 第3讲：卷积神经网络
├── lesson04-self-attention/      # 第4讲：自注意力机制
├── lesson05-transformer/         # 第5讲：Transformer
├── lesson06-gan/                 # 第6讲：生成对抗网络
├── lesson07-bert/                # 第7讲：BERT
├── lesson08-anomaly-detection/   # 第8讲：异常检测
├── lesson09-explainable-ai/      # 第9讲：可解释AI
├── lesson10-attack/              # 第10讲：对抗攻击
├── lesson11-domain-adaptation/   # 第11讲：领域适应
├── lesson12-rl/                  # 第12讲：强化学习
├── lesson13-network-compression/ # 第13讲：网络压缩
├── lesson14-life-long-learning/  # 第14讲：终身学习
└── lesson15-meta-learning/       # 第15讲：元学习
```

## 🚀 快速开始

### 环境准备

1. **克隆项目**
   ```bash
   git clone <repository-url>
   cd machine-learning-lessions
   ```

2. **创建虚拟环境（推荐）**
   ```bash
   # 使用conda
   conda create -n ml2021 python=3.8
   conda activate ml2021
   
   # 或使用venv
   python -m venv ml2021
   source ml2021/bin/activate  # macOS/Linux
   # ml2021\Scripts\activate   # Windows
   ```

3. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

### 第一讲：梯度下降法

第一讲是整个课程的基础，重点学习梯度下降算法的核心概念。

#### 🎯 核心概念
- **损失函数（Loss Function）**：衡量模型预测与真实值的差异
- **梯度（Gradient）**：指示参数更新的方向
- **学习率（Learning Rate）**：控制参数更新的步长
- **迭代优化**：通过多轮训练逐步优化参数

#### 📁 文件说明

```
lesson01-gradient-descent/
├── linear_regression_gradient_descent.py  # 完整的线性回归实现
├── simple_gradient_descent_demo.py        # 简化的梯度下降演示
└── data_utils.py                          # 数据生成和处理工具
```

#### 🏃‍♂️ 运行示例

1. **完整的线性回归训练**
   ```bash
   cd lesson01-gradient-descent
   python linear_regression_gradient_descent.py
   ```
   
   这个脚本将会：
   - 生成训练数据
   - 训练线性回归模型
   - 可视化训练过程
   - 比较不同学习率的效果

2. **简化的概念演示**
   ```bash
   python simple_gradient_descent_demo.py
   ```
   
   这个脚本提供：
   - 最简单的梯度下降演示
   - 逐步展示每一轮的计算过程
   - 直观的可视化效果

3. **数据工具演示**
   ```bash
   python data_utils.py
   ```

## 🔍 第一讲详细说明

### 数学原理

**线性回归模型**：
```
y = w * x + b
```

**损失函数（均方误差）**：
```
L = (1/2n) * Σ(y_pred - y_true)²
```

**梯度计算**：
```
∂L/∂w = (1/n) * Σ(y_pred - y_true) * x
∂L/∂b = (1/n) * Σ(y_pred - y_true)
```

**参数更新**：
```
w = w - learning_rate * ∂L/∂w
b = b - learning_rate * ∂L/∂b
```

### 关键特性

- ✅ **完整的训练流程**：从数据生成到模型训练的完整过程
- ✅ **详细的可视化**：训练过程、参数变化、优化轨迹
- ✅ **多种学习率比较**：展示学习率对训练效果的影响
- ✅ **逐步计算展示**：帮助理解每一步的数学计算
- ✅ **中文注释**：详细的中文解释和注释

## 🛠️ 技术栈

- **Python 3.8+**
- **NumPy**：数值计算
- **Matplotlib**：数据可视化
- **Pandas**：数据处理
- **Scikit-learn**：机器学习工具
- **PyTorch**：深度学习框架

## 📖 学习建议

1. **按顺序学习**：建议按课程顺序逐步学习，每个概念都有其依赖关系
2. **动手实践**：运行代码，观察结果，尝试修改参数
3. **理解原理**：不仅要会用，更要理解背后的数学原理
4. **可视化分析**：充分利用可视化来理解算法的行为
5. **参数实验**：尝试不同的参数设置，观察对结果的影响

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 📄 许可证

本项目采用MIT许可证，详情请查看LICENSE文件。

## 🙏 致谢

感谢李宏毅教授提供的优质机器学习课程内容。

---

**开始你的机器学习之旅吧！** 🚀
