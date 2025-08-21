# 第一讲：梯度下降法 (Gradient Descent)

## 🎯 学习目标

通过本讲的学习，你将掌握：
- 梯度下降法的基本原理
- 如何实现线性回归的梯度下降训练
- 学习率对训练过程的影响
- 损失函数的计算和可视化
- 参数优化的完整流程

## 📚 核心概念

### 1. 梯度下降法原理

梯度下降是机器学习中最重要的优化算法之一。它的核心思想是：
- 沿着损失函数梯度的**反方向**更新参数
- 梯度指向函数增长最快的方向，反方向就是下降最快的方向
- 通过多次迭代，逐步找到损失函数的最小值点

### 2. 线性回归模型

**模型公式**：`y = w * x + b`
- `w`：权重参数（斜率）
- `b`：偏置参数（截距）
- `x`：输入特征
- `y`：输出预测值

### 3. 损失函数

**均方误差（MSE）**：`L = (1/2n) * Σ(y_pred - y_true)²`
- 衡量预测值与真实值之间的差异
- 值越小表示模型拟合越好

### 4. 梯度计算

**权重梯度**：`∂L/∂w = (1/n) * Σ(y_pred - y_true) * x`

**偏置梯度**：`∂L/∂b = (1/n) * Σ(y_pred - y_true)`

### 5. 参数更新

**更新公式**：
- `w = w - learning_rate * ∂L/∂w`
- `b = b - learning_rate * ∂L/∂b`

## 📁 文件说明

### 1. `linear_regression_gradient_descent.py`
**完整的线性回归梯度下降实现**

主要功能：
- ✅ 完整的`LinearRegressionGD`类实现
- ✅ 前向传播、损失计算、梯度计算、参数更新
- ✅ 训练过程可视化（损失曲线、参数变化、优化轨迹）
- ✅ 回归结果可视化
- ✅ 不同学习率效果比较
- ✅ 详细的中文注释和说明

**核心方法**：
```python
class LinearRegressionGD:
    def forward(self, x)                    # 前向传播
    def compute_loss(self, x, y)            # 计算损失
    def compute_gradients(self, x, y)       # 计算梯度
    def update_parameters(self, dw, db)     # 更新参数
    def fit(self, x, y, epochs)             # 训练模型
    def predict(self, x)                    # 模型预测
```

### 2. `simple_gradient_descent_demo.py`
**简化的概念演示**

主要功能：
- ✅ 最简单的一维函数优化演示
- ✅ 逐步展示线性回归的每一轮计算
- ✅ 直观的可视化效果
- ✅ 适合初学者理解基本概念

**演示内容**：
- 寻找函数 `f(x) = (x-3)²` 的最小值点
- 逐步展示线性回归训练的5个步骤

### 3. `data_utils.py`
**数据生成和处理工具**

主要功能：
- ✅ 生成线性回归数据
- ✅ 生成多项式回归数据
- ✅ 生成分类数据
- ✅ 数据可视化
- ✅ 数据分割和保存

## 🏃‍♂️ 快速开始

### 1. 运行完整示例
```bash
cd lesson01-gradient-descent
python linear_regression_gradient_descent.py
```

**预期输出**：
- 训练过程的详细日志
- 4个可视化图表：
  1. 损失函数变化曲线
  2. 权重参数变化
  3. 偏置参数变化
  4. 参数空间中的优化轨迹
- 回归结果可视化
- 不同学习率的比较

### 2. 运行简化演示
```bash
python simple_gradient_descent_demo.py
```

**预期输出**：
- 一维函数优化的完整过程
- 线性回归逐步计算演示
- 优化路径可视化

### 3. 数据工具演示
```bash
python data_utils.py
```

**预期输出**：
- 线性回归数据生成和可视化
- 多项式回归数据
- 分类数据生成

## 🔧 参数调优实验

### 学习率影响实验

尝试修改学习率，观察对训练效果的影响：

```python
# 在 linear_regression_gradient_descent.py 中修改
model = LinearRegressionGD(learning_rate=0.001)  # 很小的学习率
model = LinearRegressionGD(learning_rate=0.1)    # 中等学习率
model = LinearRegressionGD(learning_rate=1.0)    # 很大的学习率
```

**观察现象**：
- **学习率过小**：收敛很慢，需要更多训练轮数
- **学习率适中**：收敛速度适中，效果最好
- **学习率过大**：可能发散，损失函数不收敛

### 训练轮数实验

```python
model.fit(x_train, y_train, epochs=100)   # 较少轮数
model.fit(x_train, y_train, epochs=1000)  # 中等轮数
model.fit(x_train, y_train, epochs=5000)  # 较多轮数
```

### 数据噪声实验

```python
# 在 data_utils.py 中修改噪声水平
x, y = generate_linear_data(noise_std=0.1)   # 低噪声
x, y = generate_linear_data(noise_std=0.5)   # 中等噪声
x, y = generate_linear_data(noise_std=1.0)   # 高噪声
```

## 🤔 思考题

1. **为什么梯度下降要沿着梯度的反方向更新参数？**

2. **学习率设置得过大或过小会有什么问题？**

3. **如果数据中有很多噪声，对梯度下降训练有什么影响？**

4. **如何判断模型已经收敛？**

5. **线性回归的梯度下降一定能找到全局最优解吗？**

## 💡 扩展练习

1. **实现批量梯度下降（Batch GD）**
   - 每次使用所有数据计算梯度

2. **实现随机梯度下降（SGD）**
   - 每次只使用一个样本计算梯度

3. **实现小批量梯度下降（Mini-batch GD）**
   - 每次使用一小批数据计算梯度

4. **添加动量（Momentum）**
   - 考虑历史梯度信息，加速收敛

5. **实现多元线性回归**
   - 扩展到多个输入特征

## 📖 相关资源

- [李宏毅机器学习2021课程](https://speech.ee.ntu.edu.tw/~hylee/ml/2021-spring.html)
- [梯度下降法详解](https://zh.wikipedia.org/wiki/梯度下降法)
- [线性回归原理](https://zh.wikipedia.org/wiki/线性回归)

---

**掌握梯度下降，开启机器学习之门！** 🚀
