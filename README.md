# Dog Breed Identification

基于深度学习的狗品种识别项目，使用多种卷积神经网络模型对120个狗品种进行细粒度图像分类。

## 项目简介

本项目是人工智能课程设计的第三个任务，旨在通过深度学习技术实现狗品种的自动识别。项目实现了多种主流深度学习模型，包括VGG、ResNet、ResNet+SE注意力机制以及集成学习方案，并对比了不同模型的性能表现。

## 数据集

- **来源**: [Kaggle Dog Breed Identification Competition](https://www.kaggle.com/c/dog-breed-identification/overview)
- **类别数**: 120个狗品种
- **训练集**: 8,177张图像
- **验证集**: 2,045张图像
- **测试集**: 10,357张图像
- **图像特点**: 尺寸不一，需要统一调整为224×224像素

## 项目结构

```
third/
├── VGG/                          # VGG16模型实现
│   ├── kaggle/                   # Kaggle版本实现
│   └── kaggle_dalao/             # 优化版本
├── ResNet/                       # ResNet18模型实现
│   ├── test0518V1.py             # 从零训练版本
│   └── kaggle_dalao/             # 迁移学习版本
├── ResNet50/                     # ResNet50模型实现
│   ├── luchi_dalao/              # 主要实现
│   └── ending/                   # 最终版本
├── ResNet_Seblock/               # ResNet+SE注意力机制
│   └── fangzhang_ResNet_kaggle_dalao/
├── Enrique_Manuel_dalao/         # 集成学习方案
│   └── dalao_bendi_V1/           # Inception+ResNet50融合
├── data_figure/                  # 数据可视化
│   ├── demo1.py                  # 数据分析脚本
│   └── demo2.py                  # 可视化脚本
└── dog-breed-identification/     # 原始数据集
    ├── train/                    # 训练图像
    ├── test/                     # 测试图像
    └── labels.csv                # 标签文件
```

## 模型实现

### 1. VGG16（迁移学习）

**实现文件**: [VGG/kaggle/dalao.py](VGG/kaggle/dalao.py)

- **架构**: 使用预训练的VGG16模型
- **策略**: 冻结特征提取层，只训练分类层
- **修改**: 将最后一层全连接层输出修改为120类
- **优化器**: SGD
- **学习率**: 0.001

**关键代码**:
```python
vgg16 = models.vgg16(pretrained=True)
# 冻结特征层
for param in vgg16.features.parameters():
    param.requires_grad = False
# 修改分类层
vgg16.classifier[6] = nn.Linear(n_inputs, 120)
```

### 2. ResNet18（从零训练）

**实现文件**: [ResNet/test0518V1.py](ResNet/test0518V1.py)

- **架构**: 标准ResNet18结构
- **残差块**: BasicBlock，配置为[2,2,2,2]
- **训练策略**: 从零开始训练，无预训练权重
- **优化器**: Adam
- **学习率**: 0.001，使用StepLR调度器
- **训练轮数**: 10 epochs

### 3. ResNet50（迁移学习）⭐

**实现文件**: [ResNet50/luchi_dalao/dalao_bendi.py](ResNet50/luchi_dalao/dalao_bendi.py)

- **架构**: 预训练ResNet50 + 自定义输出层
- **特征提取**: 使用ImageNet预训练权重，冻结参数
- **输出网络**: 1000 → 256 → 120（带ReLU激活）
- **优化器**: Adam
- **学习率**: 1e-4
- **权重衰减**: 1e-3
- **训练轮数**: 50 epochs
- **批大小**: 128

**网络结构**:
```python
finetune_net.features = torchvision.models.resnet50(weights="ResNet50_Weights.DEFAULT")
finetune_net.output_new = nn.Sequential(
    nn.Linear(1000, 256),
    nn.ReLU(),
    nn.Linear(256, 120)
)
```

### 4. ResNet + SE注意力机制

**实现文件**: [ResNet_Seblock/fangzhang_ResNet_kaggle_dalao/PhanDai_ResNet_seblock_kaggle_dalao/fangzhao_dalao_V1.py](ResNet_Seblock/fangzhang_ResNet_kaggle_dalao/PhanDai_ResNet_seblock_kaggle_dalao/fangzhao_dalao_V1.py)

- **创新点**: 在ResNet残差块中集成SE（Squeeze-and-Excitation）注意力机制
- **SE块作用**: 通过通道注意力机制自适应地重新校准通道特征响应
- **降维比例**: 16（reduction ratio）

**SE块结构**:
```python
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )
```

### 5. 集成学习方案（Inception-V3 + ResNet50）

**实现文件**: [Enrique_Manuel_dalao/dalao.py](Enrique_Manuel_dalao/dalao.py)

- **模型融合**: Inception-V3和ResNet50并行处理
- **特征拼接**: 两个模型的输出特征拼接后通过共享分类头
- **多投票机制**: 5个输出头进行集成投票
- **参数量**: 45.79M
- **模型大小**: 174.99 MB
- **计算量**: 13.98 GFLOPs/图像

**融合架构**:
```python
class Model(nn.Module):
    def forward(self, x):
        X1 = self.inception_model(x)  # Inception-V3特征
        X2 = self.resnet50_model(x)   # ResNet50特征
        X = torch.cat([X1, X2], dim=1) # 特征拼接
        # 5个输出头用于投票
        return [P1, P2, P3, P4, P5]
```

## 数据预处理

### 训练集增强
```python
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

### 验证/测试集预处理
```python
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

## 实验结果

### 模型性能对比

| 模型 | Top-1准确率 | Top-5准确率 | 训练轮数 | 训练时间 | 推理速度 | 显存占用 |
|------|------------|------------|---------|---------|---------|---------|
| VGG16 | - | - | 15 | - | - | - |
| ResNet18 | - | - | 10 | - | - | - |
| **ResNet50** | **88.26%** | **98.58%** | 50 | 98.67分钟 | 4.96 ms/图像 | 1537.80 MB |
| ResNet+SE | - | - | - | - | - | - |
| Inception+ResNet50 | 73.40% | 95.40% | 6 | 15.47分钟 | 2.81 ms/图像 | 1082.59 MB |

### ResNet50详细训练过程 ⭐

**模型配置**:
- 参数量: 0.29M
- 模型大小: 98.79 MB
- 计算量: 8.27 GFLOPs/图像
- 批大小: 128
- 学习率: 1e-4
- 优化器: Adam (weight_decay=1e-3)

**训练曲线关键节点**:

| Epoch | 训练损失 | 训练准确率 | 验证损失 | 验证准确率 | Top-5准确率 |
|-------|---------|-----------|---------|-----------|-----------|
| 1 | 4.696 | 4.99% | 4.562 | 16.29% | 41.71% |
| 5 | 3.168 | 51.46% | 2.860 | 68.61% | 94.77% |
| 10 | 1.779 | 66.41% | 1.375 | 81.42% | 98.09% |
| 20 | 1.167 | 72.91% | 0.680 | 86.36% | 98.48% |
| 30 | 1.020 | 74.68% | 0.532 | 87.19% | 98.48% |
| 40 | 0.972 | 75.29% | 0.471 | 87.29% | 98.63% |
| **50** | **0.907** | **76.88%** | **0.432** | **88.26%** | **98.58%** |

**最终性能**:
- 最佳验证损失: 0.43189 (Epoch 50)
- 最终Top-1准确率: 88.26%
- 最终Top-5准确率: 98.58%
- 训练总耗时: 98.67分钟
- 平均每epoch耗时: 118.41秒
- 推理平均速度: 4.960 ms/图像
- 推理最大显存占用: 1537.80 MB

**训练曲线图**:
- 准确率曲线: [ResNet50/luchi_dalao/dalao_bendi/acc_curve_resnet50.png](ResNet50/luchi_dalao/dalao_bendi/acc_curve_resnet50.png)
- 损失曲线: [ResNet50/luchi_dalao/dalao_bendi/loss_curve_resnet50.png](ResNet50/luchi_dalao/dalao_bendi/loss_curve_resnet50.png)

### 集成学习方案详细结果

**模型配置**:
- 参数量: 45.79M
- 模型大小: 174.99 MB
- 计算量: 13.98 GFLOPs/图像
- 批大小: 64
- 训练轮数: 6 epochs

**训练过程**:

| Epoch | 训练损失 | 训练准确率 | 验证损失 | 验证准确率 | Top-5准确率 |
|-------|---------|-----------|---------|-----------|-----------|
| 1 | 2.008 | 50.54% | 1.064 | 68.46% | 94.13% |
| **2** | **0.887** | **74.00%** | **0.898** | **73.40%** | **95.40%** |
| 3 | 0.592 | 82.18% | 0.982 | 73.59% | 94.38% |
| 4 | 0.441 | 86.95% | 1.044 | 70.66% | 93.99% |
| 5 | 0.338 | 90.12% | 0.987 | 73.84% | 94.82% |
| 6 | 0.271 | 91.72% | 1.056 | 74.13% | 94.96% |

**最终性能**:
- 最佳验证损失: 0.89824 (Epoch 2)
- 最优Top-1准确率: 73.40%
- 最优Top-5准确率: 95.40%
- 训练总耗时: 15.47分钟
- 推理平均速度: 2.805 ms/图像
- 推理最大显存占用: 1082.59 MB

**训练曲线图**:
- 准确率曲线: [Enrique_Manuel_dalao/dalao_bendi_V1/acc_curve_ensemble.png](Enrique_Manuel_dalao/dalao_bendi_V1/acc_curve_ensemble.png)
- 损失曲线: [Enrique_Manuel_dalao/dalao_bendi_V1/loss_curve_ensemble.png](Enrique_Manuel_dalao/dalao_bendi_V1/loss_curve_ensemble.png)

### 实验结论

1. **ResNet50表现最佳**: 在所有测试的模型中，ResNet50迁移学习方案取得了最高的88.26%准确率，证明了预训练模型在细粒度分类任务中的有效性。

2. **Top-5准确率优异**: ResNet50的Top-5准确率达到98.58%，说明模型在前5个预测中几乎总能包含正确答案，这对于辅助决策系统非常有价值。

3. **训练效率**: ResNet50虽然训练了50个epoch，但在前10个epoch就已经达到81.42%的准确率，显示出良好的收敛速度。

4. **集成学习的权衡**: Inception+ResNet50集成方案虽然推理速度更快（2.81 ms vs 4.96 ms），但准确率较低（73.40% vs 88.26%），适合对速度要求高但对准确率要求相对较低的场景。

5. **过拟合控制**: 从训练曲线可以看出，ResNet50的训练准确率（76.88%）和验证准确率（88.26%）差距较大，这是因为使用了强数据增强和冻结特征提取层的策略。

## 使用方法

### 环境要求

```
Python >= 3.8
PyTorch >= 1.10.0
torchvision >= 0.11.0
CUDA >= 11.0 (推荐使用GPU)
```

### 安装依赖

```bash
pip install torch torchvision
pip install numpy pandas matplotlib tqdm
```

### 训练模型

#### ResNet50训练
```bash
cd ResNet50/luchi_dalao
python dalao_bendi.py
```

#### 集成学习训练
```bash
cd Enrique_Manuel_dalao
python dalao_bendi.py
```

#### VGG16训练
```bash
cd VGG/kaggle
python dalao.py
```

### 测试和推理

训练完成后，模型会自动在测试集上进行推理，并生成提交文件 `submission.csv`，格式符合Kaggle竞赛要求。

### 数据集准备

1. 从Kaggle下载数据集：https://www.kaggle.com/c/dog-breed-identification/data
2. 解压到 `dog-breed-identification/` 目录
3. 确保目录结构如下：
```
dog-breed-identification/
├── train/          # 训练图像
├── test/           # 测试图像
├── labels.csv      # 训练标签
└── sample_submission.csv
```

## 技术特点

### 1. 迁移学习
- 使用ImageNet预训练权重作为初始化
- 冻结特征提取层，只训练分类层
- 显著减少训练时间和所需数据量

### 2. 数据增强
- 随机裁剪和缩放
- 随机水平翻转
- 颜色抖动（亮度、对比度、饱和度）
- 标准化处理

### 3. 注意力机制
- SE（Squeeze-and-Excitation）块
- 通道级别的特征重标定
- 提升模型对重要特征的关注度

### 4. 集成学习
- 多模型融合（Inception-V3 + ResNet50）
- 多投票机制（5个输出头）
- 提高预测的鲁棒性

### 5. 性能优化
- 混合精度训练
- 学习率调度策略
- 早停机制防止过拟合

## 项目亮点

1. **多模型对比研究**: 实现了5种不同的深度学习架构，从基础VGG到先进的注意力机制和集成学习，全面对比了不同方法的性能。

2. **详细的实验记录**: 完整记录了训练过程中的损失、准确率变化，提供了可视化的训练曲线图，便于分析模型行为。

3. **高准确率**: ResNet50模型在验证集上达到88.26%的Top-1准确率和98.58%的Top-5准确率，表现优异。

4. **工程化实践**: 代码结构清晰，包含完整的数据预处理、模型训练、验证和测试流程，可直接用于实际应用。

5. **性能分析**: 提供了详细的推理速度和显存占用数据，便于在不同硬件环境下选择合适的模型。

## 参考资料

- [Kaggle Dog Breed Identification Competition](https://www.kaggle.com/c/dog-breed-identification)
- [Deep Residual Learning for Image Recognition (ResNet)](https://arxiv.org/abs/1512.03385)
- [Very Deep Convolutional Networks for Large-Scale Image Recognition (VGG)](https://arxiv.org/abs/1409.1556)
- [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)
- [Going Deeper with Convolutions (Inception)](https://arxiv.org/abs/1409.4842)

## 致谢

本项目是人工智能课程设计的第三个任务，感谢课程老师的指导和Kaggle社区提供的数据集。

---

**项目完成时间**: 2025年5月-6月
**课程**: 人工智能课程设计（三）
**任务**: Dog Breed Identification - 120类狗品种细粒度图像分类




