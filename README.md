# Fruit-Classifier · 50 类果蔬图像分类 (PyTorch)

自实现 **ResNet（残差结构）** 从零训练的多类别图像分类项目，覆盖 50 种常见果蔬。
包含完整的训练、测试集评估、单张图片预测流程，并支持数据增强、断点续训、中断自动保存与训练日志导出。

## 目录结构

```
fruit-classifier/
├── config.py          # 所有配置：数据路径、模型、超参、训练/预测模式
├── model.py           # 模型定义：ResNet(默认) / GoogLeNet / VGG / LeNet / AlexNet
├── train.py           # 训练入口（含断点续训、中断保存、日志输出）
├── pred.py            # 推理入口：predict_mode=0 测试集评估 / 1 单张图片预测
├── requirements.txt   # Python 依赖
├── README.md
└── .gitignore
```

> 数据文件体积大（约 5 GB / 7.2 万张），**不入 git 仓库**，请单独存放（本机示例放于仓库外的 `BIG/` 目录）。

## 数据集

50 类果蔬图像分类数据集（`Orange/Apple/.../Watermelon` 共 50 类，见 `config.py` 的 `class_names`），
按以下结构组织，类别文件夹名即标签：

```
data/
├── train/<类别>/图片     # 训练集
├── val/<类别>/图片       # 验证集
├── test/<类别>/图片      # 测试集
└── single/               # 单张预测用图片（banana.jpg 等）
```

原始数据已上传至 ModelScope 数据集仓库（git-lfs / zip 两种方式均可下载）：

**https://modelscope.cn/datasets/z2105372313/fruit-classifier**

> ⚠️ 类别名中存在数据集原始拼写（如 `Chilli Peper`、`Jalepeno`、`Raddish`、`Sweetpotato`），
> 它们是**文件夹名本身**，加载与改名时务必保持一致，否则 ImageFolder 会匹配不到。

## 环境安装

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

- 依赖中 `torch==2.11.0+cu128`、`torchvision==0.26.0+cu128` 带 CUDA 12.8 构建号，
  **只存在于 PyTorch 官方源**，因此必须加 `--extra-index-url`。
- CPU 机器：去掉 `+cu128` 后缀后从默认 PyPI 安装即可。
- 详细依赖与说明见 `requirements.txt` 内注释。

## 快速开始

### 0. 修改数据路径（重要）

`config.py` 中的 `data_root / val_root / test_root / single_root / save_path` 默认指向作者本机布局
（`./BIG/...`），**请改为你实际存放数据与权重的位置**。

### 1. 训练

```bash
python train.py
```

关键配置（均在 `config.py`）：

| 配置 | 默认 | 说明 |
|---|---|---|
| `epochs` | 5 | 本轮训练轮数（开启续训时叠加在已训轮次上） |
| `batch_size` | 100 | 批大小 |
| `num_workers` | 10 | 数据加载并行数 |
| `learning_rate` | 1e-4 | Adam 学习率 |
| `input_size` | 224 | 输入图尺寸 |
| `train_mode` | 1 | `0` 从头训练 / `1` 加载完整模型续训 / `2` 加载中断模型续训 |

- 每轮打印训练/验证 loss 与 acc；
- 训练过程写入 `train_log/train.csv`；
- 最佳权重与当前权重保存为含 `model_state_dict` 的 checkpoint；
- 手动中断（Ctrl+C）会自动把模型保存到 `interrupted_*.pth`，下次可用 `train_mode=2` 续训。

### 2. 测试集评估

```bash
python pred.py   # 需将 config.predict_mode 设为 0
```

遍历 `test_root` 全部图片，输出整体准确率。

### 3. 单张图片预测

```bash
python pred.py   # 需将 config.predict_mode 设为 1
```

自动扫描 `single_root` 目录下所有图片（jpg/jpeg/png/webp），输出：

```
banana.jpg           -> Banana   (置信度: 0.9999)
```

## ⚠️ 类别索引陷阱（务必阅读）

`torchvision.datasets.ImageFolder` 会**按文件夹名的字母序**给类别编号（`Apple`=0, `Avocado`=1, ...），
而**不是**数据集原始顺序。因此：

- `config.py` 的 `class_names` 必须保持**字母序**（本项目已按字母序排列）；
- 若改成"原始顺序"，训练本身不受影响，但预测显示的名字会整体错位
  （典型症状：`banana.jpg -> Jalepeno` 且置信度接近 1.0，其实模型认对了，只是查错了表）；
- 最稳妥的写法是在预测代码里直接用数据集自带的顺序：

```python
from torchvision.datasets import ImageFolder
classes = ImageFolder(root=cfg.data_root).classes   # 与训练时完全一致的字母序
label   = classes[predicted_idx]
```

## 训练结果（参考）

自实现 ResNet（无预训练）训练 30 轮：

| 指标 | 数值 |
|---|---|
| 训练集准确率 | ~83.3% |
| 验证集准确率 | ~82.1%（第 30 轮） |
| 测试集（同分布） | ~70%+ |

真实网络图片（背景/拍摄风格与训练集差异大）仍存在误判，属数据域差异。
如需更强的真实场景泛化，建议：使用预训练主干（如 `torchvision` 的 ResNet 预训练权重）、
扩充真实照片数据、或推理时做多尺度/TTA。

## 相关链接

- ModelScope 数据集（50 类果蔬原图）：https://modelscope.cn/datasets/z2105372313/fruit-classifier
- ModelScope 模型仓库（代码 + 权重）：https://modelscope.cn/models/z2105372313/fruit-classifier
