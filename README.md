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

> 数据文件体积大（约 5 GB / 7.2 万张），储放在modelscope

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


## 环境安装




## 快速开始

### 1，cd ./<自己的空文件夹>

### 2，下载源码和安装环境
```bash
git clone https://github.com/Minecraft-JoJo/fruit-classifier.git #克隆仓库
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128 #环境安装

```
- 依赖中 `torch==2.11.0+cu128`、`torchvision==0.26.0+cu128` 带 CUDA 12.8 构建号，
  **只存在于 PyTorch 官方源**，因此必须加 `--extra-index-url`。
- CPU 机器：去掉 `+cu128` 后缀后从默认 PyPI 安装即可。
- 详细依赖与说明见 `requirements.txt` 内注释。



### 3，下载数据集和权重（可选）
```bash
git lfs install
mkdir -p BIG/fruit-classifier
cd BIG
git clone https://www.modelscope.cn/datasets/z2105372313/fruit-classifier.git #克隆仓库
cd ..

```
### 4，(可选)只下载模型权重

```bash
mkdir -p BIG/fruit-classifier/save
mkdir -p BIG/fruit-classifier/data/single

modelscope download --dataset z2105372313/fruit-classifier save/checkpoint_all.pth --local_dir ./BIG/fruit-classifier/save/ 
```



### 关于py
### 1，训练
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
python pred.py   # 下载了数据集，可以将 config.predict_mode 设为 0
```

如果下载了数据集，遍历 `test_root` 全部图片，输出整体准确率。

### 3. 单张图片预测  预测图片放在BIG/fruit-classifier/data/single内，即可自动检测

```bash
python pred.py   # 需将 config.predict_mode 设为 1
```

自动扫描 `single_root` 目录下所有图片（jpg/jpeg/png/webp），输出：

```
banana.jpg           -> Banana   (置信度: 0.9999)
```



## 训练结果（参考）

自实现 ResNet（无预训练）训练 40 轮：

| 指标 | 数值 |
|---|---|
| 训练集准确率 | ~91.4% |
| 验证集准确率 | ~87.0%（第 38 轮） |
| 测试集（同分布） | ~82%+ |

真实网络图片（背景/拍摄风格与训练集差异大）仍存在误判，属数据域差异。
使用无预训练模型，无迁移学习

## 相关链接

- ModelScope 数据集（50 类果蔬原图）：https://modelscope.cn/datasets/z2105372313/fruit-classifier

