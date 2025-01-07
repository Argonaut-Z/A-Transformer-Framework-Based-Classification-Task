# 项目说明

## A Transformer Framework Based Classification Task

一个**基于Transformer Encoder网络架构**的文本分类模型

## 1. 环境准备

在conda环境下创建虚拟环境并安装如下必要库：

```shell
conda create -n transformer_classification python=3.10
conda activate transformer_classification
pip install -r requirements.txt
```

requirements.txt 内容如下：

```txt
torch==2.2.0
torchtext==0.17.0
numpy==1.26.4
```

也可以直接使用命令：

```shell
pip install torch==2.2.0 torchtext==0.17.0 numpy==1.26.4
```

## 2. 数据

数据已下载完毕，保存在该项目根目录下的data文件夹中

本项目使用了 **AG's News Topic Classification Dataset**，这是一个用于文本分类任务的经典数据集，包含新闻主题分类的标注数据。以下是数据集的详细信息：

------

#### **数据集来源**

- **名称**：AG's News Topic Classification Dataset
- **创建者**：Xiang Zhang（xiang.zhang@nyu.edu）
- **更新时间**：2015年9月9日
- **链接**：[AG Corpus of News Articles](http://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html)
- 背景：
  - 该数据集来源于 **ComeToMyHead** 新闻搜索引擎，这是一个从 2004 年运行的学术新闻搜索引擎。
  - 数据集中收集了来自 2000 多个新闻来源的超过 100 万篇新闻文章，经过预处理后用于研究数据挖掘、信息检索、数据压缩等任务。
  - 本项目使用的数据集由 Xiang Zhang 从原始新闻集合中筛选构建，作为文本分类基准测试数据集，广泛应用于学术研究。

#### **数据集构成**

- **类别**： 数据集中包含 4 个类别，每个类别表示一个新闻主题：
  - **World**（世界新闻）
  - **Sports**（体育新闻）
  - **Business**（商业新闻）
  - **Sci/Tech**（科学与技术新闻）
- **数据量**：
  - 训练集：
    - 每个类别包含 30,000 条训练样本。
    - 总计 120,000 条训练样本。
  - 测试集：
    - 每个类别包含 1,900 条测试样本。
    - 总计 7,600 条测试样本。
- **数据文件**：
  - **`classes.txt`**：包含类别对应的标签列表。
  - **`train.csv`**：训练数据，按逗号分隔，包含 3 列：
    - 类别索引（1 到 4）
    - 新闻标题
    - 新闻描述
  - **`test.csv`**：测试数据，格式与训练数据相同。

```css
"3","Wall St. Bears Claw Back Into the Black (Reuters)","Reuters - Short-sellers, Wall Street's dwindling\band of ultra-cynics, are seeing green again."
"3","Carlyle Looks Toward Commercial Aerospace (Reuters)","Reuters - Private investment firm Carlyle Group,\which has a reputation for making well-timed and occasionally\controversial plays in the defense industry, has quietly placed\its bets on another part of the market."
"3","Oil and Economy Cloud Stocks' Outlook (Reuters)","Reuters - Soaring crude prices plus worries\about the economy and the outlook for earnings are expected to\hang over the stock market next week during the depth of the\summer doldrums."
```

## 3. 项目目录结构

以下是 `classification` 文件夹下的目录结构和内容介绍：

```css
classification/
├── __pycache__/                 # Python缓存文件夹（自动生成，存储已编译的`.pyc`文件）
│   ├── ClassificationModel.cpython-310.pyc
│   ├── config.cpython-310.pyc
│   ├── data_helper.cpython-310.pyc
│   ├── Embedding.cpython-310.pyc
│   ├── MyTransformer.cpython-310.pyc
│
├── cache/                       # 存储模型的缓存文件
│   ├── model.pt                 # 训练好的模型参数文件（PyTorch格式）
│
├── data/ag_news_csv/            # 数据文件夹，存放 AG News 数据集
│   ├── classes.txt              # 类别文件，包含分类任务中的4个类别
│   ├── readme.txt               # 数据集说明文件
│   ├── test.csv                 # 测试集数据
│   ├── train.csv                # 训练集数据
│
├── 项目说明.md                   # 项目整体说明文件
├── Classification.ipynb         # 用于模型训练和评估的 Jupyter Notebook 文件
├── ClassificationModel.py       # 模型定义文件
├── config.py                    # 配置文件，包含超参数和路径设置
├── data_helper.py               # 数据处理模块，包含数据加载和预处理逻辑
├── Embedding.py                 # 嵌入层模块定义
├── MyTransformer.py             # Transformer 模型的实现
├── README.md                    # 项目主文档，介绍项目背景和使用方法
├── requirements.txt             # 项目依赖文件，包含所需 Python 包及版本
├── results.txt                  # 模型评估结果文件
├── train.py                     # 训练脚本，用于执行模型训练逻辑
```

## 4. 详细内容介绍

### 4.1 项目代码

- **`Classification.ipynb`**：
  - 使用 Jupyter Notebook 记录整个模型训练、测试和评估流程。
  - 包含数据加载、模型构建、训练过程和最终结果可视化等内容。
- **`ClassificationModel.py`**：
  - 定义文本分类模型的结构（基于 Transformer Encoder）。
  - 包含分类层和损失计算逻辑。
- **`config.py`**：
  - 配置文件，定义超参数（如学习率、批量大小、嵌入维度等）。
  - 包括路径配置（如数据文件夹、模型保存路径）。
- **`data_helper.py`**：
  - 数据处理模块，包含以下功能：
    1. 加载 `train.csv` 和 `test.csv` 数据。
    2. 构建词表（Vocabulary）。
    3. 数据分词和转换为索引。
    4. 数据批量加载和生成数据迭代器。
- **`Embedding.py`**：
  - 实现文本嵌入模块，包括：
    1. Token Embedding（词嵌入）。
    2. Positional Embedding（位置嵌入）。
- **`MyTransformer.py`**：
  - 实现 Transformer Encoder 的核心组件，包括：
    1. 自注意力机制（Self-Attention）。
    2. 多头注意力（Multi-Head Attention）。
    3. 前馈网络（Feed-Forward Network）。
- **`train.py`**：
  - 训练脚本，整合数据加载、模型构建和优化逻辑。
  - 定义训练循环，计算损失，评估模型。

------

### 4.2 数据部分

- `data/ag_news_csv/`：
  - 存储 AG News 数据集，包含训练集和测试集。
  - `classes.txt`：标注 4 个分类任务的类别名称（World、Sports、Business、Sci/Tech）。
  - `train.csv` 和 `test.csv`：
    - 存储新闻的标题和描述，按类别划分。
    - 格式为 CSV 文件，每行包含类别索引、标题和描述。

------

### 4.3 结果与缓存

- `cache/model.pt`：
  - 训练好的模型参数文件（PyTorch 格式），可用于推理或继续训练。
- `results.txt`：
  - 存储模型评估结果，如测试集上的准确率、损失等。

------

### 4.4 依赖文件

- `requirements.txt`：

  - 包含项目所需的 Python 包及版本，例如：

    ```txt
    torch==2.2.0
    torchtext==0.17.0
    numpy==1.26.4
    ```

  - 用户可以通过以下命令快速安装所有依赖：

    ```bash
    pip install -r requirements.txt
    ```

------

### 4.5 项目说明文档

- **`README.md`**：
  - 介绍项目背景、数据集描述、模型结构、运行步骤和结果。
  - 包含环境安装说明、运行代码示例等。
- **`项目说明.md`**：
  - 项目的中文说明，可能包含详细的实现逻辑和流程。

------

### 4.6 自动生成的文件

- `__pycache__/`：
  - 存储 Python 编译器生成的 `.pyc` 缓存文件，用于加速模块加载。
  - 自动生成，无需手动操作。

## 5. 使用方法

- STEP 1.直接下载或克隆本项目
- STEP 2.可自定义修改配置文件`config.py`中的配置参数，也可以保持默认

### 训练

直接执行如下命令即可进行模型训练：

```shell
python train.py
```

训练过程和结果：

```shell
120000it [00:03, 37696.60it/s]
100%|████████████████████████████████| 120000/120000 [00:08<00:00, 14025.64it/s]
100%|████████████████████████████████████| 7600/7600 [00:00<00:00, 13767.54it/s]
Epoch: 0, Batch[0/469], Train loss :206.268, Train acc: 0.254
Epoch: 0, Batch[10/469], Train loss :59.090, Train acc: 0.219
Epoch: 0, Batch[20/469], Train loss :124.455, Train acc: 0.258
Epoch: 0, Batch[30/469], Train loss :52.581, Train acc: 0.242
Epoch: 0, Batch[40/469], Train loss :72.588, Train acc: 0.230
Epoch: 0, Batch[50/469], Train loss :53.792, Train acc: 0.238
Epoch: 0, Batch[60/469], Train loss :37.307, Train acc: 0.246
Epoch: 0, Batch[70/469], Train loss :33.810, Train acc: 0.281
Epoch: 0, Batch[80/469], Train loss :28.841, Train acc: 0.246
Epoch: 0, Batch[90/469], Train loss :48.872, Train acc: 0.285
Epoch: 0, Batch[100/469], Train loss :40.119, Train acc: 0.312
Epoch: 0, Batch[110/469], Train loss :33.111, Train acc: 0.289
Epoch: 0, Batch[120/469], Train loss :38.087, Train acc: 0.254
Epoch: 0, Batch[130/469], Train loss :26.282, Train acc: 0.301
Epoch: 0, Batch[140/469], Train loss :24.425, Train acc: 0.332
Epoch: 0, Batch[150/469], Train loss :29.896, Train acc: 0.340
Epoch: 0, Batch[160/469], Train loss :15.590, Train acc: 0.375
Epoch: 0, Batch[170/469], Train loss :38.700, Train acc: 0.379
Epoch: 0, Batch[180/469], Train loss :25.385, Train acc: 0.418
Epoch: 0, Batch[190/469], Train loss :15.685, Train acc: 0.406
Epoch: 0, Batch[200/469], Train loss :23.506, Train acc: 0.441
Epoch: 0, Batch[210/469], Train loss :15.537, Train acc: 0.516
Epoch: 0, Batch[220/469], Train loss :37.797, Train acc: 0.328
Epoch: 0, Batch[230/469], Train loss :11.361, Train acc: 0.551
Epoch: 0, Batch[240/469], Train loss :11.041, Train acc: 0.555
Epoch: 0, Batch[250/469], Train loss :9.043, Train acc: 0.535
Epoch: 0, Batch[260/469], Train loss :6.351, Train acc: 0.629
Epoch: 0, Batch[270/469], Train loss :20.490, Train acc: 0.570
Epoch: 0, Batch[280/469], Train loss :7.322, Train acc: 0.703
Epoch: 0, Batch[290/469], Train loss :12.949, Train acc: 0.645
Epoch: 0, Batch[300/469], Train loss :12.027, Train acc: 0.566
Epoch: 0, Batch[310/469], Train loss :13.327, Train acc: 0.664
Epoch: 0, Batch[320/469], Train loss :11.140, Train acc: 0.555
Epoch: 0, Batch[330/469], Train loss :7.925, Train acc: 0.785
Epoch: 0, Batch[340/469], Train loss :4.429, Train acc: 0.793
Epoch: 0, Batch[350/469], Train loss :12.635, Train acc: 0.637
Epoch: 0, Batch[360/469], Train loss :13.304, Train acc: 0.586
Epoch: 0, Batch[370/469], Train loss :8.791, Train acc: 0.691
Epoch: 0, Batch[380/469], Train loss :7.163, Train acc: 0.723
Epoch: 0, Batch[390/469], Train loss :8.499, Train acc: 0.699
Epoch: 0, Batch[400/469], Train loss :4.928, Train acc: 0.828
Epoch: 0, Batch[410/469], Train loss :8.157, Train acc: 0.648
Epoch: 0, Batch[420/469], Train loss :10.287, Train acc: 0.723
Epoch: 0, Batch[430/469], Train loss :5.288, Train acc: 0.742
Epoch: 0, Batch[440/469], Train loss :3.864, Train acc: 0.824
Epoch: 0, Batch[450/469], Train loss :13.785, Train acc: 0.684
Epoch: 0, Batch[460/469], Train loss :25.612, Train acc: 0.605
Epoch: 0, Train loss: 25.431, Epoch time = 130.800s
Epoch: 1, Batch[0/469], Train loss :7.928, Train acc: 0.754
Epoch: 1, Batch[10/469], Train loss :4.688, Train acc: 0.855
Epoch: 1, Batch[20/469], Train loss :7.529, Train acc: 0.750
Epoch: 1, Batch[30/469], Train loss :8.114, Train acc: 0.773
Epoch: 1, Batch[40/469], Train loss :3.901, Train acc: 0.844
Epoch: 1, Batch[50/469], Train loss :8.176, Train acc: 0.770
Epoch: 1, Batch[60/469], Train loss :2.774, Train acc: 0.801
Epoch: 1, Batch[70/469], Train loss :5.999, Train acc: 0.797
Epoch: 1, Batch[80/469], Train loss :4.107, Train acc: 0.781
Epoch: 1, Batch[90/469], Train loss :6.731, Train acc: 0.816
Epoch: 1, Batch[100/469], Train loss :3.147, Train acc: 0.812
Epoch: 1, Batch[110/469], Train loss :23.443, Train acc: 0.680
Epoch: 1, Batch[120/469], Train loss :8.735, Train acc: 0.777
Epoch: 1, Batch[130/469], Train loss :5.797, Train acc: 0.848
Epoch: 1, Batch[140/469], Train loss :3.106, Train acc: 0.832
Epoch: 1, Batch[150/469], Train loss :3.575, Train acc: 0.852
Epoch: 1, Batch[160/469], Train loss :7.226, Train acc: 0.781
Epoch: 1, Batch[170/469], Train loss :13.237, Train acc: 0.586
Epoch: 1, Batch[180/469], Train loss :5.259, Train acc: 0.863
Epoch: 1, Batch[190/469], Train loss :6.088, Train acc: 0.812
Epoch: 1, Batch[200/469], Train loss :2.698, Train acc: 0.852
Epoch: 1, Batch[210/469], Train loss :3.813, Train acc: 0.809
Epoch: 1, Batch[220/469], Train loss :3.178, Train acc: 0.871
Epoch: 1, Batch[230/469], Train loss :5.402, Train acc: 0.809
Epoch: 1, Batch[240/469], Train loss :5.980, Train acc: 0.820
Epoch: 1, Batch[250/469], Train loss :5.078, Train acc: 0.820
Epoch: 1, Batch[260/469], Train loss :3.750, Train acc: 0.836
Epoch: 1, Batch[270/469], Train loss :9.894, Train acc: 0.805
Epoch: 1, Batch[280/469], Train loss :7.712, Train acc: 0.824
Epoch: 1, Batch[290/469], Train loss :2.607, Train acc: 0.863
Epoch: 1, Batch[300/469], Train loss :7.439, Train acc: 0.758
Epoch: 1, Batch[310/469], Train loss :7.166, Train acc: 0.867
Epoch: 1, Batch[320/469], Train loss :5.768, Train acc: 0.754
Epoch: 1, Batch[330/469], Train loss :3.616, Train acc: 0.816
Epoch: 1, Batch[340/469], Train loss :4.083, Train acc: 0.859
Epoch: 1, Batch[350/469], Train loss :3.740, Train acc: 0.852
Epoch: 1, Batch[360/469], Train loss :3.329, Train acc: 0.887
Epoch: 1, Batch[370/469], Train loss :4.428, Train acc: 0.855
Epoch: 1, Batch[380/469], Train loss :5.707, Train acc: 0.801
Epoch: 1, Batch[390/469], Train loss :3.008, Train acc: 0.883
Epoch: 1, Batch[400/469], Train loss :2.176, Train acc: 0.836
Epoch: 1, Batch[410/469], Train loss :22.149, Train acc: 0.695
Epoch: 1, Batch[420/469], Train loss :6.638, Train acc: 0.895
Epoch: 1, Batch[430/469], Train loss :6.164, Train acc: 0.840
Epoch: 1, Batch[440/469], Train loss :4.708, Train acc: 0.859
Epoch: 1, Batch[450/469], Train loss :2.869, Train acc: 0.824
Epoch: 1, Batch[460/469], Train loss :4.408, Train acc: 0.844
Epoch: 1, Train loss: 6.087, Epoch time = 130.747s
Accuracy on test 0.480, max acc on test 0.000
Epoch: 2, Batch[0/469], Train loss :3.057, Train acc: 0.715
Epoch: 2, Batch[10/469], Train loss :2.681, Train acc: 0.906
Epoch: 2, Batch[20/469], Train loss :4.534, Train acc: 0.871
Epoch: 2, Batch[30/469], Train loss :5.382, Train acc: 0.789
Epoch: 2, Batch[40/469], Train loss :2.371, Train acc: 0.914
Epoch: 2, Batch[50/469], Train loss :3.687, Train acc: 0.887
Epoch: 2, Batch[60/469], Train loss :1.759, Train acc: 0.898
Epoch: 2, Batch[70/469], Train loss :4.753, Train acc: 0.840
Epoch: 2, Batch[80/469], Train loss :3.143, Train acc: 0.895
Epoch: 2, Batch[90/469], Train loss :1.510, Train acc: 0.883
Epoch: 2, Batch[100/469], Train loss :1.718, Train acc: 0.934
Epoch: 2, Batch[110/469], Train loss :1.624, Train acc: 0.855
Epoch: 2, Batch[120/469], Train loss :14.743, Train acc: 0.746
Epoch: 2, Batch[130/469], Train loss :9.990, Train acc: 0.844
Epoch: 2, Batch[140/469], Train loss :4.441, Train acc: 0.855
Epoch: 2, Batch[150/469], Train loss :2.852, Train acc: 0.859
Epoch: 2, Batch[160/469], Train loss :2.125, Train acc: 0.875
Epoch: 2, Batch[170/469], Train loss :5.033, Train acc: 0.840
Epoch: 2, Batch[180/469], Train loss :2.133, Train acc: 0.891
Epoch: 2, Batch[190/469], Train loss :5.251, Train acc: 0.832
Epoch: 2, Batch[200/469], Train loss :2.759, Train acc: 0.867
Epoch: 2, Batch[210/469], Train loss :2.382, Train acc: 0.891
Epoch: 2, Batch[220/469], Train loss :1.834, Train acc: 0.859
Epoch: 2, Batch[230/469], Train loss :4.527, Train acc: 0.852
Epoch: 2, Batch[240/469], Train loss :9.552, Train acc: 0.828
Epoch: 2, Batch[250/469], Train loss :11.042, Train acc: 0.859
Epoch: 2, Batch[260/469], Train loss :13.040, Train acc: 0.680
Epoch: 2, Batch[270/469], Train loss :10.120, Train acc: 0.844
Epoch: 2, Batch[280/469], Train loss :9.633, Train acc: 0.730
Epoch: 2, Batch[290/469], Train loss :9.405, Train acc: 0.754
Epoch: 2, Batch[300/469], Train loss :4.333, Train acc: 0.891
Epoch: 2, Batch[310/469], Train loss :2.072, Train acc: 0.867
Epoch: 2, Batch[320/469], Train loss :4.491, Train acc: 0.758
Epoch: 2, Batch[330/469], Train loss :3.452, Train acc: 0.918
Epoch: 2, Batch[340/469], Train loss :2.090, Train acc: 0.910
Epoch: 2, Batch[350/469], Train loss :3.890, Train acc: 0.758
Epoch: 2, Batch[360/469], Train loss :9.875, Train acc: 0.691
Epoch: 2, Batch[370/469], Train loss :3.469, Train acc: 0.883
Epoch: 2, Batch[380/469], Train loss :5.371, Train acc: 0.855
Epoch: 2, Batch[390/469], Train loss :9.358, Train acc: 0.738
Epoch: 2, Batch[400/469], Train loss :7.780, Train acc: 0.902
Epoch: 2, Batch[410/469], Train loss :3.005, Train acc: 0.836
Epoch: 2, Batch[420/469], Train loss :7.378, Train acc: 0.840
Epoch: 2, Batch[430/469], Train loss :5.291, Train acc: 0.836
Epoch: 2, Batch[440/469], Train loss :7.468, Train acc: 0.785
Epoch: 2, Batch[450/469], Train loss :3.019, Train acc: 0.883
Epoch: 2, Batch[460/469], Train loss :5.736, Train acc: 0.840
Epoch: 2, Train loss: 5.328, Epoch time = 130.070s
Epoch: 3, Batch[0/469], Train loss :6.738, Train acc: 0.801
Epoch: 3, Batch[10/469], Train loss :6.826, Train acc: 0.887
Epoch: 3, Batch[20/469], Train loss :1.390, Train acc: 0.906
Epoch: 3, Batch[30/469], Train loss :3.234, Train acc: 0.770
Epoch: 3, Batch[40/469], Train loss :2.560, Train acc: 0.891
Epoch: 3, Batch[50/469], Train loss :2.429, Train acc: 0.945
Epoch: 3, Batch[60/469], Train loss :3.784, Train acc: 0.832
Epoch: 3, Batch[70/469], Train loss :2.960, Train acc: 0.930
Epoch: 3, Batch[80/469], Train loss :9.129, Train acc: 0.707
Epoch: 3, Batch[90/469], Train loss :2.998, Train acc: 0.902
Epoch: 3, Batch[100/469], Train loss :3.104, Train acc: 0.848
Epoch: 3, Batch[110/469], Train loss :2.438, Train acc: 0.797
Epoch: 3, Batch[120/469], Train loss :4.396, Train acc: 0.762
Epoch: 3, Batch[130/469], Train loss :2.150, Train acc: 0.887
Epoch: 3, Batch[140/469], Train loss :4.250, Train acc: 0.855
Epoch: 3, Batch[150/469], Train loss :4.069, Train acc: 0.832
Epoch: 3, Batch[160/469], Train loss :2.343, Train acc: 0.887
Epoch: 3, Batch[170/469], Train loss :7.562, Train acc: 0.793
Epoch: 3, Batch[180/469], Train loss :3.021, Train acc: 0.887
Epoch: 3, Batch[190/469], Train loss :3.721, Train acc: 0.777
Epoch: 3, Batch[200/469], Train loss :3.423, Train acc: 0.863
Epoch: 3, Batch[210/469], Train loss :0.618, Train acc: 0.918
Epoch: 3, Batch[220/469], Train loss :4.558, Train acc: 0.879
Epoch: 3, Batch[230/469], Train loss :4.322, Train acc: 0.902
Epoch: 3, Batch[240/469], Train loss :6.436, Train acc: 0.703
Epoch: 3, Batch[250/469], Train loss :2.226, Train acc: 0.887
Epoch: 3, Batch[260/469], Train loss :5.136, Train acc: 0.719
Epoch: 3, Batch[270/469], Train loss :2.991, Train acc: 0.898
Epoch: 3, Batch[280/469], Train loss :4.983, Train acc: 0.723
Epoch: 3, Batch[290/469], Train loss :10.532, Train acc: 0.785
Epoch: 3, Batch[300/469], Train loss :11.349, Train acc: 0.828
Epoch: 3, Batch[310/469], Train loss :3.499, Train acc: 0.844
Epoch: 3, Batch[320/469], Train loss :3.684, Train acc: 0.859
Epoch: 3, Batch[330/469], Train loss :9.612, Train acc: 0.684
Epoch: 3, Batch[340/469], Train loss :3.350, Train acc: 0.855
Epoch: 3, Batch[350/469], Train loss :3.589, Train acc: 0.805
Epoch: 3, Batch[360/469], Train loss :2.483, Train acc: 0.934
Epoch: 3, Batch[370/469], Train loss :0.986, Train acc: 0.902
Epoch: 3, Batch[380/469], Train loss :1.773, Train acc: 0.844
Epoch: 3, Batch[390/469], Train loss :6.682, Train acc: 0.871
Epoch: 3, Batch[400/469], Train loss :8.887, Train acc: 0.855
Epoch: 3, Batch[410/469], Train loss :3.502, Train acc: 0.750
Epoch: 3, Batch[420/469], Train loss :3.984, Train acc: 0.738
Epoch: 3, Batch[430/469], Train loss :5.445, Train acc: 0.895
Epoch: 3, Batch[440/469], Train loss :4.749, Train acc: 0.836
Epoch: 3, Batch[450/469], Train loss :1.729, Train acc: 0.836
Epoch: 3, Batch[460/469], Train loss :2.125, Train acc: 0.887
Epoch: 3, Train loss: 5.128, Epoch time = 129.755s
Accuracy on test 0.534, max acc on test 0.480
Epoch: 4, Batch[0/469], Train loss :6.253, Train acc: 0.867
Epoch: 4, Batch[10/469], Train loss :1.331, Train acc: 0.906
Epoch: 4, Batch[20/469], Train loss :2.253, Train acc: 0.871
Epoch: 4, Batch[30/469], Train loss :2.390, Train acc: 0.875
Epoch: 4, Batch[40/469], Train loss :2.090, Train acc: 0.867
Epoch: 4, Batch[50/469], Train loss :5.751, Train acc: 0.422
Epoch: 4, Batch[60/469], Train loss :2.079, Train acc: 0.930
Epoch: 4, Batch[70/469], Train loss :1.525, Train acc: 0.941
Epoch: 4, Batch[80/469], Train loss :5.194, Train acc: 0.910
Epoch: 4, Batch[90/469], Train loss :4.406, Train acc: 0.902
Epoch: 4, Batch[100/469], Train loss :7.548, Train acc: 0.883
Epoch: 4, Batch[110/469], Train loss :2.093, Train acc: 0.910
Epoch: 4, Batch[120/469], Train loss :3.914, Train acc: 0.883
Epoch: 4, Batch[130/469], Train loss :13.227, Train acc: 0.805
Epoch: 4, Batch[140/469], Train loss :18.193, Train acc: 0.363
Epoch: 4, Batch[150/469], Train loss :5.824, Train acc: 0.840
Epoch: 4, Batch[160/469], Train loss :4.030, Train acc: 0.902
Epoch: 4, Batch[170/469], Train loss :2.567, Train acc: 0.832
Epoch: 4, Batch[180/469], Train loss :1.413, Train acc: 0.863
Epoch: 4, Batch[190/469], Train loss :1.075, Train acc: 0.910
Epoch: 4, Batch[200/469], Train loss :5.485, Train acc: 0.711
Epoch: 4, Batch[210/469], Train loss :2.919, Train acc: 0.766
Epoch: 4, Batch[220/469], Train loss :3.400, Train acc: 0.883
Epoch: 4, Batch[230/469], Train loss :1.954, Train acc: 0.867
Epoch: 4, Batch[240/469], Train loss :2.484, Train acc: 0.836
Epoch: 4, Batch[250/469], Train loss :0.974, Train acc: 0.926
Epoch: 4, Batch[260/469], Train loss :2.313, Train acc: 0.836
Epoch: 4, Batch[270/469], Train loss :0.920, Train acc: 0.875
Epoch: 4, Batch[280/469], Train loss :3.817, Train acc: 0.816
Epoch: 4, Batch[290/469], Train loss :2.200, Train acc: 0.855
Epoch: 4, Batch[300/469], Train loss :14.630, Train acc: 0.730
Epoch: 4, Batch[310/469], Train loss :1.206, Train acc: 0.891
Epoch: 4, Batch[320/469], Train loss :3.991, Train acc: 0.902
Epoch: 4, Batch[330/469], Train loss :3.783, Train acc: 0.879
Epoch: 4, Batch[340/469], Train loss :3.668, Train acc: 0.684
Epoch: 4, Batch[350/469], Train loss :5.326, Train acc: 0.902
Epoch: 4, Batch[360/469], Train loss :4.651, Train acc: 0.734
Epoch: 4, Batch[370/469], Train loss :2.046, Train acc: 0.859
Epoch: 4, Batch[380/469], Train loss :4.192, Train acc: 0.852
Epoch: 4, Batch[390/469], Train loss :1.911, Train acc: 0.766
Epoch: 4, Batch[400/469], Train loss :2.813, Train acc: 0.812
Epoch: 4, Batch[410/469], Train loss :3.419, Train acc: 0.875
Epoch: 4, Batch[420/469], Train loss :3.965, Train acc: 0.816
Epoch: 4, Batch[430/469], Train loss :2.498, Train acc: 0.789
Epoch: 4, Batch[440/469], Train loss :2.440, Train acc: 0.812
Epoch: 4, Batch[450/469], Train loss :2.188, Train acc: 0.887
Epoch: 4, Batch[460/469], Train loss :2.372, Train acc: 0.828
Epoch: 4, Train loss: 3.909, Epoch time = 130.042s
Epoch: 5, Batch[0/469], Train loss :1.506, Train acc: 0.898
Epoch: 5, Batch[10/469], Train loss :1.705, Train acc: 0.812
Epoch: 5, Batch[20/469], Train loss :3.111, Train acc: 0.777
Epoch: 5, Batch[30/469], Train loss :4.079, Train acc: 0.805
Epoch: 5, Batch[40/469], Train loss :11.174, Train acc: 0.594
Epoch: 5, Batch[50/469], Train loss :2.919, Train acc: 0.773
Epoch: 5, Batch[60/469], Train loss :4.038, Train acc: 0.926
Epoch: 5, Batch[70/469], Train loss :2.316, Train acc: 0.910
Epoch: 5, Batch[80/469], Train loss :1.902, Train acc: 0.742
Epoch: 5, Batch[90/469], Train loss :2.413, Train acc: 0.812
Epoch: 5, Batch[100/469], Train loss :1.491, Train acc: 0.793
Epoch: 5, Batch[110/469], Train loss :2.495, Train acc: 0.777
Epoch: 5, Batch[120/469], Train loss :8.391, Train acc: 0.891
Epoch: 5, Batch[130/469], Train loss :3.326, Train acc: 0.715
Epoch: 5, Batch[140/469], Train loss :3.448, Train acc: 0.898
Epoch: 5, Batch[150/469], Train loss :3.936, Train acc: 0.543
Epoch: 5, Batch[160/469], Train loss :2.993, Train acc: 0.762
Epoch: 5, Batch[170/469], Train loss :3.510, Train acc: 0.680
Epoch: 5, Batch[180/469], Train loss :1.653, Train acc: 0.887
Epoch: 5, Batch[190/469], Train loss :1.449, Train acc: 0.895
Epoch: 5, Batch[200/469], Train loss :9.671, Train acc: 0.465
Epoch: 5, Batch[210/469], Train loss :5.506, Train acc: 0.855
Epoch: 5, Batch[220/469], Train loss :4.832, Train acc: 0.859
Epoch: 5, Batch[230/469], Train loss :2.901, Train acc: 0.859
Epoch: 5, Batch[240/469], Train loss :6.839, Train acc: 0.848
Epoch: 5, Batch[250/469], Train loss :3.868, Train acc: 0.648
Epoch: 5, Batch[260/469], Train loss :4.549, Train acc: 0.824
Epoch: 5, Batch[270/469], Train loss :1.855, Train acc: 0.820
Epoch: 5, Batch[280/469], Train loss :2.013, Train acc: 0.859
Epoch: 5, Batch[290/469], Train loss :1.285, Train acc: 0.781
Epoch: 5, Batch[300/469], Train loss :2.634, Train acc: 0.738
Epoch: 5, Batch[310/469], Train loss :2.011, Train acc: 0.852
Epoch: 5, Batch[320/469], Train loss :4.438, Train acc: 0.676
Epoch: 5, Batch[330/469], Train loss :2.619, Train acc: 0.867
Epoch: 5, Batch[340/469], Train loss :1.824, Train acc: 0.867
Epoch: 5, Batch[350/469], Train loss :1.330, Train acc: 0.863
Epoch: 5, Batch[360/469], Train loss :1.207, Train acc: 0.770
Epoch: 5, Batch[370/469], Train loss :6.377, Train acc: 0.457
Epoch: 5, Batch[380/469], Train loss :7.703, Train acc: 0.773
Epoch: 5, Batch[390/469], Train loss :4.245, Train acc: 0.762
Epoch: 5, Batch[400/469], Train loss :1.777, Train acc: 0.812
Epoch: 5, Batch[410/469], Train loss :2.580, Train acc: 0.816
Epoch: 5, Batch[420/469], Train loss :1.603, Train acc: 0.805
Epoch: 5, Batch[430/469], Train loss :1.004, Train acc: 0.758
Epoch: 5, Batch[440/469], Train loss :3.161, Train acc: 0.758
Epoch: 5, Batch[450/469], Train loss :1.415, Train acc: 0.895
Epoch: 5, Batch[460/469], Train loss :1.277, Train acc: 0.820
Epoch: 5, Train loss: 3.309, Epoch time = 130.546s
Accuracy on test 0.470, max acc on test 0.534
Epoch: 6, Batch[0/469], Train loss :4.692, Train acc: 0.719
Epoch: 6, Batch[10/469], Train loss :1.556, Train acc: 0.867
Epoch: 6, Batch[20/469], Train loss :6.315, Train acc: 0.594
Epoch: 6, Batch[30/469], Train loss :1.152, Train acc: 0.895
Epoch: 6, Batch[40/469], Train loss :0.740, Train acc: 0.836
Epoch: 6, Batch[50/469], Train loss :0.949, Train acc: 0.871
Epoch: 6, Batch[60/469], Train loss :0.493, Train acc: 0.895
Epoch: 6, Batch[70/469], Train loss :2.789, Train acc: 0.602
Epoch: 6, Batch[80/469], Train loss :1.881, Train acc: 0.891
Epoch: 6, Batch[90/469], Train loss :0.762, Train acc: 0.855
Epoch: 6, Batch[100/469], Train loss :1.635, Train acc: 0.672
Epoch: 6, Batch[110/469], Train loss :3.535, Train acc: 0.855
Epoch: 6, Batch[120/469], Train loss :1.005, Train acc: 0.848
Epoch: 6, Batch[130/469], Train loss :1.580, Train acc: 0.840
Epoch: 6, Batch[140/469], Train loss :1.120, Train acc: 0.906
Epoch: 6, Batch[150/469], Train loss :3.360, Train acc: 0.660
Epoch: 6, Batch[160/469], Train loss :1.103, Train acc: 0.812
Epoch: 6, Batch[170/469], Train loss :0.999, Train acc: 0.828
Epoch: 6, Batch[180/469], Train loss :0.569, Train acc: 0.918
Epoch: 6, Batch[190/469], Train loss :0.527, Train acc: 0.887
Epoch: 6, Batch[200/469], Train loss :1.731, Train acc: 0.742
Epoch: 6, Batch[210/469], Train loss :2.372, Train acc: 0.863
Epoch: 6, Batch[220/469], Train loss :2.360, Train acc: 0.719
Epoch: 6, Batch[230/469], Train loss :2.032, Train acc: 0.668
Epoch: 6, Batch[240/469], Train loss :1.796, Train acc: 0.750
Epoch: 6, Batch[250/469], Train loss :0.588, Train acc: 0.898
Epoch: 6, Batch[260/469], Train loss :0.978, Train acc: 0.883
Epoch: 6, Batch[270/469], Train loss :0.606, Train acc: 0.867
Epoch: 6, Batch[280/469], Train loss :0.697, Train acc: 0.910
Epoch: 6, Batch[290/469], Train loss :2.832, Train acc: 0.828
Epoch: 6, Batch[300/469], Train loss :5.787, Train acc: 0.586
Epoch: 6, Batch[310/469], Train loss :1.309, Train acc: 0.836
Epoch: 6, Batch[320/469], Train loss :1.492, Train acc: 0.875
Epoch: 6, Batch[330/469], Train loss :1.742, Train acc: 0.633
Epoch: 6, Batch[340/469], Train loss :2.637, Train acc: 0.816
Epoch: 6, Batch[350/469], Train loss :1.270, Train acc: 0.789
Epoch: 6, Batch[360/469], Train loss :2.975, Train acc: 0.703
Epoch: 6, Batch[370/469], Train loss :3.694, Train acc: 0.879
Epoch: 6, Batch[380/469], Train loss :1.676, Train acc: 0.738
Epoch: 6, Batch[390/469], Train loss :0.539, Train acc: 0.898
Epoch: 6, Batch[400/469], Train loss :1.423, Train acc: 0.727
Epoch: 6, Batch[410/469], Train loss :0.666, Train acc: 0.898
Epoch: 6, Batch[420/469], Train loss :0.702, Train acc: 0.820
Epoch: 6, Batch[430/469], Train loss :5.024, Train acc: 0.672
Epoch: 6, Batch[440/469], Train loss :4.531, Train acc: 0.871
Epoch: 6, Batch[450/469], Train loss :1.314, Train acc: 0.879
Epoch: 6, Batch[460/469], Train loss :0.761, Train acc: 0.859
Epoch: 6, Train loss: 1.637, Epoch time = 130.921s
Epoch: 7, Batch[0/469], Train loss :1.112, Train acc: 0.828
Epoch: 7, Batch[10/469], Train loss :3.782, Train acc: 0.465
Epoch: 7, Batch[20/469], Train loss :2.475, Train acc: 0.809
Epoch: 7, Batch[30/469], Train loss :2.474, Train acc: 0.707
Epoch: 7, Batch[40/469], Train loss :1.291, Train acc: 0.902
Epoch: 7, Batch[50/469], Train loss :0.709, Train acc: 0.906
Epoch: 7, Batch[60/469], Train loss :0.829, Train acc: 0.816
Epoch: 7, Batch[70/469], Train loss :1.397, Train acc: 0.613
Epoch: 7, Batch[80/469], Train loss :1.570, Train acc: 0.785
Epoch: 7, Batch[90/469], Train loss :1.951, Train acc: 0.766
Epoch: 7, Batch[100/469], Train loss :2.623, Train acc: 0.461
Epoch: 7, Batch[110/469], Train loss :0.872, Train acc: 0.832
Epoch: 7, Batch[120/469], Train loss :0.639, Train acc: 0.797
Epoch: 7, Batch[130/469], Train loss :0.685, Train acc: 0.828
Epoch: 7, Batch[140/469], Train loss :0.506, Train acc: 0.840
Epoch: 7, Batch[150/469], Train loss :0.445, Train acc: 0.859
Epoch: 7, Batch[160/469], Train loss :0.400, Train acc: 0.895
Epoch: 7, Batch[170/469], Train loss :0.330, Train acc: 0.906
Epoch: 7, Batch[180/469], Train loss :0.445, Train acc: 0.871
Epoch: 7, Batch[190/469], Train loss :0.422, Train acc: 0.863
Epoch: 7, Batch[200/469], Train loss :0.719, Train acc: 0.852
Epoch: 7, Batch[210/469], Train loss :0.383, Train acc: 0.891
Epoch: 7, Batch[220/469], Train loss :0.405, Train acc: 0.852
Epoch: 7, Batch[230/469], Train loss :0.384, Train acc: 0.918
Epoch: 7, Batch[240/469], Train loss :0.426, Train acc: 0.895
Epoch: 7, Batch[250/469], Train loss :0.598, Train acc: 0.871
Epoch: 7, Batch[260/469], Train loss :1.572, Train acc: 0.609
Epoch: 7, Batch[270/469], Train loss :0.676, Train acc: 0.809
Epoch: 7, Batch[280/469], Train loss :0.716, Train acc: 0.816
Epoch: 7, Batch[290/469], Train loss :0.505, Train acc: 0.848
Epoch: 7, Batch[300/469], Train loss :0.534, Train acc: 0.898
Epoch: 7, Batch[310/469], Train loss :0.621, Train acc: 0.805
Epoch: 7, Batch[320/469], Train loss :0.494, Train acc: 0.887
Epoch: 7, Batch[330/469], Train loss :0.484, Train acc: 0.832
Epoch: 7, Batch[340/469], Train loss :1.436, Train acc: 0.918
Epoch: 7, Batch[350/469], Train loss :1.082, Train acc: 0.832
Epoch: 7, Batch[360/469], Train loss :0.726, Train acc: 0.805
Epoch: 7, Batch[370/469], Train loss :1.185, Train acc: 0.832
Epoch: 7, Batch[380/469], Train loss :0.665, Train acc: 0.875
Epoch: 7, Batch[390/469], Train loss :0.940, Train acc: 0.816
Epoch: 7, Batch[400/469], Train loss :0.746, Train acc: 0.859
Epoch: 7, Batch[410/469], Train loss :0.421, Train acc: 0.895
Epoch: 7, Batch[420/469], Train loss :0.393, Train acc: 0.883
Epoch: 7, Batch[430/469], Train loss :0.468, Train acc: 0.859
Epoch: 7, Batch[440/469], Train loss :0.332, Train acc: 0.902
Epoch: 7, Batch[450/469], Train loss :0.728, Train acc: 0.824
Epoch: 7, Batch[460/469], Train loss :0.576, Train acc: 0.902
Epoch: 7, Train loss: 0.831, Epoch time = 131.418s
Accuracy on test 0.726, max acc on test 0.534
Epoch: 8, Batch[0/469], Train loss :0.353, Train acc: 0.918
Epoch: 8, Batch[10/469], Train loss :0.265, Train acc: 0.926
Epoch: 8, Batch[20/469], Train loss :0.365, Train acc: 0.891
Epoch: 8, Batch[30/469], Train loss :0.354, Train acc: 0.891
Epoch: 8, Batch[40/469], Train loss :0.559, Train acc: 0.832
Epoch: 8, Batch[50/469], Train loss :0.514, Train acc: 0.840
Epoch: 8, Batch[60/469], Train loss :0.356, Train acc: 0.918
Epoch: 8, Batch[70/469], Train loss :0.409, Train acc: 0.887
Epoch: 8, Batch[80/469], Train loss :0.460, Train acc: 0.875
Epoch: 8, Batch[90/469], Train loss :0.377, Train acc: 0.879
Epoch: 8, Batch[100/469], Train loss :0.546, Train acc: 0.902
Epoch: 8, Batch[110/469], Train loss :0.458, Train acc: 0.863
Epoch: 8, Batch[120/469], Train loss :0.501, Train acc: 0.863
Epoch: 8, Batch[130/469], Train loss :0.296, Train acc: 0.922
Epoch: 8, Batch[140/469], Train loss :0.340, Train acc: 0.883
Epoch: 8, Batch[150/469], Train loss :1.341, Train acc: 0.648
Epoch: 8, Batch[160/469], Train loss :1.137, Train acc: 0.684
Epoch: 8, Batch[170/469], Train loss :0.929, Train acc: 0.594
Epoch: 8, Batch[180/469], Train loss :0.747, Train acc: 0.648
Epoch: 8, Batch[190/469], Train loss :0.763, Train acc: 0.672
Epoch: 8, Batch[200/469], Train loss :0.599, Train acc: 0.734
Epoch: 8, Batch[210/469], Train loss :0.742, Train acc: 0.734
Epoch: 8, Batch[220/469], Train loss :0.588, Train acc: 0.785
Epoch: 8, Batch[230/469], Train loss :0.669, Train acc: 0.832
Epoch: 8, Batch[240/469], Train loss :0.620, Train acc: 0.824
Epoch: 8, Batch[250/469], Train loss :0.464, Train acc: 0.898
Epoch: 8, Batch[260/469], Train loss :0.523, Train acc: 0.836
Epoch: 8, Batch[270/469], Train loss :0.614, Train acc: 0.785
Epoch: 8, Batch[280/469], Train loss :0.585, Train acc: 0.820
Epoch: 8, Batch[290/469], Train loss :0.639, Train acc: 0.840
Epoch: 8, Batch[300/469], Train loss :0.499, Train acc: 0.848
Epoch: 8, Batch[310/469], Train loss :0.985, Train acc: 0.539
Epoch: 8, Batch[320/469], Train loss :0.636, Train acc: 0.797
Epoch: 8, Batch[330/469], Train loss :0.472, Train acc: 0.836
Epoch: 8, Batch[340/469], Train loss :0.508, Train acc: 0.852
Epoch: 8, Batch[350/469], Train loss :0.411, Train acc: 0.859
Epoch: 8, Batch[360/469], Train loss :0.425, Train acc: 0.863
Epoch: 8, Batch[370/469], Train loss :0.360, Train acc: 0.914
Epoch: 8, Batch[380/469], Train loss :0.296, Train acc: 0.934
Epoch: 8, Batch[390/469], Train loss :0.492, Train acc: 0.871
Epoch: 8, Batch[400/469], Train loss :0.386, Train acc: 0.902
Epoch: 8, Batch[410/469], Train loss :0.731, Train acc: 0.816
Epoch: 8, Batch[420/469], Train loss :0.493, Train acc: 0.844
Epoch: 8, Batch[430/469], Train loss :0.372, Train acc: 0.875
Epoch: 8, Batch[440/469], Train loss :0.569, Train acc: 0.871
Epoch: 8, Batch[450/469], Train loss :0.791, Train acc: 0.820
Epoch: 8, Batch[460/469], Train loss :0.511, Train acc: 0.828
Epoch: 8, Train loss: 0.545, Epoch time = 131.104s
Epoch: 9, Batch[0/469], Train loss :0.523, Train acc: 0.840
Epoch: 9, Batch[10/469], Train loss :0.630, Train acc: 0.816
Epoch: 9, Batch[20/469], Train loss :0.541, Train acc: 0.887
Epoch: 9, Batch[30/469], Train loss :0.282, Train acc: 0.926
Epoch: 9, Batch[40/469], Train loss :0.390, Train acc: 0.895
Epoch: 9, Batch[50/469], Train loss :0.347, Train acc: 0.879
Epoch: 9, Batch[60/469], Train loss :0.640, Train acc: 0.859
Epoch: 9, Batch[70/469], Train loss :0.451, Train acc: 0.859
Epoch: 9, Batch[80/469], Train loss :0.379, Train acc: 0.906
Epoch: 9, Batch[90/469], Train loss :0.517, Train acc: 0.836
Epoch: 9, Batch[100/469], Train loss :0.454, Train acc: 0.871
Epoch: 9, Batch[110/469], Train loss :0.714, Train acc: 0.781
Epoch: 9, Batch[120/469], Train loss :0.637, Train acc: 0.785
Epoch: 9, Batch[130/469], Train loss :0.735, Train acc: 0.707
Epoch: 9, Batch[140/469], Train loss :0.898, Train acc: 0.652
Epoch: 9, Batch[150/469], Train loss :0.647, Train acc: 0.695
Epoch: 9, Batch[160/469], Train loss :0.804, Train acc: 0.707
Epoch: 9, Batch[170/469], Train loss :0.758, Train acc: 0.766
Epoch: 9, Batch[180/469], Train loss :0.850, Train acc: 0.730
Epoch: 9, Batch[190/469], Train loss :0.695, Train acc: 0.785
Epoch: 9, Batch[200/469], Train loss :0.776, Train acc: 0.680
Epoch: 9, Batch[210/469], Train loss :0.796, Train acc: 0.754
Epoch: 9, Batch[220/469], Train loss :0.716, Train acc: 0.746
Epoch: 9, Batch[230/469], Train loss :0.668, Train acc: 0.797
Epoch: 9, Batch[240/469], Train loss :0.713, Train acc: 0.750
Epoch: 9, Batch[250/469], Train loss :0.756, Train acc: 0.715
Epoch: 9, Batch[260/469], Train loss :0.800, Train acc: 0.707
Epoch: 9, Batch[270/469], Train loss :0.647, Train acc: 0.758
Epoch: 9, Batch[280/469], Train loss :0.661, Train acc: 0.738
Epoch: 9, Batch[290/469], Train loss :0.485, Train acc: 0.820
Epoch: 9, Batch[300/469], Train loss :0.637, Train acc: 0.793
Epoch: 9, Batch[310/469], Train loss :0.569, Train acc: 0.836
Epoch: 9, Batch[320/469], Train loss :0.558, Train acc: 0.812
Epoch: 9, Batch[330/469], Train loss :0.519, Train acc: 0.820
Epoch: 9, Batch[340/469], Train loss :0.623, Train acc: 0.816
Epoch: 9, Batch[350/469], Train loss :0.509, Train acc: 0.812
Epoch: 9, Batch[360/469], Train loss :0.437, Train acc: 0.848
Epoch: 9, Batch[370/469], Train loss :0.435, Train acc: 0.844
Epoch: 9, Batch[380/469], Train loss :0.445, Train acc: 0.867
Epoch: 9, Batch[390/469], Train loss :0.639, Train acc: 0.805
Epoch: 9, Batch[400/469], Train loss :0.461, Train acc: 0.855
Epoch: 9, Batch[410/469], Train loss :0.475, Train acc: 0.852
Epoch: 9, Batch[420/469], Train loss :0.341, Train acc: 0.883
Epoch: 9, Batch[430/469], Train loss :0.407, Train acc: 0.840
Epoch: 9, Batch[440/469], Train loss :0.338, Train acc: 0.906
Epoch: 9, Batch[450/469], Train loss :0.480, Train acc: 0.848
Epoch: 9, Batch[460/469], Train loss :0.407, Train acc: 0.875
Epoch: 9, Train loss: 0.571, Epoch time = 130.873s
Accuracy on test 0.844, max acc on test 0.726
120000it [00:03, 36944.79it/s]
100%|████████████████████████████████| 120000/120000 [00:08<00:00, 13514.88it/s]
100%|████████████████████████████████████| 7600/7600 [00:00<00:00, 14417.51it/s]
## 成功载入已有模型，进行追加训练......
Epoch: 0, Batch[0/469], Train loss :0.488, Train acc: 0.887
Epoch: 0, Batch[10/469], Train loss :0.428, Train acc: 0.875
Epoch: 0, Batch[20/469], Train loss :0.426, Train acc: 0.875
Epoch: 0, Batch[30/469], Train loss :0.293, Train acc: 0.922
Epoch: 0, Batch[40/469], Train loss :0.351, Train acc: 0.895
Epoch: 0, Batch[50/469], Train loss :0.293, Train acc: 0.891
Epoch: 0, Batch[60/469], Train loss :0.328, Train acc: 0.887
Epoch: 0, Batch[70/469], Train loss :0.366, Train acc: 0.898
Epoch: 0, Batch[80/469], Train loss :0.336, Train acc: 0.906
Epoch: 0, Batch[90/469], Train loss :0.360, Train acc: 0.883
Epoch: 0, Batch[100/469], Train loss :0.402, Train acc: 0.875
Epoch: 0, Batch[110/469], Train loss :0.330, Train acc: 0.902
Epoch: 0, Batch[120/469], Train loss :0.347, Train acc: 0.895
Epoch: 0, Batch[130/469], Train loss :0.351, Train acc: 0.898
Epoch: 0, Batch[140/469], Train loss :0.440, Train acc: 0.852
Epoch: 0, Batch[150/469], Train loss :0.356, Train acc: 0.906
Epoch: 0, Batch[160/469], Train loss :0.278, Train acc: 0.906
Epoch: 0, Batch[170/469], Train loss :0.478, Train acc: 0.848
Epoch: 0, Batch[180/469], Train loss :0.504, Train acc: 0.840
Epoch: 0, Batch[190/469], Train loss :0.336, Train acc: 0.910
Epoch: 0, Batch[200/469], Train loss :0.373, Train acc: 0.879
Epoch: 0, Batch[210/469], Train loss :0.338, Train acc: 0.906
Epoch: 0, Batch[220/469], Train loss :0.395, Train acc: 0.887
Epoch: 0, Batch[230/469], Train loss :0.299, Train acc: 0.895
Epoch: 0, Batch[240/469], Train loss :0.530, Train acc: 0.879
Epoch: 0, Batch[250/469], Train loss :0.371, Train acc: 0.898
Epoch: 0, Batch[260/469], Train loss :0.411, Train acc: 0.875
Epoch: 0, Batch[270/469], Train loss :0.422, Train acc: 0.871
Epoch: 0, Batch[280/469], Train loss :0.333, Train acc: 0.891
Epoch: 0, Batch[290/469], Train loss :0.408, Train acc: 0.875
Epoch: 0, Batch[300/469], Train loss :0.349, Train acc: 0.902
Epoch: 0, Batch[310/469], Train loss :0.328, Train acc: 0.914
Epoch: 0, Batch[320/469], Train loss :0.364, Train acc: 0.898
Epoch: 0, Batch[330/469], Train loss :0.409, Train acc: 0.879
Epoch: 0, Batch[340/469], Train loss :0.401, Train acc: 0.879
Epoch: 0, Batch[350/469], Train loss :0.308, Train acc: 0.914
Epoch: 0, Batch[360/469], Train loss :0.257, Train acc: 0.914
Epoch: 0, Batch[370/469], Train loss :0.358, Train acc: 0.887
Epoch: 0, Batch[380/469], Train loss :0.331, Train acc: 0.910
Epoch: 0, Batch[390/469], Train loss :0.347, Train acc: 0.891
Epoch: 0, Batch[400/469], Train loss :0.308, Train acc: 0.891
Epoch: 0, Batch[410/469], Train loss :0.515, Train acc: 0.844
Epoch: 0, Batch[420/469], Train loss :0.342, Train acc: 0.914
Epoch: 0, Batch[430/469], Train loss :0.513, Train acc: 0.879
Epoch: 0, Batch[440/469], Train loss :0.339, Train acc: 0.895
Epoch: 0, Batch[450/469], Train loss :0.386, Train acc: 0.879
Epoch: 0, Batch[460/469], Train loss :0.288, Train acc: 0.898
Epoch: 0, Train loss: 0.363, Epoch time = 131.286s
Epoch: 1, Batch[0/469], Train loss :0.472, Train acc: 0.879
Epoch: 1, Batch[10/469], Train loss :0.319, Train acc: 0.883
Epoch: 1, Batch[20/469], Train loss :0.329, Train acc: 0.898
Epoch: 1, Batch[30/469], Train loss :0.369, Train acc: 0.910
Epoch: 1, Batch[40/469], Train loss :0.386, Train acc: 0.879
Epoch: 1, Batch[50/469], Train loss :0.432, Train acc: 0.887
Epoch: 1, Batch[60/469], Train loss :0.336, Train acc: 0.902
Epoch: 1, Batch[70/469], Train loss :0.316, Train acc: 0.910
Epoch: 1, Batch[80/469], Train loss :0.198, Train acc: 0.926
Epoch: 1, Batch[90/469], Train loss :0.306, Train acc: 0.879
Epoch: 1, Batch[100/469], Train loss :0.265, Train acc: 0.930
Epoch: 1, Batch[110/469], Train loss :0.253, Train acc: 0.914
Epoch: 1, Batch[120/469], Train loss :0.334, Train acc: 0.887
Epoch: 1, Batch[130/469], Train loss :0.305, Train acc: 0.895
Epoch: 1, Batch[140/469], Train loss :0.352, Train acc: 0.902
Epoch: 1, Batch[150/469], Train loss :0.550, Train acc: 0.875
Epoch: 1, Batch[160/469], Train loss :0.373, Train acc: 0.887
Epoch: 1, Batch[170/469], Train loss :0.301, Train acc: 0.918
Epoch: 1, Batch[180/469], Train loss :0.329, Train acc: 0.898
Epoch: 1, Batch[190/469], Train loss :0.319, Train acc: 0.902
Epoch: 1, Batch[200/469], Train loss :0.354, Train acc: 0.895
Epoch: 1, Batch[210/469], Train loss :0.418, Train acc: 0.879
Epoch: 1, Batch[220/469], Train loss :0.303, Train acc: 0.934
Epoch: 1, Batch[230/469], Train loss :0.450, Train acc: 0.879
Epoch: 1, Batch[240/469], Train loss :0.369, Train acc: 0.887
Epoch: 1, Batch[250/469], Train loss :0.382, Train acc: 0.875
Epoch: 1, Batch[260/469], Train loss :0.355, Train acc: 0.898
Epoch: 1, Batch[270/469], Train loss :0.407, Train acc: 0.867
Epoch: 1, Batch[280/469], Train loss :0.412, Train acc: 0.895
Epoch: 1, Batch[290/469], Train loss :0.372, Train acc: 0.891
Epoch: 1, Batch[300/469], Train loss :0.353, Train acc: 0.883
Epoch: 1, Batch[310/469], Train loss :0.335, Train acc: 0.891
Epoch: 1, Batch[320/469], Train loss :0.465, Train acc: 0.871
Epoch: 1, Batch[330/469], Train loss :0.357, Train acc: 0.910
Epoch: 1, Batch[340/469], Train loss :0.345, Train acc: 0.887
Epoch: 1, Batch[350/469], Train loss :0.350, Train acc: 0.875
Epoch: 1, Batch[360/469], Train loss :0.294, Train acc: 0.910
Epoch: 1, Batch[370/469], Train loss :0.348, Train acc: 0.879
Epoch: 1, Batch[380/469], Train loss :0.331, Train acc: 0.875
Epoch: 1, Batch[390/469], Train loss :0.498, Train acc: 0.859
Epoch: 1, Batch[400/469], Train loss :0.318, Train acc: 0.895
Epoch: 1, Batch[410/469], Train loss :0.312, Train acc: 0.906
Epoch: 1, Batch[420/469], Train loss :0.319, Train acc: 0.895
Epoch: 1, Batch[430/469], Train loss :0.338, Train acc: 0.898
Epoch: 1, Batch[440/469], Train loss :0.307, Train acc: 0.898
Epoch: 1, Batch[450/469], Train loss :0.339, Train acc: 0.895
Epoch: 1, Batch[460/469], Train loss :0.398, Train acc: 0.867
Epoch: 1, Train loss: 0.339, Epoch time = 129.923s
Accuracy on test 0.867, max acc on test 0.000
Epoch: 2, Batch[0/469], Train loss :0.257, Train acc: 0.902
Epoch: 2, Batch[10/469], Train loss :0.405, Train acc: 0.875
Epoch: 2, Batch[20/469], Train loss :0.374, Train acc: 0.898
Epoch: 2, Batch[30/469], Train loss :0.371, Train acc: 0.867
Epoch: 2, Batch[40/469], Train loss :0.301, Train acc: 0.906
Epoch: 2, Batch[50/469], Train loss :0.260, Train acc: 0.898
Epoch: 2, Batch[60/469], Train loss :0.305, Train acc: 0.914
Epoch: 2, Batch[70/469], Train loss :0.253, Train acc: 0.918
Epoch: 2, Batch[80/469], Train loss :0.192, Train acc: 0.949
Epoch: 2, Batch[90/469], Train loss :0.347, Train acc: 0.875
Epoch: 2, Batch[100/469], Train loss :0.264, Train acc: 0.918
Epoch: 2, Batch[110/469], Train loss :0.331, Train acc: 0.926
Epoch: 2, Batch[120/469], Train loss :0.288, Train acc: 0.898
Epoch: 2, Batch[130/469], Train loss :0.256, Train acc: 0.914
Epoch: 2, Batch[140/469], Train loss :0.354, Train acc: 0.895
Epoch: 2, Batch[150/469], Train loss :0.326, Train acc: 0.930
Epoch: 2, Batch[160/469], Train loss :0.284, Train acc: 0.914
Epoch: 2, Batch[170/469], Train loss :0.344, Train acc: 0.891
Epoch: 2, Batch[180/469], Train loss :0.255, Train acc: 0.934
Epoch: 2, Batch[190/469], Train loss :0.297, Train acc: 0.914
Epoch: 2, Batch[200/469], Train loss :0.242, Train acc: 0.922
Epoch: 2, Batch[210/469], Train loss :0.320, Train acc: 0.898
Epoch: 2, Batch[220/469], Train loss :0.330, Train acc: 0.906
Epoch: 2, Batch[230/469], Train loss :0.324, Train acc: 0.910
Epoch: 2, Batch[240/469], Train loss :0.344, Train acc: 0.906
Epoch: 2, Batch[250/469], Train loss :0.319, Train acc: 0.898
Epoch: 2, Batch[260/469], Train loss :0.300, Train acc: 0.898
Epoch: 2, Batch[270/469], Train loss :0.371, Train acc: 0.910
Epoch: 2, Batch[280/469], Train loss :0.337, Train acc: 0.902
Epoch: 2, Batch[290/469], Train loss :0.256, Train acc: 0.922
Epoch: 2, Batch[300/469], Train loss :0.343, Train acc: 0.883
Epoch: 2, Batch[310/469], Train loss :0.401, Train acc: 0.867
Epoch: 2, Batch[320/469], Train loss :0.413, Train acc: 0.879
Epoch: 2, Batch[330/469], Train loss :0.293, Train acc: 0.910
Epoch: 2, Batch[340/469], Train loss :0.263, Train acc: 0.926
Epoch: 2, Batch[350/469], Train loss :0.276, Train acc: 0.914
Epoch: 2, Batch[360/469], Train loss :0.364, Train acc: 0.883
Epoch: 2, Batch[370/469], Train loss :0.211, Train acc: 0.910
Epoch: 2, Batch[380/469], Train loss :0.325, Train acc: 0.891
Epoch: 2, Batch[390/469], Train loss :0.251, Train acc: 0.902
Epoch: 2, Batch[400/469], Train loss :0.371, Train acc: 0.895
Epoch: 2, Batch[410/469], Train loss :0.387, Train acc: 0.879
Epoch: 2, Batch[420/469], Train loss :0.261, Train acc: 0.902
Epoch: 2, Batch[430/469], Train loss :0.321, Train acc: 0.918
Epoch: 2, Batch[440/469], Train loss :0.370, Train acc: 0.887
Epoch: 2, Batch[450/469], Train loss :0.402, Train acc: 0.879
Epoch: 2, Batch[460/469], Train loss :0.336, Train acc: 0.914
Epoch: 2, Train loss: 0.330, Epoch time = 130.142s
Epoch: 3, Batch[0/469], Train loss :0.312, Train acc: 0.906
Epoch: 3, Batch[10/469], Train loss :0.317, Train acc: 0.926
Epoch: 3, Batch[20/469], Train loss :0.340, Train acc: 0.887
Epoch: 3, Batch[30/469], Train loss :0.273, Train acc: 0.910
Epoch: 3, Batch[40/469], Train loss :0.317, Train acc: 0.887
Epoch: 3, Batch[50/469], Train loss :0.270, Train acc: 0.941
Epoch: 3, Batch[60/469], Train loss :0.330, Train acc: 0.910
Epoch: 3, Batch[70/469], Train loss :0.396, Train acc: 0.902
Epoch: 3, Batch[80/469], Train loss :0.277, Train acc: 0.914
Epoch: 3, Batch[90/469], Train loss :0.366, Train acc: 0.883
Epoch: 3, Batch[100/469], Train loss :0.189, Train acc: 0.941
Epoch: 3, Batch[110/469], Train loss :0.274, Train acc: 0.906
Epoch: 3, Batch[120/469], Train loss :0.360, Train acc: 0.906
Epoch: 3, Batch[130/469], Train loss :0.345, Train acc: 0.910
Epoch: 3, Batch[140/469], Train loss :0.260, Train acc: 0.918
Epoch: 3, Batch[150/469], Train loss :0.266, Train acc: 0.926
Epoch: 3, Batch[160/469], Train loss :0.359, Train acc: 0.887
Epoch: 3, Batch[170/469], Train loss :0.285, Train acc: 0.922
Epoch: 3, Batch[180/469], Train loss :0.378, Train acc: 0.875
Epoch: 3, Batch[190/469], Train loss :0.358, Train acc: 0.875
Epoch: 3, Batch[200/469], Train loss :0.387, Train acc: 0.879
Epoch: 3, Batch[210/469], Train loss :0.214, Train acc: 0.945
Epoch: 3, Batch[220/469], Train loss :0.303, Train acc: 0.926
Epoch: 3, Batch[230/469], Train loss :0.292, Train acc: 0.906
Epoch: 3, Batch[240/469], Train loss :0.348, Train acc: 0.883
Epoch: 3, Batch[250/469], Train loss :0.319, Train acc: 0.918
Epoch: 3, Batch[260/469], Train loss :0.276, Train acc: 0.934
Epoch: 3, Batch[270/469], Train loss :0.279, Train acc: 0.914
Epoch: 3, Batch[280/469], Train loss :0.202, Train acc: 0.930
Epoch: 3, Batch[290/469], Train loss :0.271, Train acc: 0.926
Epoch: 3, Batch[300/469], Train loss :0.312, Train acc: 0.914
Epoch: 3, Batch[310/469], Train loss :0.255, Train acc: 0.914
Epoch: 3, Batch[320/469], Train loss :0.369, Train acc: 0.883
Epoch: 3, Batch[330/469], Train loss :0.520, Train acc: 0.879
Epoch: 3, Batch[340/469], Train loss :0.270, Train acc: 0.953
Epoch: 3, Batch[350/469], Train loss :0.380, Train acc: 0.914
Epoch: 3, Batch[360/469], Train loss :0.433, Train acc: 0.902
Epoch: 3, Batch[370/469], Train loss :0.317, Train acc: 0.914
Epoch: 3, Batch[380/469], Train loss :0.364, Train acc: 0.898
Epoch: 3, Batch[390/469], Train loss :0.335, Train acc: 0.887
Epoch: 3, Batch[400/469], Train loss :0.533, Train acc: 0.887
Epoch: 3, Batch[410/469], Train loss :0.324, Train acc: 0.914
Epoch: 3, Batch[420/469], Train loss :0.326, Train acc: 0.906
Epoch: 3, Batch[430/469], Train loss :0.297, Train acc: 0.910
Epoch: 3, Batch[440/469], Train loss :0.359, Train acc: 0.902
Epoch: 3, Batch[450/469], Train loss :0.296, Train acc: 0.895
Epoch: 3, Batch[460/469], Train loss :0.268, Train acc: 0.922
Epoch: 3, Train loss: 0.315, Epoch time = 131.454s
Accuracy on test 0.866, max acc on test 0.867
Epoch: 4, Batch[0/469], Train loss :0.280, Train acc: 0.906
Epoch: 4, Batch[10/469], Train loss :0.312, Train acc: 0.902
Epoch: 4, Batch[20/469], Train loss :0.210, Train acc: 0.930
Epoch: 4, Batch[30/469], Train loss :0.250, Train acc: 0.918
Epoch: 4, Batch[40/469], Train loss :0.333, Train acc: 0.906
Epoch: 4, Batch[50/469], Train loss :0.280, Train acc: 0.941
Epoch: 4, Batch[60/469], Train loss :0.360, Train acc: 0.887
Epoch: 4, Batch[70/469], Train loss :0.277, Train acc: 0.918
Epoch: 4, Batch[80/469], Train loss :0.178, Train acc: 0.949
Epoch: 4, Batch[90/469], Train loss :0.266, Train acc: 0.910
Epoch: 4, Batch[100/469], Train loss :0.276, Train acc: 0.922
Epoch: 4, Batch[110/469], Train loss :0.318, Train acc: 0.914
Epoch: 4, Batch[120/469], Train loss :0.339, Train acc: 0.906
Epoch: 4, Batch[130/469], Train loss :0.246, Train acc: 0.926
Epoch: 4, Batch[140/469], Train loss :0.334, Train acc: 0.906
Epoch: 4, Batch[150/469], Train loss :0.237, Train acc: 0.930
Epoch: 4, Batch[160/469], Train loss :0.296, Train acc: 0.922
Epoch: 4, Batch[170/469], Train loss :0.293, Train acc: 0.902
Epoch: 4, Batch[180/469], Train loss :0.266, Train acc: 0.922
Epoch: 4, Batch[190/469], Train loss :0.369, Train acc: 0.887
Epoch: 4, Batch[200/469], Train loss :0.274, Train acc: 0.934
Epoch: 4, Batch[210/469], Train loss :0.278, Train acc: 0.922
Epoch: 4, Batch[220/469], Train loss :0.378, Train acc: 0.910
Epoch: 4, Batch[230/469], Train loss :0.321, Train acc: 0.895
Epoch: 4, Batch[240/469], Train loss :0.336, Train acc: 0.898
Epoch: 4, Batch[250/469], Train loss :0.315, Train acc: 0.918
Epoch: 4, Batch[260/469], Train loss :0.211, Train acc: 0.938
Epoch: 4, Batch[270/469], Train loss :0.267, Train acc: 0.938
Epoch: 4, Batch[280/469], Train loss :0.309, Train acc: 0.906
Epoch: 4, Batch[290/469], Train loss :0.332, Train acc: 0.895
Epoch: 4, Batch[300/469], Train loss :0.245, Train acc: 0.930
Epoch: 4, Batch[310/469], Train loss :0.301, Train acc: 0.934
Epoch: 4, Batch[320/469], Train loss :0.310, Train acc: 0.914
Epoch: 4, Batch[330/469], Train loss :0.232, Train acc: 0.938
Epoch: 4, Batch[340/469], Train loss :0.199, Train acc: 0.945
Epoch: 4, Batch[350/469], Train loss :0.233, Train acc: 0.922
Epoch: 4, Batch[360/469], Train loss :0.272, Train acc: 0.930
Epoch: 4, Batch[370/469], Train loss :0.218, Train acc: 0.953
Epoch: 4, Batch[380/469], Train loss :0.279, Train acc: 0.891
Epoch: 4, Batch[390/469], Train loss :0.231, Train acc: 0.938
Epoch: 4, Batch[400/469], Train loss :0.386, Train acc: 0.906
Epoch: 4, Batch[410/469], Train loss :0.254, Train acc: 0.934
Epoch: 4, Batch[420/469], Train loss :0.340, Train acc: 0.895
Epoch: 4, Batch[430/469], Train loss :0.395, Train acc: 0.887
Epoch: 4, Batch[440/469], Train loss :0.295, Train acc: 0.906
Epoch: 4, Batch[450/469], Train loss :0.326, Train acc: 0.883
Epoch: 4, Batch[460/469], Train loss :0.355, Train acc: 0.910
Epoch: 4, Train loss: 0.301, Epoch time = 131.056s
Epoch: 5, Batch[0/469], Train loss :0.280, Train acc: 0.918
Epoch: 5, Batch[10/469], Train loss :0.278, Train acc: 0.918
Epoch: 5, Batch[20/469], Train loss :0.273, Train acc: 0.922
Epoch: 5, Batch[30/469], Train loss :0.222, Train acc: 0.949
Epoch: 5, Batch[40/469], Train loss :0.171, Train acc: 0.941
Epoch: 5, Batch[50/469], Train loss :0.256, Train acc: 0.926
Epoch: 5, Batch[60/469], Train loss :0.348, Train acc: 0.906
Epoch: 5, Batch[70/469], Train loss :0.330, Train acc: 0.922
Epoch: 5, Batch[80/469], Train loss :0.228, Train acc: 0.941
Epoch: 5, Batch[90/469], Train loss :0.250, Train acc: 0.910
Epoch: 5, Batch[100/469], Train loss :0.222, Train acc: 0.922
Epoch: 5, Batch[110/469], Train loss :0.303, Train acc: 0.914
Epoch: 5, Batch[120/469], Train loss :0.297, Train acc: 0.883
Epoch: 5, Batch[130/469], Train loss :0.303, Train acc: 0.922
Epoch: 5, Batch[140/469], Train loss :0.186, Train acc: 0.941
Epoch: 5, Batch[150/469], Train loss :0.246, Train acc: 0.918
Epoch: 5, Batch[160/469], Train loss :0.302, Train acc: 0.906
Epoch: 5, Batch[170/469], Train loss :0.416, Train acc: 0.910
Epoch: 5, Batch[180/469], Train loss :0.323, Train acc: 0.938
Epoch: 5, Batch[190/469], Train loss :0.183, Train acc: 0.953
Epoch: 5, Batch[200/469], Train loss :0.249, Train acc: 0.930
Epoch: 5, Batch[210/469], Train loss :0.178, Train acc: 0.949
Epoch: 5, Batch[220/469], Train loss :0.252, Train acc: 0.918
Epoch: 5, Batch[230/469], Train loss :0.359, Train acc: 0.906
Epoch: 5, Batch[240/469], Train loss :0.222, Train acc: 0.941
Epoch: 5, Batch[250/469], Train loss :0.260, Train acc: 0.926
Epoch: 5, Batch[260/469], Train loss :0.288, Train acc: 0.914
Epoch: 5, Batch[270/469], Train loss :0.301, Train acc: 0.930
Epoch: 5, Batch[280/469], Train loss :0.201, Train acc: 0.926
Epoch: 5, Batch[290/469], Train loss :0.287, Train acc: 0.918
Epoch: 5, Batch[300/469], Train loss :0.220, Train acc: 0.934
Epoch: 5, Batch[310/469], Train loss :0.252, Train acc: 0.926
Epoch: 5, Batch[320/469], Train loss :0.161, Train acc: 0.961
Epoch: 5, Batch[330/469], Train loss :0.314, Train acc: 0.918
Epoch: 5, Batch[340/469], Train loss :0.237, Train acc: 0.945
Epoch: 5, Batch[350/469], Train loss :0.253, Train acc: 0.957
Epoch: 5, Batch[360/469], Train loss :0.353, Train acc: 0.926
Epoch: 5, Batch[370/469], Train loss :0.326, Train acc: 0.906
Epoch: 5, Batch[380/469], Train loss :0.335, Train acc: 0.895
Epoch: 5, Batch[390/469], Train loss :0.260, Train acc: 0.922
Epoch: 5, Batch[400/469], Train loss :0.261, Train acc: 0.938
Epoch: 5, Batch[410/469], Train loss :0.313, Train acc: 0.883
Epoch: 5, Batch[420/469], Train loss :0.302, Train acc: 0.910
Epoch: 5, Batch[430/469], Train loss :0.196, Train acc: 0.926
Epoch: 5, Batch[440/469], Train loss :0.331, Train acc: 0.898
Epoch: 5, Batch[450/469], Train loss :0.209, Train acc: 0.934
Epoch: 5, Batch[460/469], Train loss :0.299, Train acc: 0.910
Epoch: 5, Train loss: 0.287, Epoch time = 130.830s
Accuracy on test 0.874, max acc on test 0.867
Epoch: 6, Batch[0/469], Train loss :0.287, Train acc: 0.934
Epoch: 6, Batch[10/469], Train loss :0.183, Train acc: 0.945
Epoch: 6, Batch[20/469], Train loss :0.333, Train acc: 0.922
Epoch: 6, Batch[30/469], Train loss :0.236, Train acc: 0.941
Epoch: 6, Batch[40/469], Train loss :0.326, Train acc: 0.902
Epoch: 6, Batch[50/469], Train loss :0.228, Train acc: 0.926
Epoch: 6, Batch[60/469], Train loss :0.228, Train acc: 0.934
Epoch: 6, Batch[70/469], Train loss :0.267, Train acc: 0.910
Epoch: 6, Batch[80/469], Train loss :0.358, Train acc: 0.887
Epoch: 6, Batch[90/469], Train loss :0.279, Train acc: 0.914
Epoch: 6, Batch[100/469], Train loss :0.262, Train acc: 0.941
Epoch: 6, Batch[110/469], Train loss :0.221, Train acc: 0.918
Epoch: 6, Batch[120/469], Train loss :0.441, Train acc: 0.895
Epoch: 6, Batch[130/469], Train loss :0.325, Train acc: 0.891
Epoch: 6, Batch[140/469], Train loss :0.294, Train acc: 0.914
Epoch: 6, Batch[150/469], Train loss :0.306, Train acc: 0.898
Epoch: 6, Batch[160/469], Train loss :0.303, Train acc: 0.910
Epoch: 6, Batch[170/469], Train loss :0.324, Train acc: 0.938
Epoch: 6, Batch[180/469], Train loss :0.331, Train acc: 0.918
Epoch: 6, Batch[190/469], Train loss :0.437, Train acc: 0.906
Epoch: 6, Batch[200/469], Train loss :0.259, Train acc: 0.914
Epoch: 6, Batch[210/469], Train loss :0.296, Train acc: 0.898
Epoch: 6, Batch[220/469], Train loss :0.220, Train acc: 0.938
Epoch: 6, Batch[230/469], Train loss :0.361, Train acc: 0.891
Epoch: 6, Batch[240/469], Train loss :0.187, Train acc: 0.938
Epoch: 6, Batch[250/469], Train loss :0.300, Train acc: 0.926
Epoch: 6, Batch[260/469], Train loss :0.262, Train acc: 0.930
Epoch: 6, Batch[270/469], Train loss :0.349, Train acc: 0.887
Epoch: 6, Batch[280/469], Train loss :0.294, Train acc: 0.906
Epoch: 6, Batch[290/469], Train loss :0.273, Train acc: 0.930
Epoch: 6, Batch[300/469], Train loss :0.332, Train acc: 0.883
Epoch: 6, Batch[310/469], Train loss :0.544, Train acc: 0.812
Epoch: 6, Batch[320/469], Train loss :0.394, Train acc: 0.902
Epoch: 6, Batch[330/469], Train loss :0.481, Train acc: 0.898
Epoch: 6, Batch[340/469], Train loss :0.348, Train acc: 0.902
Epoch: 6, Batch[350/469], Train loss :0.397, Train acc: 0.891
Epoch: 6, Batch[360/469], Train loss :0.308, Train acc: 0.910
Epoch: 6, Batch[370/469], Train loss :0.357, Train acc: 0.887
Epoch: 6, Batch[380/469], Train loss :0.249, Train acc: 0.938
Epoch: 6, Batch[390/469], Train loss :0.420, Train acc: 0.875
Epoch: 6, Batch[400/469], Train loss :0.482, Train acc: 0.887
Epoch: 6, Batch[410/469], Train loss :0.352, Train acc: 0.922
Epoch: 6, Batch[420/469], Train loss :0.225, Train acc: 0.922
Epoch: 6, Batch[430/469], Train loss :0.366, Train acc: 0.898
Epoch: 6, Batch[440/469], Train loss :0.519, Train acc: 0.844
Epoch: 6, Batch[450/469], Train loss :0.353, Train acc: 0.895
Epoch: 6, Batch[460/469], Train loss :0.354, Train acc: 0.898
Epoch: 6, Train loss: 0.317, Epoch time = 130.309s
Epoch: 7, Batch[0/469], Train loss :0.377, Train acc: 0.867
Epoch: 7, Batch[10/469], Train loss :0.335, Train acc: 0.898
Epoch: 7, Batch[20/469], Train loss :0.347, Train acc: 0.891
Epoch: 7, Batch[30/469], Train loss :0.484, Train acc: 0.863
Epoch: 7, Batch[40/469], Train loss :0.349, Train acc: 0.883
Epoch: 7, Batch[50/469], Train loss :0.403, Train acc: 0.859
Epoch: 7, Batch[60/469], Train loss :0.435, Train acc: 0.859
Epoch: 7, Batch[70/469], Train loss :0.444, Train acc: 0.871
Epoch: 7, Batch[80/469], Train loss :0.441, Train acc: 0.859
Epoch: 7, Batch[90/469], Train loss :0.307, Train acc: 0.902
Epoch: 7, Batch[100/469], Train loss :0.393, Train acc: 0.867
Epoch: 7, Batch[110/469], Train loss :0.446, Train acc: 0.867
Epoch: 7, Batch[120/469], Train loss :0.416, Train acc: 0.891
Epoch: 7, Batch[130/469], Train loss :0.462, Train acc: 0.863
Epoch: 7, Batch[140/469], Train loss :0.422, Train acc: 0.855
Epoch: 7, Batch[150/469], Train loss :0.435, Train acc: 0.852
Epoch: 7, Batch[160/469], Train loss :0.363, Train acc: 0.871
Epoch: 7, Batch[170/469], Train loss :0.271, Train acc: 0.906
Epoch: 7, Batch[180/469], Train loss :0.498, Train acc: 0.891
Epoch: 7, Batch[190/469], Train loss :0.351, Train acc: 0.898
Epoch: 7, Batch[200/469], Train loss :0.293, Train acc: 0.930
Epoch: 7, Batch[210/469], Train loss :0.298, Train acc: 0.922
Epoch: 7, Batch[220/469], Train loss :0.431, Train acc: 0.863
Epoch: 7, Batch[230/469], Train loss :0.364, Train acc: 0.891
Epoch: 7, Batch[240/469], Train loss :0.310, Train acc: 0.883
Epoch: 7, Batch[250/469], Train loss :0.389, Train acc: 0.883
Epoch: 7, Batch[260/469], Train loss :0.304, Train acc: 0.914
Epoch: 7, Batch[270/469], Train loss :0.385, Train acc: 0.891
Epoch: 7, Batch[280/469], Train loss :0.576, Train acc: 0.871
Epoch: 7, Batch[290/469], Train loss :0.616, Train acc: 0.852
Epoch: 7, Batch[300/469], Train loss :0.354, Train acc: 0.898
Epoch: 7, Batch[310/469], Train loss :0.387, Train acc: 0.887
Epoch: 7, Batch[320/469], Train loss :0.297, Train acc: 0.902
Epoch: 7, Batch[330/469], Train loss :0.355, Train acc: 0.898
Epoch: 7, Batch[340/469], Train loss :0.323, Train acc: 0.895
Epoch: 7, Batch[350/469], Train loss :0.391, Train acc: 0.895
Epoch: 7, Batch[360/469], Train loss :0.356, Train acc: 0.871
Epoch: 7, Batch[370/469], Train loss :0.350, Train acc: 0.895
Epoch: 7, Batch[380/469], Train loss :0.296, Train acc: 0.926
Epoch: 7, Batch[390/469], Train loss :0.266, Train acc: 0.898
Epoch: 7, Batch[400/469], Train loss :0.409, Train acc: 0.859
Epoch: 7, Batch[410/469], Train loss :0.575, Train acc: 0.840
Epoch: 7, Batch[420/469], Train loss :0.349, Train acc: 0.895
Epoch: 7, Batch[430/469], Train loss :0.468, Train acc: 0.844
Epoch: 7, Batch[440/469], Train loss :0.610, Train acc: 0.832
Epoch: 7, Batch[450/469], Train loss :0.424, Train acc: 0.820
Epoch: 7, Batch[460/469], Train loss :0.415, Train acc: 0.840
Epoch: 7, Train loss: 0.377, Epoch time = 130.789s
Accuracy on test 0.852, max acc on test 0.874
Epoch: 8, Batch[0/469], Train loss :0.386, Train acc: 0.875
Epoch: 8, Batch[10/469], Train loss :0.339, Train acc: 0.871
Epoch: 8, Batch[20/469], Train loss :0.385, Train acc: 0.879
Epoch: 8, Batch[30/469], Train loss :0.342, Train acc: 0.895
Epoch: 8, Batch[40/469], Train loss :0.394, Train acc: 0.898
Epoch: 8, Batch[50/469], Train loss :0.297, Train acc: 0.918
Epoch: 8, Batch[60/469], Train loss :0.446, Train acc: 0.844
Epoch: 8, Batch[70/469], Train loss :0.327, Train acc: 0.887
Epoch: 8, Batch[80/469], Train loss :0.386, Train acc: 0.863
Epoch: 8, Batch[90/469], Train loss :0.508, Train acc: 0.863
Epoch: 8, Batch[100/469], Train loss :0.444, Train acc: 0.855
Epoch: 8, Batch[110/469], Train loss :0.305, Train acc: 0.883
Epoch: 8, Batch[120/469], Train loss :0.378, Train acc: 0.891
Epoch: 8, Batch[130/469], Train loss :0.326, Train acc: 0.895
Epoch: 8, Batch[140/469], Train loss :0.363, Train acc: 0.895
Epoch: 8, Batch[150/469], Train loss :0.418, Train acc: 0.863
Epoch: 8, Batch[160/469], Train loss :0.344, Train acc: 0.895
Epoch: 8, Batch[170/469], Train loss :0.252, Train acc: 0.910
Epoch: 8, Batch[180/469], Train loss :0.400, Train acc: 0.875
Epoch: 8, Batch[190/469], Train loss :0.374, Train acc: 0.883
Epoch: 8, Batch[200/469], Train loss :0.362, Train acc: 0.887
Epoch: 8, Batch[210/469], Train loss :0.422, Train acc: 0.879
Epoch: 8, Batch[220/469], Train loss :0.404, Train acc: 0.898
Epoch: 8, Batch[230/469], Train loss :0.520, Train acc: 0.809
Epoch: 8, Batch[240/469], Train loss :0.421, Train acc: 0.883
Epoch: 8, Batch[250/469], Train loss :0.481, Train acc: 0.875
Epoch: 8, Batch[260/469], Train loss :0.359, Train acc: 0.902
Epoch: 8, Batch[270/469], Train loss :0.345, Train acc: 0.898
Epoch: 8, Batch[280/469], Train loss :0.379, Train acc: 0.887
Epoch: 8, Batch[290/469], Train loss :0.349, Train acc: 0.883
Epoch: 8, Batch[300/469], Train loss :0.507, Train acc: 0.883
Epoch: 8, Batch[310/469], Train loss :0.566, Train acc: 0.844
Epoch: 8, Batch[320/469], Train loss :0.330, Train acc: 0.883
Epoch: 8, Batch[330/469], Train loss :0.460, Train acc: 0.848
Epoch: 8, Batch[340/469], Train loss :0.338, Train acc: 0.918
Epoch: 8, Batch[350/469], Train loss :0.444, Train acc: 0.879
Epoch: 8, Batch[360/469], Train loss :0.401, Train acc: 0.883
Epoch: 8, Batch[370/469], Train loss :0.458, Train acc: 0.879
Epoch: 8, Batch[380/469], Train loss :0.395, Train acc: 0.891
Epoch: 8, Batch[390/469], Train loss :0.414, Train acc: 0.867
Epoch: 8, Batch[400/469], Train loss :0.336, Train acc: 0.898
Epoch: 8, Batch[410/469], Train loss :0.415, Train acc: 0.879
Epoch: 8, Batch[420/469], Train loss :0.245, Train acc: 0.922
Epoch: 8, Batch[430/469], Train loss :0.319, Train acc: 0.895
Epoch: 8, Batch[440/469], Train loss :0.473, Train acc: 0.844
Epoch: 8, Batch[450/469], Train loss :0.390, Train acc: 0.883
Epoch: 8, Batch[460/469], Train loss :0.272, Train acc: 0.910
Epoch: 8, Train loss: 0.382, Epoch time = 129.971s
Epoch: 9, Batch[0/469], Train loss :0.449, Train acc: 0.867
Epoch: 9, Batch[10/469], Train loss :0.433, Train acc: 0.863
Epoch: 9, Batch[20/469], Train loss :0.450, Train acc: 0.855
Epoch: 9, Batch[30/469], Train loss :0.429, Train acc: 0.844
Epoch: 9, Batch[40/469], Train loss :0.421, Train acc: 0.895
Epoch: 9, Batch[50/469], Train loss :0.335, Train acc: 0.887
Epoch: 9, Batch[60/469], Train loss :0.362, Train acc: 0.859
Epoch: 9, Batch[70/469], Train loss :0.259, Train acc: 0.914
Epoch: 9, Batch[80/469], Train loss :0.423, Train acc: 0.871
Epoch: 9, Batch[90/469], Train loss :0.400, Train acc: 0.855
Epoch: 9, Batch[100/469], Train loss :0.348, Train acc: 0.875
Epoch: 9, Batch[110/469], Train loss :0.365, Train acc: 0.887
Epoch: 9, Batch[120/469], Train loss :0.446, Train acc: 0.848
Epoch: 9, Batch[130/469], Train loss :0.725, Train acc: 0.824
Epoch: 9, Batch[140/469], Train loss :0.595, Train acc: 0.727
Epoch: 9, Batch[150/469], Train loss :0.432, Train acc: 0.844
Epoch: 9, Batch[160/469], Train loss :0.538, Train acc: 0.855
Epoch: 9, Batch[170/469], Train loss :0.390, Train acc: 0.891
Epoch: 9, Batch[180/469], Train loss :0.533, Train acc: 0.855
Epoch: 9, Batch[190/469], Train loss :0.337, Train acc: 0.887
Epoch: 9, Batch[200/469], Train loss :0.261, Train acc: 0.918
Epoch: 9, Batch[210/469], Train loss :0.531, Train acc: 0.836
Epoch: 9, Batch[220/469], Train loss :0.391, Train acc: 0.875
Epoch: 9, Batch[230/469], Train loss :0.423, Train acc: 0.887
Epoch: 9, Batch[240/469], Train loss :0.409, Train acc: 0.895
Epoch: 9, Batch[250/469], Train loss :0.361, Train acc: 0.871
Epoch: 9, Batch[260/469], Train loss :0.368, Train acc: 0.895
Epoch: 9, Batch[270/469], Train loss :0.461, Train acc: 0.836
Epoch: 9, Batch[280/469], Train loss :0.402, Train acc: 0.875
Epoch: 9, Batch[290/469], Train loss :0.448, Train acc: 0.855
Epoch: 9, Batch[300/469], Train loss :0.477, Train acc: 0.859
Epoch: 9, Batch[310/469], Train loss :0.388, Train acc: 0.883
Epoch: 9, Batch[320/469], Train loss :0.344, Train acc: 0.875
Epoch: 9, Batch[330/469], Train loss :0.464, Train acc: 0.863
Epoch: 9, Batch[340/469], Train loss :0.386, Train acc: 0.883
Epoch: 9, Batch[350/469], Train loss :0.497, Train acc: 0.824
Epoch: 9, Batch[360/469], Train loss :0.442, Train acc: 0.883
Epoch: 9, Batch[370/469], Train loss :0.445, Train acc: 0.855
Epoch: 9, Batch[380/469], Train loss :0.352, Train acc: 0.887
Epoch: 9, Batch[390/469], Train loss :0.356, Train acc: 0.875
Epoch: 9, Batch[400/469], Train loss :0.386, Train acc: 0.879
Epoch: 9, Batch[410/469], Train loss :0.372, Train acc: 0.867
Epoch: 9, Batch[420/469], Train loss :0.611, Train acc: 0.754
Epoch: 9, Batch[430/469], Train loss :0.407, Train acc: 0.863
Epoch: 9, Batch[440/469], Train loss :0.501, Train acc: 0.844
Epoch: 9, Batch[450/469], Train loss :0.445, Train acc: 0.848
Epoch: 9, Batch[460/469], Train loss :0.366, Train acc: 0.879
Epoch: 9, Train loss: 0.421, Epoch time = 129.786s
Accuracy on test 0.789, max acc on test 0.874
```

训练10个epoch之后，在测试集上可以达到在最大准确率为`0.874`。

