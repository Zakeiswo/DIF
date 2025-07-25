# DIF: Depth Injection Framework for RGBD Salient Object Detection

[English](#english) | [中文](#chinese)

<a name="english"></a>
## English

This is the official implementation of our TIP 2023 paper "Depth Injection Framework for RGBD Salient Object Detection" ([IEEE Xplore](https://ieeexplore.ieee.org/document/10258039)).

### Project Structure

- `models/`: Contains model definitions and network architectures
- `data/`: Contains data loading and processing code
- `metric/`: Contains implementations of evaluation metrics
- `utils/`: Contains utility functions and configuration loading
- `configs/`: Contains configuration files

### Usage

#### Training

```bash
python train.py --config configs/default.yaml
```

#### Testing

Test on one or multiple datasets and calculate evaluation metrics:

```bash
python test.py --config configs/default.yaml --checkpoint path/to/your/model_checkpoint.pth
```

#### Exporting Saliency Maps

Export saliency maps to a specified directory:

```bash
python export_saliency.py --checkpoint path/to/your/model_checkpoint.pth --output-dir ./results
```

Export saliency maps and evaluate performance simultaneously:

```bash
python export_saliency.py --checkpoint path/to/your/model_checkpoint.pth --output-dir ./results --evaluate
```

#### Evaluating Saved Saliency Maps

Evaluate previously saved saliency maps:

```bash
python evaluate_saliency.py --pred-dir ./results/dataset_name --dataset-name dataset_name
```

If the saliency map directory name is the same as the dataset name, you can simplify to:

```bash
python evaluate_saliency.py --pred-dir ./results/dataset_name
```

### Evaluation Metrics

This project supports the following evaluation metrics:

- F-measure
- MAE (Mean Absolute Error)
- S-measure (Structure measure)
- E-measure (Enhanced alignment measure)
- WFM (Weighted F-measure)

### Datasets

This project supports the following RGB-D salient object detection datasets:

- DUT-RGBD
- NLPR
- NJU2K
- STEREO1000
- STEREO(797)
- DES(RGBD135)
- LFSD
- SIP

#### Saliency Maps

Download our pre-computed saliency maps:
- **File**: Saliency_Maps.zip
- **Link**: https://pan.baidu.com/s/1JJ9XtxO5fIZNE_lBVptNmQ 
- **Extraction Code**: 5vjp

#### Test Set Release

You can download the test set used in our experiments:
- **File**: TestSet.zip
- **Link**: https://pan.baidu.com/s/1UQ66rs0GiT2i7uNH8XP7qg
- **Extraction Code**: 5h3r

### Pre-trained Models and Results

Our model was trained on the combined dataset of DUT-RGBD, NJU2K, and NLPR.

#### Pre-trained Model

Download our best checkpoint:
- **File**: best_epoch.pth
- **Link**: https://pan.baidu.com/s/1tRnhJZNTzLVFzXcbhM8Dyw 
- **Extraction Code**: syh5

### Requirements

- Python 3.7+
- PyTorch 1.7+
- CUDA 10.2+
- NumPy
- Pillow
- tqdm
- PyYAML

### Acknowledgements

We would like to thank the authors of the following repositories for their excellent work and code:

- [HDFNet](https://github.com/lartpang/HDFNet)
- [A2dele](https://github.com/DUT-IIAU-OIP-Lab/CVPR2020-A2dele)

### Citation

If you find this work useful for your research, please consider citing our paper:

```bibtex
@article{yao2023depth,
  title={Depth injection framework for RGBD salient object detection},
  author={Yao, Shunyu and Zhang, Miao and Piao, Yongri and Qiu, Chaoyi and Lu, Huchuan},
  journal={IEEE Transactions on Image Processing},
  volume={32},
  pages={5340--5352},
  year={2023},
  publisher={IEEE}
}
```

### Contact and Questions

Contact: Shunyu Yao  
Email: yao_shunyu@foxmail.com or ysyfeverfew@mail.dlut.edu.cn

--------

<a name="chinese"></a>
## 中文

这是我们发表在 TIP 2023 的论文 "Depth Injection Framework for RGBD Salient Object Detection" 的官方实现（[IEEE Xplore](https://ieeexplore.ieee.org/document/10258039)）。

### 项目结构

- `models/`: 包含模型定义和网络结构
- `data/`: 包含数据加载和处理相关代码
- `metric/`: 包含评估指标的实现
- `utils/`: 包含工具函数和配置加载
- `configs/`: 包含配置文件

### 使用方法

#### 训练模型

```bash
python train.py --config configs/default.yaml
```

#### 测试模型

测试单个或多个数据集并计算评估指标：

```bash
python test.py --config configs/default.yaml --checkpoint path/to/your/model_checkpoint.pth
```

#### 导出显著图

导出显著图到指定目录：

```bash
python export_saliency.py --checkpoint path/to/your/model_checkpoint.pth --output-dir ./results
```

导出显著图并同时评估性能：

```bash
python export_saliency.py --checkpoint path/to/your/model_checkpoint.pth --output-dir ./results --evaluate
```

#### 评估已保存的显著图

对已保存的显著图进行评估：

```bash
python evaluate_saliency.py --pred-dir ./results/dataset_name --dataset-name dataset_name
```

如果显著图目录名与数据集名称相同，可以简化为：

```bash
python evaluate_saliency.py --pred-dir ./results/dataset_name
```

### 评估指标

本项目支持以下评估指标：

- F-measure (F 测度)
- MAE (平均绝对误差)
- S-measure (结构测度)
- E-measure (增强对齐测度)
- WFM (加权 F 测度)

### 数据集

本项目支持以下RGB-D显著性目标检测数据集：

- DUT-RGBD
- NLPR
- NJU2K
- STEREO1000
- STEREO(797)
- DES(RGBD135)
- LFSD
- SIP

#### 显著图

下载我们预先计算的显著图：
- **文件**：Saliency_Maps.zip
- **链接**：https://pan.baidu.com/s/1JJ9XtxO5fIZNE_lBVptNmQ 
- **提取码**：5vjp

#### 测试集公布

您可以下载我们实验所用的测试集：
- **文件**：TestSet.zip
- **链接**：https://pan.baidu.com/s/1UQ66rs0GiT2i7uNH8XP7qg
- **提取码**：5h3r

### 预训练模型与结果

我们的模型在 DUT-RGBD、NJU2K 和 NLPR 的组合数据集上进行训练。

#### 预训练模型

下载我们的最佳检查点：
- **文件**：best_epoch.pth
- **链接**：https://pan.baidu.com/s/1tRnhJZNTzLVFzXcbhM8Dyw 
- **提取码**：syh5


### 环境要求

- Python 3.7+
- PyTorch 1.7+
- CUDA 10.2+
- NumPy
- Pillow
- tqdm
- PyYAML

### 致谢

我们感谢以下代码库的作者提供的优秀工作和代码：

- [HDFNet](https://github.com/lartpang/HDFNet)
- [A2dele](https://github.com/DUT-IIAU-OIP-Lab/CVPR2020-A2dele)

### 引用

如果您发现我们的工作对您的研究有用，请考虑引用我们的论文：

```bibtex
@article{yao2023depth,
  title={Depth injection framework for RGBD salient object detection},
  author={Yao, Shunyu and Zhang, Miao and Piao, Yongri and Qiu, Chaoyi and Lu, Huchuan},
  journal={IEEE Transactions on Image Processing},
  volume={32},
  pages={5340--5352},
  year={2023},
  publisher={IEEE}
}
```

### 联系方式

联系人：姚舜禹  
邮箱：yao_shunyu@foxmail.com 或 ysyfeverfew@mail.dlut.edu.cn
