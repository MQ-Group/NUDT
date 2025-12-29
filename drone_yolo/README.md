# 无人机识别场景

## 概述
本项目基于 ultralytics 库实现无人机识别场景下基于drone_yolo的无人机识别任务。它支持对抗样本生成、攻击评估、防御机制和模型训练。


## 环境变量：
* `input_path`（必填）: 指定输入路径，在此路径下有权重文件和数据集文件。
* `output_path`（必填）: 指定输出路径，在此路径下保存生成的对抗样本和防御训练的权重。
* `process`（必填）: 指定进程名称，支持枚举值（第一个为默认值）:`adv`, `attack`, `defend`, `train`, `test`, `sample`。
* ~~`model`（必填）: 指定模型名称，支持枚举值:`drone_yolo`。~~
* ~~`data`（必填）: 指定数据集，支持枚举值:`drone`, `drone type`, `yolo drone detection`, `foggy_drone`。~~
* ~~`class_number`（必填）: 指定目标类别数量为1 。~~
* `attack_method`（选填，默认为`fgsm`）: 指定攻击方法，若`process`为`adv`或`attack`则必填，支持枚举值（第一个为默认值）: `fgsm`, `pgd`, `bim`, `cw`, `deepfool`。
* `defend_method`（选填，默认为`scale`）: 指定防御方法，若`process`为`defend`则必填，支持枚举值（第一个为默认值）:`scale`, `compression`, `fgsm_denoise`, `neural_cleanse`, `pgd_purifier`。
`epochs`（选填，默认为`100`）：训练迭代次数，若`process`为`train`时有效。
* `batch`（选填，默认为`16`）：训练批处理大小，若`process`为`train`或`test`时有效。
* `device`（选填，默认为`cpu`）：使用cpu或cuda，支持枚举值（第一个为默认值）:`cpu`, `cuda`。
* `workers`（选填，默认为`0`）：加载数据集时workers的数量。
* `selected_samples`（选填，默认为0表示数据集全部样本数据）: 若`process`为`adv`时有效，生成对抗样本时使用的样本数。
* `epsilon`（选填，默认为`8/255`）：扰动强度参数，控制对抗扰动大小。
* `step_size`（选填，默认为`2/255`）：步长，迭代攻击的更新幅度。
* `max_iterations`（选填，默认为`50`）：最大迭代次数。
* `random_start`（选填，默认为`False`）：是否随机初始化扰动，支持枚举值（第一个为默认值）:`False`, `True`。
* `loss_function`（选填，默认为`cross_entropy`）：损失函数类型，支持枚举值（第一个为默认值）:`cross_entropy`, `mse`, `l1`, `binary_cross_entropy`。
* `optimization_method`（选填，默认为`adam`）：优化方法，支持枚举值（第一个为默认值）:`adam`, `sgd`。
* `scaling_factor`（选填，默认为0.9）：缩放因子。
* `interpolate_method`（选填，默认为`bilinear`）：插值方法，支持枚举值（第一个为默认值）:`bilinear`, `nearest`。
* `image_quality`（选填，默认为90）：图像质量。
* `filter_kernel_size`（选填，默认为3）：滤波器核大小。


### 攻击防御方法的有效参数
|  | fgsm | pgd | bim | cw | deepfool | scale | compression | fgsm_denoise | neural_cleanse | pgd_purifier |
|---|---|---|---|---|---|---|---|---|---|---|
| epsilon | 1 | 1 | 1 |  |  |  |  | 1 |  | 1 |
| step_size |  | 1 | 1 |  |  |  |  |  |  | 1 |
| max_iterations |  | 1 | 1 | 1 |  |  |  |  |  | 1 |
| random_start |  | 1 |  |  |  |  |  |  |  |  |
| loss_function | 1 | 1 | 1 |  |  |  |  |  |  |  |
| optimization_method |  |  |  | 1 |  |  |  |  |  |  |
| lr |  |  |  | 1 |  |  |  |  |  |  |
| scaling_factor |  |  |  |  |  | 1 |  |  |  |  |
| interpolate_method |  |  |  |  |  | 1 |  |  |  |  |
| image_quality |  |  |  |  |  |  | 1 |  |  |  |
| filter_kernel_size |  |  |  |  |  |  |  |  | 1 |  |

## 快速开始

### 构建 Docker 镜像
```bash
cd drone_yolo
docker build -t drone_yolo:latest .
```

## 输入目录结构
```
input/
├── model/
│   └── model_name/                 # 模型目录
│       └── weight.pt               # 模型权重
│       └── model_cfg.yaml          # 模型配置文件
└── data/
    └── data_name/                  # 数据集目录
        └── data/                   # 数据集
        └── data_cfg.yaml           # 数据集配置文件
```




论文：https://www.mdpi.com/2504-446X/7/8/526

主要改进：\
1.加P2层（主要提升）；\
2.主干网络下采样更换成RepVGG结构；\
3.颈部自底向上concat多融合一个上层特征（几乎无提升）
