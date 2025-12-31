# 无人机识别场景

## 概述
本项目基于 ultralytics 库实现无人机识别场景下基于drone_yolo的无人机识别任务。它支持对抗样本生成、攻击评估、防御机制和模型训练。


## 环境变量：
* `input_path`（str 必填）: 指定输入路径，在此路径下有权重文件和数据集文件。
* `output_path`（str 必填）: 指定输出路径，在此路径下保存生成的对抗样本和防御训练的权重。
* `process`（str 必填）: 指定进程名称，支持枚举值:`adv`, `attack`, `defend`, `train`, `test`, `sample`。
* ~~`model`（str 必填）: 指定模型名称，支持枚举值:`drone_yolo`。~~
* ~~`data`（str 必填）: 指定数据集，支持枚举值:`drone`, `drone type`, `yolo drone detection`, `foggy_drone`。~~
* ~~`class_number`（int 必填）: 指定目标类别数量为1 。~~
* `attack_method`（str 选填，默认为`fgsm`）: 指定攻击方法，若`process`为`adv`或`attack`则必填，支持枚举值（第一个为默认值）: `fgsm`, `pgd`, `bim`, `cw`, `deepfool`。
* `defend_method`（str 选填，默认为`scale`）: 指定防御方法，若`process`为`defend`则必填，支持枚举值（第一个为默认值）:`scale`, `compression`, `fgsm_denoise`, `neural_cleanse`, `pgd_purifier`。
* `epochs`（int 选填，默认为`100`）：训练迭代次数，若`process`为`train`时有效。
* `batch`（int 选填，默认为`16`）：训练批处理大小，若`process`为`train`或`test`时有效。
* `device`（str 选填，默认为`cpu`）：使用cpu或cuda，支持枚举值（第一个为默认值）:`cpu`, `cuda`。
* `workers`（int 选填，默认为`0`）：加载数据集时workers的数量。
* `selected_samples`（int 选填，默认为64，0表示数据集全部样本数据）: 若`process`为`adv`时有效，生成对抗样本时使用的样本数。
* `epsilon`（float 选填，默认为`8/255`）：扰动强度参数，控制对抗扰动大小。
* `step_size`（float 选填，默认为`2/255`）：步长，迭代攻击的更新幅度。
* `max_iterations`（int 选填，默认为`50`）：最大迭代次数。
* `random_start`（bool 选填，默认为`False`）：是否随机初始化扰动，支持枚举值（第一个为默认值）:`False`, `True`。
* `loss_function`（str 选填，默认为`cross_entropy`）：损失函数类型，支持枚举值（第一个为默认值）:`cross_entropy`, `mse`, `l1`, `binary_cross_entropy`。
* `optimization_method`（str 选填，默认为`adam`）：优化方法，支持枚举值（第一个为默认值）:`adam`, `sgd`。
* `lr`（float 选填，默认为`0.001`）：优化器学习率。
* `scaling_factor`（float 选填，默认为`0.9`）：缩放因子。
* `interpolate_method`（str 选填，默认为`bilinear`）：插值方法，支持枚举值（第一个为默认值）:`bilinear`, `nearest`。
* `image_quality`（int 选填，默认为`90`）：图像质量。
* `filter_kernel_size`（int 选填，默认为`3`）：滤波器核大小。


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

### 输入输出目录结构
```
docker_inout_dir/
├── input/
│   ├── model/
│   │   └── model_name/                 # 模型目录
│   │       └── weight.pt               # 模型权重
│   │       └── model_cfg.yaml          # 模型配置文件
│   └── data/
│       └── data_name/                  # 数据集目录
│           └── data/                   # 数据集
│           └── data_cfg.yaml           # 数据集配置文件
├── output/
```

### 运行 Docker 镜像
```bash
cd docker_inout_dir

docker run --rm \
    -v /data6/user23215430/nudt/drone_yolo/input:/project/input:ro \
    -v /data6/user23215430/nudt/drone_yolo/output:/project/output:rw \
    -e INPUT_PATH=/project/input \
    -e OUTPUT_PATH=/project/output \
    -e process=train \
    -e attack_method=fgsm \
    -e defend_method=scale \
    -e epochs=100 \
    -e batch=16 \
    -e device=cuda \
    -e workers=0 \
    -e selected_samples=64 \
    -e epsilon=0.0313 \
    -e step_size=0.0078 \
    -e max_iterations=50 \
    -e random_start=False \
    -e loss_function=cross_entropy \
    -e optimization_method=adam \
    -e lr=0.001 \
    -e scaling_factor=0.9 \
    -e interpolate_method=bilinear \
    -e image_quality=90 \
    -e filter_kernel_size=3 \
    drone_yolo:latest
```

## 基于YOLOv8改进1-Drone-YOLO复现
### 博客
https://blog.csdn.net/weixin_45679938/article/details/139077352

### 论文
https://www.mdpi.com/2504-446X/7/8/526

### 主要改进
1.加P2层（主要提升）；\
2.主干网络下采样更换成RepVGG结构；\
3.颈部自底向上concat多融合一个上层特征（几乎无提升）
