# 车辆识别场景

## 概述
本项目基于 adversarial-attacks-pytorch 库实现样本攻击和防御场景。它支持对抗样本生成、攻击评估、防御训练、对抗样本检测。


## 环境变量：
* `input_path`（str 必填）: 指定输入路径，在此路径下有权重文件和数据集文件。
* `output_path`（str 必填）: 指定输出路径，在此路径下保存生成的对抗样本和防御训练的权重。
* `process`（str 必填）: 指定进程名称，支持枚举值:`adv`, `attack`, `defend`, `detect`~~, `test`, `sample`~~。
* ~~`model`（str 必填）: 指定模型名称，支持枚举值:`yolov5`, `yolov8`, `yolov10` 。~~
* ~~`data`（str 必填）: 指定数据集，支持枚举值:`kitti`, `bdd100k`, `ua-detrac`, `dawn`, `special_vehicle`, `flir_adas`。~~
* ~~`class_number`（int 必填）: 指定目标类别数量，与数据集绑定，对于kitti数据集为`8` 。~~
* `attack_method`（str 选填，默认为`fgsm`）: 指定攻击方法，若`process`为`adv`或`attack`则必填，支持枚举值（第一个为默认值）: `fgsm`, `pgd`, `bim`, `cw`, `deepfool`, `gn`, `jitter`, `yopo`, `pgdrs`, `trades`, `free`, `fast`。
* `defend_method`（str 选填，默认为`yopo`）: 指定防御方法，若`process`为`defend`则必填，支持枚举值（第一个为默认值）:`yopo`, `pgdrs`, `trades`, `free`, `fast`, `fgsm`, `pgd`, `bim`, `cw`, `deepfool`, `gn`, `jitter`。
* `detect_method`（str 选填，默认为`spatial_smoothing`）: 指定防御方法，若`process`为`detect`则必填，支持枚举值（第一个为默认值）:`spatial_smoothing`, `feature_squeezing`, `local_intrinsic_dimensionality`。
* ~~`epochs`（int 选填，默认为`100`）：训练迭代次数，若`process`为`train`时有效。~~
* `batch`（int 选填，默认为`16`）：训练批处理大小，若`process`为`train`或`test`时有效。
* `device`（str 选填，默认为`cpu`）：使用cpu或cuda，支持枚举值（第一个为默认值）:`cpu`, `cuda`。
* `workers`（int 选填，默认为`0`）：加载数据集时workers的数量。
* `selected_samples`（int 选填，默认为64，0表示数据集全部样本数据）: 若`process`为`adv`或`detect`时有效，使用的样本数。
* `epsilon`（float 选填，默认为`8/255`）：扰动强度参数，控制对抗扰动大小。
* `step_size`（float 选填，默认为`2/255`）：步长，迭代攻击的更新幅度。
* `max_iterations`（int 选填，默认为`50`）：最大迭代次数。
* `random_start`（bool 选填，默认为`False`）：是否随机初始化扰动，支持枚举值（第一个为默认值）:`False`, `True`。
* ~~`loss_function`（str 选填，默认为`cross_entropy`）：损失函数类型，支持枚举值（第一个为默认值）:`cross_entropy`, `mse`, `l1`, `binary_cross_entropy`。~~
* ~~`optimization_method`（str 选填，默认为`adam`）：优化方法，支持枚举值（第一个为默认值）:`adam`, `sgd`。~~
* `lr`（float 选填，默认为`0.001`）：优化器学习率。
* ~~`scaling_factor`（float 选填，默认为`0.9`）：缩放因子。~~
* ~~`interpolate_method`（str 选填，默认为`bilinear`）：插值方法，支持枚举值（第一个为默认值）:`bilinear`, `nearest`。~~
* ~~`image_quality`（int 选填，默认为`90`）：图像质量。~~
* ~~`filter_kernel_size`（int 选填，默认为`3`）：滤波器核大小。~~
* `scale`（int 选填，默认为`10`）：尺度。
* `std`（float 选填，默认为`0.1`）：标准差。
* `noise_type`str 选填，默认为`guassian`）：噪声类型，支持枚举值（第一个为默认值）:`guassian`, `uniform`。
* `noise_sd`（float 选填，默认为`0.5`）：噪声标准差。
* `kernel_size`（int 选填，默认为`3`）：滤波器核大小。
* `bit_depth`（int 选填，默认为`4`）：比特深度。
* `k_nearest`（int 选填，默认为`20`）：`process`为`detect`且`detect_method`为`local_intrinsic_dimensionality`时有效，注意该值不能大于`selected_samples`。
* `detection_threshold`（float 选填，默认为`0.5`）：样本检测阈值，样本检测方法结果若大于该值表示检测样本为对抗样本，`process`为`detect`时有效。
* `max_queries`（int 选填，默认为10）：最大查询次数，即攻击向模型发送的最大查询数限制。
* `binary_search_steps`（int 选填，默认均为10）：二分搜索的迭代次数，用于调整约束参数。
* `norm`（str 选填，HSJA默认为L2，NES默认为Linf）：距离度量范数，支持枚举值：`L2`、`Linf`。

### 说明
-- `process`为`adv`，使用原始cifar10数据集，生成对抗样本保存为.dat文件做为对抗样本数据集，其中也包含原始样本 \
-- `process`为`attack`，只能使用生成对抗样本数据集 \
-- `process`为`defend`，只能使用原始cifar10数据集 \
-- `process`为`detect`，只能使用原始cifar10数据集

### 攻击防御检测方法的有效参数
ss == spatial_smoothing \
fs == feature_squeezing \
lid = local_intrinsic_dimensionality

| 参数 | fgsm | pgd | bim | cw | deepfool | gn | jitter | boundary | zoo | hsja | nes | yopo | pgdrs | trades | free | fast | ss | fs | lid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| epsilon | 1 | 1 | 1 | | | | 1 | 1 | | | 1 | 1 | 1 | 1 | 1 | 1 | | | |
| step_size | | 1 | 1 | | | | 1 | 1 | | | | | 1 | 1 | 1 | | | | |
| max_iterations | | 1 | 1 | 1 | 1 | | 1 | | 1 | 1 | | | 1 | 1 | 1 | | | | |
| random_start | | 1 | | | | | 1 | | | | | | | | | | | | |
| lr | | | | 1 | | | | | 1 | | 1 | | | | | | | | |
| std | | | | | | 1 | 1 | | | | | | | | | | | | |
| scale | | | | | | | 1 | | | | | | | | | | | | |
| max_queries | | | | | | | | 1 | | 1 | 1 | | | | | | | | |
| binary_search_steps | | | | | | | | 1 | 1 | | | | | | | | | | |
| norm | | | | | | | | | | 1 | 1 | | | | | | | | |
| noise_type | | | | | | | | | | | | | 1 | | | | | | |
| noise_sd | | | | | | | | | | | | | 1 | | | | | | |
| kernel_size | | | | | | | | | | | | | | | | | 1 | 1 | |
| bit_depth | | | | | | | | | | | | | | | | | | 1 | |
| k_nearest | | | | | | | | | | | | | | | | | | | 1 |
| detection_threshold | | | | | | | | | | | | | | | | | 1 | 1 | 1 |


## 快速开始

### 构建 Docker 镜像
```bash
cd sample_attack_defend
docker build -t sample_attack_defend:latest .
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
    -e defend_method=yopo \
    -e detect_method=spatial_smoothing \
    -e epochs=100 \
    -e batch=16 \
    -e device=cuda \
    -e workers=0 \
    -e selected_samples=64 \
    -e epsilon=0.0313 \
    -e step_size=0.0078 \
    -e max_iterations=50 \
    -e random_start=False \
    -e lr=0.001 \
    -e scale=10 \
    -e std=0.1 \
    -e noise_type=guassian \
    -e noise_sd=0.5 \
    -e kernel_size=3 \       
    -e bit_depth=4 \
    -e k_nearest= 20 \
    -e detection_threshold=0.5 \
    drone_yolo:latest
```

