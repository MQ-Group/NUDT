# 车辆识别场景

## 概述
本项目基于 ultralytics 库实现车辆识别场景下yolov5, yolov8, yolov10的车辆识别任务。它支持对抗样本生成、攻击评估、防御机制和模型训练。

## 环境变量：
* `input_path`（必填）: 指定输入路径，在此路径下有权重文件和数据集文件。
* `output_path`（必填）: 指定输出路径，在此路径下保存生成的对抗样本和防御训练的权重。
* `process`（必填）: 指定进程名称，支持枚举值（第一个为默认值）:`adv`, `attack`, `defend`, `train`。
* ~~`model`（必填）: 指定模型名称，支持枚举值:`yolov5`, `yolov8`, `yolov10` 。~~
* ~~`data`（必填）: 指定数据集，支持枚举值:`kitti`, `bdd100k`, `ua-detrac`, `dawn`, `special_vehicle`。~~
* ~~`class_number`（必填）: 指定目标类别数量，与数据集绑定，对于kitti数据集为`8` 。~~
* `attack_method`（选填）: 指定攻击方法，若`process`为`adv`或`attack`则必填，支持枚举值（第一个为默认值）: `cw`, `deepfool`, `bim`, `fgsm`, `pgd`。
* `defend_method`（选填）: 指定防御方法，若`process`为`defend`则必填，支持枚举值（第一个为默认值）:`scale`, `compression`, `fgsm_denoise`, `neural_cleanse`, `pgd_purifier`。
* `epochs`（选填，默认为`100`）：训练迭代次数，若`process`为`train`时有效。
* `batch`（选填，默认为`16`）：训练批处理大小，若`process`为`train`时有效。
* `device`（选填，默认为`0`）：使用哪个gpu。
* `workers`（选填，默认为`0`）：加载数据集时workers的数量。
* `selected_samples`（选填，默认为0表示数据集全部样本数据）: 若`process`为`adv`时有效，生成对抗样本时使用的样本数。
* `epsilon`（选填，默认为`8/255`）：扰动强度参数，控制对抗扰动大小。
* `step_size`（选填，默认为`2/255`）：步长，迭代攻击的更新幅度。
* `max_iterations`（选填，默认为`50`）：最大迭代次数。
* `random_start`（选填，默认为`False`）：是否随机初始化扰动。
* `loss_function`（选填）：损失函数类型，支持枚举值（第一个为默认值）:`cross_entropy`, `mse`, `l1`, `binary_cross_entropy`。
* `optimization_method`（选填）：优化方法，`attack_method`为`bim`时有效，支持枚举值（第一个为默认值）:`adam`, `sgd`。



## 快速开始

### 构建 Docker 镜像
```bash
cd vehicle_yolo
docker build -t vehicle_yolo:latest .
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



