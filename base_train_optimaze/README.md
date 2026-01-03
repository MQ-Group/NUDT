# 车辆识别场景

## 概述
本项目基于 adversarial-attacks-pytorch 库实现样本攻击和防御场景。它支持对抗样本生成、攻击评估、防御训练、对抗样本检测。


## 环境变量：
* `input_path`（str 必填）: 指定输入路径，在此路径下有权重文件和数据集文件。
* `output_path`（str 必填）: 指定输出路径，在此路径下保存生成的对抗样本和防御训练的权重。
* `process`（str 必填）: 指定进程名称，支持枚举值:`train`, `test`, `fit`。
* ~~`model`（str 必填）: 指定模型名称，支持枚举值:`vgg`, `resnet`, `inception`, `lenet` 。~~
* ~~`data`（str 必填）: 指定数据集，支持枚举值:`cifar10`, `cifar100`, `imagenet`, `mnist`。~~
* ~~`class_number`（int 必填）: 指定目标类别数量，与数据集绑定，`cifar10`: `10`, `cifar100`: `100`, `imagenet`: `1000`, `mnist`: `10`。~~
* `resume_from_checkpoint`（bool 选填，默认为False）：是否在已有权重的基础上进行训练，若`process`为`train`或`fit`时有效。
* `epochs`（int 选填，默认为`100`）：训练迭代次数，若`process`为`train`或`fit`时有效。
* `batch`（int 选填，默认为`16`）：训练批处理大小。
* `device`（str 选填，默认为`cpu`）：使用cpu或cuda，支持枚举值（第一个为默认值）:`cpu`, `cuda`。
* `workers`（int 选填，默认为`0`）：加载数据集时workers的数量。
* `optimizer`（str 选填，默认为adam）：优化器类型，支持枚举值：`sgd`, `adam`, `adamw`, `rmsprop`, `adagrad`, `adadelta`, `adamax`, `nadam`, `radam`, `lbfgs`, `sgd_nesterov`, `asgd`, `rprop`。
* `scheduler`（str 选填，默认为steplr）：学习率调度器类型，支持枚举值：`steplr`, `multisteplr`, `exponential`, `cosine`, `cyclic`, `onecycle`, `lambda`, `cosine_warm`。
* `loss_function`（str 选填，默认为cross_entropy）：损失函数类型，支持枚举值：`cross_entropy`, `mse`, `l1`, `binary_cross_entropy`。
* `lr`（float 选填，默认为0.001）：学习率。
* `weight_decay`（float 选填，默认为0.0）：权重衰减`（L2惩罚项）。
* `momentum`（float 选填，默认为0.9）：动量因子。
* `betas`（tuple 选填，默认为(0.9, 0.999)）：Adam/AdamW/Adamax/NAdam优化器的beta参数。
* `eps`（float 选填，默认为1e-8）：数值稳定性的epsilon值。
* `amsgrad`（bool 选填）：是否使用Adam的AMSGrad变体。
* `dampening`（float 选填，默认为0）：动量阻尼因子。
* `nesterov`（bool 选填）：是否启用Nesterov动量。
* `alpha`（float 选填，默认为0.99）：RMSprop的平滑常数。
* `centered`（bool 选填）：是否计算中心化的RMSProp。
* `rho`（float 选填，默认为0.9）：计算运行平均值的系数`（Adadelta）。
* `max_iter`（int 选填，默认为20）：LBFGS每次优化步骤的最大迭代次数。
* `history_size`（int 选填，默认为100）：LBFGS的更新历史大小。
* `lambd`（float 选填，默认为1e-4）：ASGD的衰减项。
* `alpha_asgd`（float 选填，默认为0.75）：ASGD中eta更新的幂。
* `t0`（float 选填，默认为1e6）：ASGD开始平均的起始点。
* `lr_decay_rate`（float 选填，默认为0.1）：学习率衰减率。
* `lr_decay_step`（int 选填，默认为10）：学习率衰减步长。
* `lr_decay_min_lr`（float 选填，默认为1e-6）：最小学习率。
* `max_epochs`（int 选填，默认为100）：最大训练轮数。
* `milestones`（list 选填，默认为[30, 60, 90]）：MultiStepLR的里程碑列表。
* `T_max`（int 选填，默认为50）：CosineAnnealingLR的`T_max`参数。
* `plateau_mode`（str 选填，默认为min）：ReduceLROnPlateau的模式，支持枚举值：`min`、`max`。
* `patience`（int 选填，默认为5）：ReduceLROnPlateau的耐心值。
* `threshold`（float 选填，默认为1e-4）：ReduceLROnPlateau的阈值。
* `T_0`（int 选填，默认为10）：CosineAnnealingWarmRestarts的`T_0`参数。
* `T_mult`（int 选填，默认为2）：CosineAnnealingWarmRestarts的`T_mult`参数。
* `max_lr`（float 选填，默认为0.01）：OneCycleLR的最大学习率。
* `total_steps`（int 选填，默认为10000）：OneCycleLR的总步数。
* `steps_per_epoch`（int 选填，默认为100）：OneCycleLR的每轮步数。
* `pct_start`（float 选填，默认为0.3）：OneCycleLR的上升阶段百分比。
* `anneal_strategy`（str 选填，默认为cos）：OneCycleLR的退火策略，支持枚举值：`cos`、`linear`。


### 说明
-- `process`为`train`，训练 \
-- `process`为`test`测试 \
-- `process`为`fit`， 边训练边测试

### optimizer的类型与参数对应关系
| 参数 | sgd | adam | adamw | rmsprop | adagrad | adadelta | adamax | nadam | radam | lbfgs | sgd_nesterov | asgd | rprop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lr | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| weight_decay | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | |
| momentum | 1 | | | 1 | | | | | | | 1 | | |
| dampening | 1 | | | | | | | | | | | | |
| nesterov | 1 | | | | | | | | | | 1 | | |
| betas | | 1 | 1 | | | | 1 | 1 | 1 | | | | |
| eps | | 1 | 1 | 1 | | 1 | 1 | 1 | 1 | | | | |
| amsgrad | | 1 | 1 | | | | | | | | | | |
| alpha | | | | 1 | | | | | | | | 1 | |
| centered | | | | 1 | | | | | | | | | |
| lr_decay | | | | | 1 | | | | | | | | |
| initial_accumulator_value | | | | | 1 | | | | | | | | |
| rho | | | | | | 1 | | | | | | | |
| momentum_decay | | | | | | | | 1 | | | | | |
| max_iter | | | | | | | | | | 1 | | | |
| max_eval | | | | | | | | | | 1 | | | |
| tolerance_grad | | | | | | | | | | 1 | | | |
| tolerance_change | | | | | | | | | | 1 | | | |
| history_size | | | | | | | | | | 1 | | | |
| line_search_fn | | | | | | | | | | 1 | | | |
| lambd | | | | | | | | | | | | 1 | |
| t0 | | | | | | | | | | | | 1 | |
| etas | | | | | | | | | | | | | 1 |
| step_sizes | | | | | | | | | | | | | 1 |


### scheduler的类型与参数对应关系
| 参数 | steplr | multisteplr | exponential | cosine | cyclic | onecycle | lambda | cosine_warm |
|------|--------|-------------|-------------|--------|--------|----------|--------|-------------|
| lr_decay_step | 1 | | | | | | | |
| lr_decay_rate | 1 | 1 | 1 | | | | | |
| milestones | | 1 | | | | | | |
| T_max | | | | 1 | | | | |
| max_epochs | | | | 1 | | 1 | | |
| lr_decay_min_lr | | | | 1 | | | | 1 |
| base_lr | | | | | 1 | | | |
| max_lr | | | | | 1 | 1 | | |
| step_size_up | | | | | 1 | | | |
| step_size_down | | | | | 1 | | | |
| cyclic_mode | | | | | 1 | | | |
| cyclic_gamma | | | | | 1 | | | |
| total_steps | | | | | | 1 | | |
| steps_per_epoch | | | | | | 1 | | |
| epochs | | | | | | 1 | | |
| pct_start | | | | | | 1 | | |
| anneal_strategy | | | | | | 1 | | |
| lr_lambda | | | | | | | 1 | |
| T_0 | | | | | | | | 1 |
| T_mult | | | | | | | | 1 |

## 快速开始

### 构建 Docker 镜像
```bash
cd base_train_optimaze
docker build -t base_train_optimaze:latest .
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
    -e resume_from_checkpoint=False \
    -e epochs=100 \
    -e batch=16 \
    -e device=cuda \
    -e workers=0 \
    base_train_optimaze:latest
```

