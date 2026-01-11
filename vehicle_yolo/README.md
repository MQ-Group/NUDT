# 车辆识别场景

## 概述
本项目基于 ultralytics 库实现车辆识别场景下yolov5, yolov8, yolov10的车辆识别任务。它支持对抗样本生成、攻击评估、防御机制和模型训练。

## 环境变量：
* `input_path`（str 必填）: 指定输入路径，在此路径下有权重文件和数据集文件。
* `output_path`（str 必填）: 指定输出路径，在此路径下保存生成的对抗样本和防御训练的权重。
* `process`（str 必填）: 指定进程名称，支持枚举值:`adv`, `attack`, `defend`, `train`, `test`, `predict`, `sample`。
* ~~`model`（str 必填）: 指定模型名称，支持枚举值:`yolov5`, `yolov8`, `yolov10` 。~~
* ~~`data`（str 必填）: 指定数据集，支持枚举值:`kitti`, `bdd100k`, `ua-detrac`, `dawn`, `special_vehicle`, `flir_adas`。~~
* ~~`class_number`（int 必填）: 指定目标类别数量，与数据集绑定，对于kitti数据集为`8` 。~~
* `attack_method`（str 选填，默认为`fgsm`）: 指定攻击方法，若`process`为`adv`或`attack`则必填，支持枚举值（第一个为默认值）: `fgsm`, `mifgsm`, `vmifgsm`, `pgd`, `bim`, `cw`, `deepfool`, `gn`, `jitter`。
* `defend_method`（str 选填，默认为`scale`）: 指定防御方法，若`process`为`defend`则必填，支持枚举值（第一个为默认值）:`scale`, `compression`, `fgsm_denoise`, `neural_cleanse`, `pgd_purifier`, `adversarial_training`。
* `selected_samples`（int 选填，默认为64，0表示数据集全部样本数据）: 若`process`为`adv`时有效，生成对抗样本时使用的样本数。
* `confidence_threshold`（float 选填，默认为`0.1`）：攻击防御之后置信度变化阈值，超过该值说明攻击防御成功，若`process`为`attack`或`defend`时有效。
* `resume_from_checkpoint`（bool 选填，默认为`False`）：是否在训练时在已有的权重基础上进行训练，若`process`为`train`时有效，支持枚举值（第一个为默认值）:`False`, `True`。
* `adversarial_sample_proportion`（int 选填，默认为50）: 若`process`为`defend`时有效，防御训练时，训练数据集中对抗样本所占百分比，取值在（0，100）之间。
* `epochs`（int 选填，默认为`100`）：训练迭代次数，若`process`为`train`时有效。
* `batch`（int 选填，默认为`16`）：训练批处理大小，若`process`为`train`或`test`时有效。
* `device`（str 选填，默认为`cpu`）：使用cpu或cuda，支持枚举值（第一个为默认值）:`cpu`, `cuda`。
* `workers`（int 选填，默认为`0`）：加载数据集时workers的数量。
* `epsilon`（float 选填，默认为`8/255`）：扰动强度参数，控制对抗扰动大小。
* `step_size`（float 选填，默认为`2/255`）：步长，迭代攻击的更新幅度。
* `max_iterations`（int 选填，默认为`50`）：最大迭代次数。
* `decay`（float 选填，默认为`1.0`）：延时，取值在（0，1）之间，其值越小，迭代轮数靠前算出来的梯度对当前的梯度方向影响越小。
* `sampled_examples`（int 选填，默认为`5`）：邻近抽取的样例数。
* `random_start`（bool 选填，默认为`False`）：是否随机初始化扰动，支持枚举值（第一个为默认值）:`False`, `True`。
* ~~`loss_function`（str 选填，默认为`cross_entropy`）：损失函数类型，支持枚举值（第一个为默认值）:`cross_entropy`, `mse`, `l1`, `binary_cross_entropy`。~~
* ~~`optimization_method`（str 选填，默认为`adam`）：优化方法，支持枚举值（第一个为默认值）:`adam`, `sgd`。~~
* `lr`（float 选填，默认为`0.001`）：优化器学习率。
* `scale`（int 选填，默认为`10`）：尺度。
* `std`（float 选填，默认为`0.1`）：标准差。
* `scaling_factor`（float 选填，默认为`0.9`）：缩放因子。
* `interpolate_method`（str 选填，默认为`bilinear`）：插值方法，支持枚举值（第一个为默认值）:`bilinear`, `nearest`。
* `image_quality`（int 选填，默认为`90`）：图像质量。
* `filter_kernel_size`（int 选填，默认为`3`）：滤波器核大小。


### 攻击防御方法的有效参数
| 参数 | fgsm | mifgsm | vmifgsm | pgd | bim | cw | deepfool | gn | jitter | scale | compression | neural_cleanse | pgd_purifier | fgsm_denoise | adversarial_training |
|------|------|--------|---------|-----|-----|----|----------|----|--------|-------|-------------|----------------|--------------|--------------|--------------|
| epsilon | 1 | 1 | 1 | 1 | 1 | | | | 1 | | | | 1 | 1 | |
| step_size | | 1 | 1 | 1 | 1 | | | | 1 | | | | 1 | | |
| max_iterations | | 1 | 1 | 1 | 1 | 1 | 1 | | 1 | | | | 1 | |
| decay | | 1 | 1 | | | | | | | | | | | | |
| sampled_examples | | | 1 | | | | | | | | | | | | |
| random_start | | | | 1 | | | | | 1 | | | | | | |
| lr | | | | | | 1 | | | | | | | | | |
| std | | | | | | | | 1 | 1 | | | | | | |
| scale | | | | | | | | | 1 | | | | | | |
| scaling_factor | | | | | | | | | | 1 | | | | | |
| interpolate_method | | | | | | | | | | 1 | | | | | |
| image_quality | | | | | | | | | | | 1 | | | | |
| filter_kernel_size | | | | | | | | | | | | 1 | | | |


### 说明
-- `process`为`adv`：生成对抗样本：使用原始数据集的val子集 -> 通过`selected_samples`参数从原始数据集中选择多少张样本 -> 选择攻击方法（需要使用模型） -> 生成对抗样本 -> 保存对抗样本数据集（原始样本也会一并保存） \
-- `process`为`attack`：攻击：使用生成对抗样本数据集 -> 加载模型进行推理 -> 通过比较原始样本和对抗样本的预测类别是否相同和置信度降低幅度是否大于`confidence_threshold`判断是否攻击成功 -> 打印成功率 -> 保存识别图片 \
-- `process`为`defend`：防御：使用生成对抗样本数据集 -> 选择防御方法 -> 加载模型进行推理 -> 通过比较对抗样本经过防御方法后的预测类别是否正确和置信度提升幅度是否大于 `confidence_threshold`判断是否防御成功 -> 打印成功率 -> 保存识别图片 \
-- `process`为`train`：训练：使用原始数据集的train子集 -> 通过`resume_from_checkpoint`参数决定是否在已有的权重基础上进行训练 -> 执行训练 -> 保存权重文件 \
-- `process`为`test`：测试：使用原始数据集的val子集 -> 执行测试 -> 保存预测图片 \
-- `process`为`predict`：预测：使用图片进行推理（所有图片放在一个文件夹中，该文件夹与一个.yaml一起压缩） -> 保存运行图像 \
-- `process`为`sample`：抽样：使用原始数据集 -> 通过`selected_samples`参数从原始数据集中选择多少张样本（train和val子集都选择） -> 按原始数据集格式保存

## 快速开始

### 构建 Docker 镜像
```bash
cd vehicle_yolo
docker build -t vehicle_yolo:latest .
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
    -v ./input:/project/input:ro \
    -v ./output:/project/output:rw \
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
    -e decay=1.0 \
    -e sampled_examples=5 \
    -e random_start=False \
    -e loss_function=cross_entropy \
    -e optimization_method=adam \
    -e lr=0.001 \
    -e scaling_factor=0.9 \
    -e interpolate_method=bilinear \
    -e image_quality=90 \
    -e filter_kernel_size=3 \
    vehicle_yolo:latest
```
