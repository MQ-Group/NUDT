# NUDT 智能安全训练靶场

## 场景代码
- base_train_optimaze: 基础模型训练与优化
- drone_yolo: 无人机识别场景Drone-Yolo模型
- sample_attack_defend: 样本攻防
- vehicle_ssd_fasterrcnn: 车辆识别场景SSD、Faster-RCNN模型
- vehicle_yolo: 车辆识别场景Yolov5、Yolov8、Yolov10模型

## 数据集
- [KITTI](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/kitti.yaml):`https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/kitti.yaml`
- [BDD100K](https://blog.csdn.net/weixin_52514564/article/details/129891785):`https://blog.csdn.net/weixin_52514564/article/details/129891785`, 百度网盘下载，按博客代码说明进行转化为yolo数据集格式
- [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/scripts/get_coco.sh):`https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/scripts/get_coco.sh`
- [COCO2017](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml):`https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml`
- [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/scripts/get_imagenet.sh):`https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/scripts/get_imagenet.sh`
- [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz):`https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz`, `trochvision.datasets.CIFAR10(download=True)`
- [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz):`https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz`, `trochvision.datasets.CIFAR100(download=True)`
- [MNIST](): `trochvision.datasets.MNIST(download=True)`
- [VOC](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/VOC.yaml):`https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/VOC.yaml`
- [CityScapes](https://www.kaggle.com/datasets/rifqinaufalabdjul/cityscapes-in-yolo-format):`https://www.kaggle.com/datasets/rifqinaufalabdjul/cityscapes-in-yolo-format`
- [UA-DETRAC](https://github.com/forever208/yolov5_train_on_UA-DETRAC):`https://github.com/forever208/yolov5_train_on_UA-DETRAC`
- [FLIR-ADAS-v2](https://gitcode.com/Open-source-documentation-tutorial/836a0):`https://gitcode.com/Open-source-documentation-tutorial/836a0`, `https://blog.csdn.net/Jiuyee/article/details/127334227`
- [DAWN|车辆检测数据集|恶劣天气数据集](https://www.kaggle.com/datasets/shuvoalok/dawn-dataset):`https://www.kaggle.com/datasets/shuvoalok/dawn-dataset`, `https://www.selectdataset.com/dataset/f92a848fcc49862fed38b347befee28b`
- [特种车辆数据集](https://tianchi.aliyun.com/dataset/180177):`https://tianchi.aliyun.com/dataset/180177`
- [Drone](https://www.kaggle.com/datasets/otabekpuladjonov/drone-det):`https://www.kaggle.com/datasets/otabekpuladjonov/drone-det`, `https://www.selectdataset.com/dataset/149b881732d16c3e7dcfd5b7baebcad4`
- [Drone-type](https://www.selectdataset.com/dataset/0c3df9e24ed6c9e9cf9b3496bbfcba3d):`https://www.selectdataset.com/dataset/0c3df9e24ed6c9e9cf9b3496bbfcba3d`
- [YOLO Drone Detection Dataset](https://www.kaggle.com/datasets/muki2003/yolo-drone-detection-dataset):`https://www.kaggle.com/datasets/muki2003/yolo-drone-detection-dataset`, `https://www.selectdataset.com/dataset/45627d8baa249b56699828423e7c4772`
- [图像加雾算法](https://blog.csdn.net/m0_62919535/article/details/139319954):`https://blog.csdn.net/m0_62919535/article/details/139319954`
- [ST-CMDS](https://openslr.org/38/):`https://openslr.org/38/`
- [KeSpeech](https://modelscope.cn/datasets/pengzhendong/KeSpeech/files):`https://modelscope.cn/datasets/pengzhendong/KeSpeech/files`
- [Mandarin_Chinese_Scripted_Speech_Corpus](https://www.modelscope.cn/datasets/Magic_Data/Mandarin_Chinese_Scripted_Speech_Corpus/files):`https://www.modelscope.cn/datasets/Magic_Data/Mandarin_Chinese_Scripted_Speech_Corpus/files`
- [AISHELL-1](https://www.modelscope.cn/datasets/OmniData/AISHELL-1/files):`https://www.modelscope.cn/datasets/OmniData/AISHELL-1/files`
- [AISHELL-3](https://www.modelscope.cn/datasets/OmniData/AISHELL-3/files):`https://www.modelscope.cn/datasets/OmniData/AISHELL-3/files`
- [如何在ModelScope平台下载数据集?](https://www.modelscope.cn/docs/datasets/download#3-%E4%BD%BF%E7%94%A8%E5%91%BD%E4%BB%A4%E8%A1%8C%E5%B7%A5%E5%85%B7%E4%B8%8B%E8%BD%BD%E6%95%B0%E6%8D%AE%E9%9B%86%E6%96%87%E4%BB%B6)
- [cifar转PNG](https://github.com/knjcode/cifar2png):`https://github.com/knjcode/cifar2png`
- [minst转PNG](https://xinancsd.github.io/MachineLearning/mnist_parser.html):`https://xinancsd.github.io/MachineLearning/mnist_parser.html`