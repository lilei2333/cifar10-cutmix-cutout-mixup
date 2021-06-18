# 比较cutout、mixup和cutmix在cifar-10数据集上的表现


本项目是神经网络和深度学习课程的期末PJ。我使用Resnet32来跑cifar-10分类问题。
为了让其他人能够复现我的结果，我将训练好的模型放在了百度网盘，链接：https://pan.baidu.com/s/1jjSMquKD3vryhvokqwhrFQ，提取码：hl2k


## Requirements
整个项目使用Python 3.6 和 PyTorch 1.7.1构建。可使用下面的命令安装所有依赖包:
```
pip install -r requirements.txt
```

## How to train
使用train.sh文件里的命令，四条命令分别为训练baseline、cutout、mixup和cutmix所用

第一次运行时，会连网下载数据并放到data文件夹下，后面运行时不会再次下载。
默认的日志文件和tensorboard文件都在logs文件夹下面。若要使用tensorboard查看结果，可用
```
tensorboard --logdir=/logs
```

我用一张12-GB的Titan X GPU训练200个epoch，四个任务用时接近，均为1.2小时。四个任务同时训练，总耗时2.6小时。


# results
如下表所示，对比baseline，使用cutmix后测试集准确率提高最多

|                          |     Baseline    |     +Cutmix    |     +Cutout    |     +Mixup    |
|:------------------------:|-----------------|----------------|----------------|---------------|
|     测试集准确率（%）    |       91.50     |      93.42     |      93.28     |      93.25    |
