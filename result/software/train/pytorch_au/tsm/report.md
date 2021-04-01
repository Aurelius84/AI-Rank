
# PyTorch TSM 性能测试

此处给出了[Pytorch实现](https://github.com/mit-han-lab/temporal-shift-module)的TSM模型，任务的详细复现流程，包括环境搭建、复现脚本、测试结果和测试日志等。

<!-- omit in toc -->
## 目录
- [PyTorch TSM 性能测试](#pytorch-tsm-性能测试)
  - [一、环境搭建](#一环境搭建)
  - [二、测试步骤](#二测试步骤)
    - [1.单卡Time2Train及吞吐测试](#1单卡time2train及吞吐测试)
    - [2.单卡准确率测试](#2单卡准确率测试)
  - [三、日志数据](#三日志数据)

## 一、环境搭建

我们使用paddle提供的docker，在其中安装pytorch，并遵循[TSM pytorch实现](https://github.com/mit-han-lab/temporal-shift-module)配置环境，主要过程如下:


- 新建docker container:

docker镜像为: paddlepaddle/paddle:2.0.0rc0-gpu-cuda10.2-cudnn7

```bash
sudo nvidia-docker run --name TSM-torch -v /home:/workspace --network=host -it  --shm-size 128g  paddlepaddle/paddle:2.0.0rc0-gpu-cuda10.2-cudnn7 /bin/bash
```

- 在docker里安装torch

```bash
pip3.7 install torch==1.7.0 torchvision==0.8.1
```

- 参考[TSM pytorch实现](https://github.com/mit-han-lab/temporal-shift-module#prerequisites) 安装依赖

```bash
pip3.7 install TensorboardX
pip3.7 install tqdm
```


## 二、测试步骤

### 1.单卡Time2Train及吞吐测试

我们使用如下脚本，跑2个epoch，测试竞品的性能数据：

```
export CUDA_VISIBLE_DEVICES=0

python3.7 main.py kinetics RGB \
     --arch resnet50 --num_segments 8 \
     --gd 20 --lr 0.02 --wd 1e-4 --lr_steps 20 40 --epochs 50 \
     --batch-size 16 -j 16 --dropout 0.5 --consensus_type=avg --eval-freq=1 \
     --shift --shift_div=8 --shift_place=blockres --npb \
     -p 1
```


### 2.八卡准确率测试

由于数据集较大，训练耗时，单卡训练预计需要20天，故使用8卡进行训练。根据作者公布的数据，50个epoch基本可以达到最佳训练效果，训练启动脚本为：

```
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python3.7 main.py kinetics RGB \
     --arch resnet50 --num_segments 8 \
     --gd 20 --lr 0.02 --wd 1e-4 --lr_steps 20 40 --epochs 50 \
     --batch-size 128 -j 16 --dropout 0.5 --consensus_type=avg --eval-freq=1 \
     --shift --shift_div=8 --shift_place=blockres --npb \
     -p 1
```

- 训练启动前，请在`ops/dataset_config.py`配置好数据路径。

- 程序会自动下载ImageNet预训练模型作为pre-train。

## 三、日志数据
- [单卡Time2Train及吞吐测试日志**TODO**]()
- [单卡准确率测试](./logs/1gpu_accuracy.log)(此日志训练使用p40显卡)

通过以上日志分析，PyTorch经过50个epoch的训练，训练精度（即`val.top1`)达到71.16 %，训练吞吐（即`train.compute_ips`）达到**TODO**img/s。
