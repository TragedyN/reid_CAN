# Collaborative Attention Network for Person Re-identification

![](https://github.com/lwplw/reid_CAN/blob/main/can.png)

In this repository, we include the 3th place solution for [LSPRC](https://lsprc.github.io/) at PRCV 2020.[Our paper](https://iopscience.iop.org/article/10.1088/1742-6596/1848/1/012074)

## Authors
- [Wenpeng Li](https://blog.csdn.net/lwplwf)
- [Yongli Sun](https://github.com/yonger001)

## Dependencies

- [x] Python == 3.6.2
- [x] PyTorch == 1.3.0
- [x] TorchVision == 0.4.0
- [x] Matplotlib == 3.0.2
- [x] Sklearn == 0.20.1
- [x] PIL == 5.3.0
- [x] Scipy == 1.1.0
- [x] Tqdm == 4.28.1
- [x] prefetch_generator = 1.0.1
- [x] OpenCV == 3.4.3


## Warning
The batchnorm.py file from PyTorch is modified in our model which is under the path of './model'.
Please ensure that the path of modules and nn is consistent with the system environment. More details please refer to the line5 and line10 in batchnorm.py.


## Data
The data structure would look like:
```
data/
    bounding_box_train/
    bounding_box_test/
    query/
```

## Project File Structure
```
|--reid_CAN
  |--data               #（Train data related）
    |--_init_.py
    |--commom.py
    |--market1501.py
    |--sampler.py
  |--loss               #（Loss function）
    |--_init_.py
    |--center_loss.py
    |--triplet_loss.py
  |--model              #（Model structure）
    |--backbones
      |--ibnnet
        |--__init__.py
        |--modules.py
        |--resnet_ibn.py
      |--__init__.py
    |--pretrained
      |--resnet50_ibn_b-9ca61e85.pth
    |--_init_.py
    |--batchnorm.py
    |--can.py
    |--layers.py
  |--utils		#（Functions）
    |--__init__.py
    |--Adam.py
    |--functions.py
    |--functions_camera.py
    |--lr_scheduler.py
    |--random_erasing.py
    |--random_patch.py
    |--utlity.py
  |--main.py
  |--option.py          #（Parameter details）
  |--README.md
  |--run.sh
  |--trainer.py
```

## Get Started

```bash
cd ./reid_CAN
sh run.sh
```
In the **run.sh** file, cancel the comment of the training or testing command.
and  run `sh run.sh`, you can quick stark with training or testing.

## Train
You can specify more parameters in option.py.
Change the parameters `--datadir` to your own train data path

```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py --datadir ./../REID_datasets/prcv2020_v2 --backbone_name resnet50_ibn_b --batchid 8 --batchimage 4 --batchtest 16 --test_every 10 --epochs 601 --loss 1*CrossEntropy_Loss+1*Triplet_Loss+0.0005*Center_Loss --margin 1.3 --nGPU 1 --lr 3.5e-4 --optimizer ADAM_GCC --reset --amsgrad --num_classes 1295 --height 384 --width 192 --save_models --save prcv2020_train_v2-can
```

Parameter details are as follows：
- `-CUDA_VISIBLE_DEVICES`:  GPU ID for training.
- `-batchid`: Number of training IDs sent to each batch
- `-batchimage`: Number of images sent to training per ID
- `-test_every`: How many epochs do a test
- `-epochs`: Total training epoch number
- `-loss`: Weights * loss function 
- `-margin`: Tripletloss parameter
- `-nGPU`: How many GPU choose to train
- `-lr`: Learning rate
- `-num_classes`: Number of independent IDs
- `-save_models`: Save model for each test
- `-save`: Model saved path

Refer to option.py for other related parameters

## Test
Change the parameters `--datadir` to your own test data path

```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py --datadir ./../REID_datasets/prcv2020_v2 --backbone_name resnet50_ibn_a --margin 1.3 --nGPU 1 --test_only --resume 0 --pre_train model_best.pt --batchtest 16 --num_classes 1295 --height 384 --width 192 --save test-prcv2020_v2_can
```

Parameter details are as follows：
- `-test_only`: Test switch, only test if open
- `-pre_train`: The model choosed to test

Refer to `option.py` for other related parameters


## Citation
If you find [our work](https://iopscience.iop.org/article/10.1088/1742-6596/1848/1/012074) useful in your research, please consider citing:
````bibtex
@article{li2021collaborative,
    title={Collaborative Attention Network for Person Re-identification},
    author={Li, Wenpeng and Sun, Yongli and Wang, Jinjun and Cao, Junliang and Xu, Han and Yang, Xiangru and Sun, Guangze and Ma, Yangyang and Long, Yilin},
    journal={Journal of Physics: Conference Series},
    doi={10.1088/1742-6596/1848/1/012074},
    year={2021},
}
````
