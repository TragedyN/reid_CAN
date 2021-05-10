#!/usr/bin/env bash

# train
CUDA_VISIBLE_DEVICES=0 python3 main.py --datadir ./../REID_datasets/prcv2020_v2 --backbone_name resnet50_ibn_b --batchid 8 --batchimage 4 --batchtest 16 --test_every 10 --epochs 601 --loss 1*CrossEntropy_Loss+1*Triplet_Loss+0.0005*Center_Loss --margin 1.3 --nGPU 1 --lr 3.5e-4 --optimizer ADAM_GCC --reset --amsgrad --num_classes 1295 --height 384 --width 192 --save_models --save prcv2020_train_v2-can_b

# test
# CUDA_VISIBLE_DEVICES=0 python3 main.py --datadir ./../REID_datasets/prcv2020_v2 --backbone_name resnet50_ibn_b --margin 1.3 --nGPU 1 --test_only --resume 0 --pre_train model_best.pt --batchtest 16 --num_classes 1295 --height 384 --width 192 --save test-prcv2020_v2_can_b