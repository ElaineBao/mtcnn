#!/usr/bin/env bash

# run this experiment with
# nohup bash script/train_p_net.sh 0,1 &> train_p_net.log &
# to use gpu 0,1 to train
gpu=${1:0:1}

export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export PYTHONUNBUFFERED=1
export MXNET_ENABLE_GPU_P2P=0

python example/train_P_net.py --gpu $1