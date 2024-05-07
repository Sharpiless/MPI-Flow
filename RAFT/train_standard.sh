#!/bin/bash
CUDA_VISIBLE_DEVICES=3,5 python -u train.py --name mpi-0.1-0.25-rf \
    --stage mpi-flow --validation kitti \
    --restore_ckpt weights/raft-things.pth \
    --gpus 0 1 --num_steps 50000 --batch_size 6 \
    --lr 0.0001 --image_size 288 960 --wdecay 0.00001 --gamma=0.85 \
    --data_root /data1/liangyingping/MPI-Flow/dataset/debug