#!/bin/bash
echo "Start to test the model...."
device="0"


dataroot="D:\\data"  # including 'Train' and 'NTIRE_Val' floders
name="track2"

python test.py \
    --dataset_name cropplus   --model  tmrnetplus     --name $name              --dataroot $dataroot  \
    --load_iter 132                 --save_imgs True        --calc_metrics False      --gpu_id $device  -j 8
