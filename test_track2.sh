#!/bin/bash
echo "Start to test the model...."
device="2"


dataroot="/public/home/qsyan/NTIRE2024/track2/Crop2_all64"  # including 'Train' and 'NTIRE_Val' floders
name="track2"

python test.py \
    --dataset_name cropplus   --model  tmrnetplus     --name $name              --dataroot $dataroot  \
    --load_iter 133                 --save_imgs True        --calc_metrics False      --gpu_id $device  -j 8
