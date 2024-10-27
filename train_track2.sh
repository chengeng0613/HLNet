#!/bin/bash
echo "Start to train the model...."
dataroot="E:\dataset\NTIRE\track2\Crop2_all64"  # including 'Train' and 'NTIRE_Val' floders

device='0'
name="track2"
#load_path="./spynet/spynet_20210409-c6c1bd09.pth" # path of pre-trained model for BracketIRE task ./spynet/spynet_20210409-c6c1bd09.pth

build_dir="./ckpt/"$name

if [ ! -d "$build_dir" ]; then
        mkdir $build_dir
fi

LOG=./ckpt/$name/`date +%Y-%m-%d-%H-%M-%S`.txt

python train.py \
    --dataset_name cropplus    --model tmrnetplus    --name $name            --lr_policy cosine_warmup      \
    --patch_size 64                  --niter 400           --save_imgs False       --lr 1e-4          --dataroot $dataroot   \
    --batch_size 16                   --print_freq 500      --calc_metrics True     --weight_decay 0.01 \
    --gpu_ids $device  -j 8           | tee $LOG
