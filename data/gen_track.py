import os
import glob
import shutil
import numpy as np
import argparse

def get_cropped_data_per_scene(scene_dir, patch_size=64, stride=32):
    raw_file_paths = sorted(glob.glob(os.path.join(scene_dir,'x4','raw*.npy')))
    gt_file_paths=sorted(glob.glob(os.path.join(scene_dir,'raw_gt.npy')))

    crop_data = []
    file_names = [os.path.basename(path) for path in raw_file_paths]
    file_names += [os.path.basename(path) for path in gt_file_paths]

    first_image = np.load(raw_file_paths[0])
    h, w = first_image.shape[:2]

    for x in range(0, w - patch_size + 1, stride):
        for y in range(0, h - patch_size + 1, stride):
            crop_samples = []
            for file_path in raw_file_paths:
                image = np.load(file_path)
                crop = image[y:y + patch_size, x:x + patch_size]
                crop_samples.append(crop)
            gt = np.load(gt_file_paths[0])
            cropgt = gt[y*4:(y + patch_size)*4, x*4:(x + patch_size)*4]
            crop_samples.append(cropgt)
            crop_data.append((crop_samples, file_names))

    print(f"{len(crop_data)} samples of scene {scene_dir}.")
    return crop_data

def rotate_sample(data_samples, mode=0):
    flag = 0 if mode == 0 else 1  # 0: 顺时针旋转, 1: 逆时针旋转
    rotated_samples = [np.rot90(sample, k=flag) for sample in data_samples]
    return rotated_samples

def flip_sample(data_samples, mode=0):
    flipped_samples = [np.flip(sample, axis=mode) for sample in data_samples]  # 0: 垂直翻转, 1: 水平翻转
    return flipped_samples

def save_sample(data_samples, file_names, save_root, id):
    save_path = os.path.join(save_root, id)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for sample, name in zip(data_samples, file_names):
        np.save(os.path.join(save_path, name), sample)

def main():
    parser = argparse.ArgumentParser(description='Prepare cropped and augmented data for deep learning',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_root", type=str,default="/public/home/qsyan/NTIRE2024/track2/Train")
    parser.add_argument("--out_root", type=str,default="/public/home/qsyan/NTIRE2024/track2/Crop2_all64/Train/Scene")
    parser.add_argument("--patch_size", type=int, default=64, help="Size of each patch")
    parser.add_argument("--stride", type=int, default=32, help="Stride for cropping")
    parser.add_argument("--aug", action='store_true', help="Whether to perform data augmentation")

    args = parser.parse_args()

    cropped_training_data_path = args.out_root
    if not os.path.exists(cropped_training_data_path):
        os.makedirs(cropped_training_data_path)

    counter = 0
    all_scenes = sorted(glob.glob(os.path.join(args.data_root, '**', 'x4'), recursive=True))
    scene_dirs = set(os.path.dirname(path) for path in all_scenes)
    for scene_dir in scene_dirs:
        print(f"==> Processing scene: {scene_dir}")
        cropped_data = get_cropped_data_per_scene(scene_dir, patch_size=args.patch_size, stride=args.stride)
        for samples, names in cropped_data:
            save_sample(samples, names, cropped_training_data_path, str(counter).zfill(6))
            counter += 1

            if True:
                # 数据增强：旋转和翻转
                rotated_samples = rotate_sample(samples, 0)  # 顺时针旋转
                save_sample(rotated_samples, names, cropped_training_data_path, str(counter).zfill(6))
                counter += 1

                rotated_samples = rotate_sample(samples, 1)  # 逆时针旋转
                save_sample(rotated_samples, names, cropped_training_data_path, str(counter).zfill(6))
                counter += 1

                flipped_samples = flip_sample(samples, 0)  # 垂直翻转
                save_sample(flipped_samples, names, cropped_training_data_path, str(counter).zfill(6))
                counter += 1

                flipped_samples = flip_sample(samples, 1)  # 水平翻转
                save_sample(flipped_samples, names, cropped_training_data_path, str(counter).zfill(6))
                counter += 1

if __name__ == '__main__':
    main()
