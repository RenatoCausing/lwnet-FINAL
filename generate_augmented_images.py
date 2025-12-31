"""
Generate augmented images from existing dataset and save them as new files.
Takes images 0001-0800 and creates augmented versions starting from 0801.
"""

import os
import os.path as osp
import random
import argparse
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='./', help='Root folder with Original/Segmented')
parser.add_argument('--start_index', type=int, default=1, help='Start index of source images')
parser.add_argument('--end_index', type=int, default=800, help='End index of source images')
parser.add_argument('--output_start', type=int, default=801, help='Start index for augmented images')
parser.add_argument('--num_augmentations', type=int, default=1, help='Number of augmented versions per image')
parser.add_argument('--seed', type=int, default=42, help='Random seed')


def apply_augmentation(img, target):
    """Apply random combination of augmentations"""
    
    # Random Horizontal Flip (50% chance)
    if random.random() < 0.5:
        img = TF.hflip(img)
        target = TF.hflip(target)
    
    # Random Vertical Flip (50% chance)
    if random.random() < 0.5:
        img = TF.vflip(img)
        target = TF.vflip(target)
    
    # Random Rotation (50% chance)
    if random.random() < 0.5:
        angle = random.choice([90, 180, 270])
        img = TF.rotate(img, angle)
        target = TF.rotate(target, angle)
    
    # Random Contrast (50% chance)
    if random.random() < 0.5:
        contrast_factor = random.uniform(0.7, 1.3)
        img = TF.adjust_contrast(img, contrast_factor)
    
    # Random Brightness (50% chance)
    if random.random() < 0.5:
        brightness_factor = random.uniform(0.7, 1.3)
        img = TF.adjust_brightness(img, brightness_factor)
    
    # Random Saturation (30% chance)
    if random.random() < 0.3:
        saturation_factor = random.uniform(0.7, 1.3)
        img = TF.adjust_saturation(img, saturation_factor)
    
    return img, target


def find_image(folder, base_name):
    """Find image with any extension"""
    extensions = ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG']
    for ext in extensions:
        path = osp.join(folder, base_name + ext)
        if osp.exists(path):
            return path
    return None


def generate_augmented_dataset(data_root, start_idx, end_idx, output_start, num_aug, seed):
    """Generate augmented images and save to disk"""
    
    random.seed(seed)
    
    orig_folder = osp.join(data_root, 'Original')
    seg_folder = osp.join(data_root, 'Segmented')
    
    if not osp.exists(orig_folder):
        raise ValueError(f"Original folder not found: {orig_folder}")
    if not osp.exists(seg_folder):
        raise ValueError(f"Segmented folder not found: {seg_folder}")
    
    print(f"Generating {num_aug} augmented version(s) per image")
    print(f"Source: {start_idx:04d} to {end_idx:04d}")
    print(f"Output: {output_start:04d} onwards")
    print(f"Total new images: {(end_idx - start_idx + 1) * num_aug}")
    print()
    
    output_idx = output_start
    total_images = (end_idx - start_idx + 1) * num_aug
    
    with tqdm(total=total_images, desc="Generating augmented images") as pbar:
        for source_idx in range(start_idx, end_idx + 1):
            idx_str = f"{source_idx:04d}"
            
            # Find source images
            orig_path = find_image(orig_folder, idx_str)
            seg_path = find_image(seg_folder, f"{idx_str}_segment")
            
            if orig_path is None:
                print(f"Warning: Original image not found for {idx_str}, skipping...")
                pbar.update(num_aug)
                continue
            
            if seg_path is None:
                print(f"Warning: Segmented image not found for {idx_str}_segment, skipping...")
                pbar.update(num_aug)
                continue
            
            # Load images
            try:
                img = Image.open(orig_path).convert('RGB')
                target = Image.open(seg_path).convert('L')
            except Exception as e:
                print(f"Error loading {idx_str}: {e}, skipping...")
                pbar.update(num_aug)
                continue
            
            # Generate augmented versions
            for _ in range(num_aug):
                aug_img, aug_target = apply_augmentation(img.copy(), target.copy())
                
                # Save augmented images
                output_idx_str = f"{output_idx:04d}"
                
                orig_output_path = osp.join(orig_folder, f"{output_idx_str}.png")
                seg_output_path = osp.join(seg_folder, f"{output_idx_str}_segment.png")
                
                aug_img.save(orig_output_path)
                aug_target.save(seg_output_path)
                
                output_idx += 1
                pbar.update(1)
    
    print()
    print("=" * 60)
    print("AUGMENTATION COMPLETE!")
    print("=" * 60)
    print(f"Generated {output_idx - output_start} new images")
    print(f"New image indices: {output_start:04d} to {output_idx-1:04d}")
    print(f"Total dataset size: {output_idx - 1} images")
    print()


def main():
    args = parser.parse_args()
    
    print("=" * 60)
    print("DATA AUGMENTATION - GENERATE NEW IMAGES")
    print("=" * 60)
    print()
    
    generate_augmented_dataset(
        data_root=args.data_root,
        start_idx=args.start_index,
        end_idx=args.end_index,
        output_start=args.output_start,
        num_aug=args.num_augmentations,
        seed=args.seed
    )


if __name__ == '__main__':
    """
    Example usage:
    
    # Generate 1 augmented version per image (800 -> 1600 total images)
    python generate_augmented_images.py --data_root ./ --num_augmentations 1
    
    # Generate 2 augmented versions per image (800 -> 2400 total images)
    python generate_augmented_images.py --data_root ./ --num_augmentations 2
    
    # Generate 5 augmented versions per image (800 -> 4800 total images)
    python generate_augmented_images.py --data_root ./ --num_augmentations 5
    
    # Custom range
    python generate_augmented_images.py --data_root ./ --start_index 1 --end_index 800 --output_start 801 --num_augmentations 3
    """
    main()
