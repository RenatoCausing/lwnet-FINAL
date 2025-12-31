"""
Training pipeline for FIVES dataset with data augmentation.
Expects:
  - Original images in: <data_root>/Original/  (e.g., 0801.png)
  - Segmented images in: <data_root>/Segmented/ (e.g., 0801_segment.png)

Data augmentation: flipping, rotating, scaling, contrast adjustment (applied in combination)
Image size: 1024x1024
Train/Test split: 90% training, 10% testing
"""

import sys
import json
import os
import argparse
from shutil import copyfile, rmtree
import os.path as osp
from datetime import datetime
import operator
from tqdm import tqdm
import numpy as np
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

from models.get_model import get_arch
from utils.evaluation import evaluate, ewma
from utils.model_saving_loading import save_model, str2bool, load_model
from utils.reproducibility import set_seeds

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='D:/DRIVE/FIVES', 
                    help='Root folder containing Original and Segmented subfolders')
parser.add_argument('--model_name', type=str, default='big_wnet', help='architecture')
parser.add_argument('--batch_size', type=int, default=2, help='batch size')
parser.add_argument('--grad_acc_steps', type=int, default=0, help='gradient accumulation steps (0)')
parser.add_argument('--min_lr', type=float, default=1e-8, help='minimum learning rate')
parser.add_argument('--max_lr', type=float, default=0.01, help='maximum learning rate')
parser.add_argument('--cycle_lens', type=str, default='20/50', help='cycling config (nr cycles/cycle len)')
parser.add_argument('--metric', type=str, default='auc', help='metric for monitoring (auc/loss/dice)')
parser.add_argument('--im_size', type=int, default=1024, help='image size')
parser.add_argument('--in_c', type=int, default=3, help='channels in input images')
parser.add_argument('--do_not_save', type=str2bool, nargs='?', const=True, default=False, help='avoid saving')
parser.add_argument('--save_path', type=str, default='fives_experiment', help='path to save model')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers for data loading')
parser.add_argument('--device', type=str, default='cuda:0', help='device (cpu or cuda:0)')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--start_index', type=int, default=801, help='starting image index (default: 801)')
parser.add_argument('--end_index', type=int, default=None, help='ending image index (default: auto-detect)')
parser.add_argument('--train_split', type=float, default=0.9, help='training split ratio (default: 0.9)')
parser.add_argument('--aug_prob', type=float, default=0.5, help='probability of applying each augmentation')


class FIVESDataset(Dataset):
    """
    Dataset class for FIVES dataset.
    Expects original images at: <data_root>/Original/<idx>.png
    Expects segmented images at: <data_root>/Segmented/<idx>_segment.png
    """
    
    def __init__(self, data_root, indices, im_size=1024, is_train=True, aug_prob=0.5):
        """
        Args:
            data_root: Root folder containing Original and Segmented subfolders
            indices: List of image indices to use (e.g., [801, 802, 803, ...])
            im_size: Target image size
            is_train: Whether this is training set (apply augmentation)
            aug_prob: Probability of applying each augmentation
        """
        self.data_root = data_root
        self.indices = indices
        self.im_size = im_size
        self.is_train = is_train
        self.aug_prob = aug_prob
        
        self.orig_folder = osp.join(data_root, 'Original')
        self.seg_folder = osp.join(data_root, 'Segmented')
        
        # Verify folders exist
        if not osp.exists(self.orig_folder):
            raise ValueError(f"Original folder not found: {self.orig_folder}")
        if not osp.exists(self.seg_folder):
            raise ValueError(f"Segmented folder not found: {self.seg_folder}")
    
    def __len__(self):
        return len(self.indices)
    
    def _find_image(self, folder, base_name):
        """Find image with any common extension"""
        extensions = ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG', '.tif', '.TIF', '.tiff', '.TIFF']
        for ext in extensions:
            path = osp.join(folder, base_name + ext)
            if osp.exists(path):
                return path
        return None
    
    def __getitem__(self, idx):
        img_idx = self.indices[idx]
        
        # Format index with leading zeros (4 digits)
        idx_str = f"{img_idx:04d}"
        
        # Find original and segmented images
        orig_path = self._find_image(self.orig_folder, idx_str)
        seg_path = self._find_image(self.seg_folder, f"{idx_str}_segment")
        
        if orig_path is None:
            raise FileNotFoundError(f"Original image not found for index {idx_str}")
        if seg_path is None:
            raise FileNotFoundError(f"Segmented image not found for index {idx_str}_segment")
        
        # Load images
        img = Image.open(orig_path).convert('RGB')
        target = Image.open(seg_path).convert('L')  # Convert to grayscale
        
        # Resize to target size
        img = TF.resize(img, [self.im_size, self.im_size])
        target = TF.resize(target, [self.im_size, self.im_size], interpolation=TF.InterpolationMode.NEAREST)
        
        # Apply augmentation if training
        if self.is_train:
            img, target = self._augment(img, target)
        
        # Convert to tensor
        img = TF.to_tensor(img)
        target = TF.to_tensor(target)
        
        # Convert target to binary (0 or 1)
        target = (target > 0.5).float().squeeze(0)
        
        return img, target
    
    def _augment(self, img, target):
        """
        Apply random augmentations: flipping, rotating, scaling, contrast adjustment.
        Multiple augmentations can be combined.
        """
        # Random Horizontal Flip
        if random.random() < self.aug_prob:
            img = TF.hflip(img)
            target = TF.hflip(target)
        
        # Random Vertical Flip
        if random.random() < self.aug_prob:
            img = TF.vflip(img)
            target = TF.vflip(target)
        
        # Random Rotation (0, 90, 180, 270 degrees or continuous)
        if random.random() < self.aug_prob:
            angle = random.choice([0, 90, 180, 270]) if random.random() < 0.5 else random.uniform(-45, 45)
            img = TF.rotate(img, angle, fill=(0, 0, 0))
            target = TF.rotate(target, angle, fill=(0,))
        
        # Random Scaling (zoom in/out)
        if random.random() < self.aug_prob:
            scale_factor = random.uniform(0.8, 1.2)
            new_size = int(self.im_size * scale_factor)
            
            # Resize
            img = TF.resize(img, [new_size, new_size])
            target = TF.resize(target, [new_size, new_size], interpolation=TF.InterpolationMode.NEAREST)
            
            # Center crop or pad back to original size
            if new_size > self.im_size:
                # Center crop
                img = TF.center_crop(img, [self.im_size, self.im_size])
                target = TF.center_crop(target, [self.im_size, self.im_size])
            elif new_size < self.im_size:
                # Pad
                pad_size = self.im_size - new_size
                pad_left = pad_size // 2
                pad_right = pad_size - pad_left
                pad_top = pad_size // 2
                pad_bottom = pad_size - pad_top
                img = TF.pad(img, [pad_left, pad_top, pad_right, pad_bottom], fill=0)
                target = TF.pad(target, [pad_left, pad_top, pad_right, pad_bottom], fill=0)
        
        # Random Contrast Adjustment (only for image, not target)
        if random.random() < self.aug_prob:
            contrast_factor = random.uniform(0.7, 1.3)
            img = TF.adjust_contrast(img, contrast_factor)
        
        # Random Brightness Adjustment (only for image, not target)
        if random.random() < self.aug_prob:
            brightness_factor = random.uniform(0.7, 1.3)
            img = TF.adjust_brightness(img, brightness_factor)
        
        # Random Saturation Adjustment (only for image, not target)
        if random.random() < self.aug_prob * 0.5:  # Less frequent
            saturation_factor = random.uniform(0.7, 1.3)
            img = TF.adjust_saturation(img, saturation_factor)
        
        return img, target


def get_available_indices(data_root, start_index=801):
    """
    Scan the Original folder to find all available image indices starting from start_index.
    """
    orig_folder = osp.join(data_root, 'Original')
    seg_folder = osp.join(data_root, 'Segmented')
    
    if not osp.exists(orig_folder):
        raise ValueError(f"Original folder not found: {orig_folder}")
    
    indices = []
    extensions = ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG', '.tif', '.TIF', '.tiff', '.TIFF']
    
    # Scan original folder
    for filename in os.listdir(orig_folder):
        name, ext = osp.splitext(filename)
        if ext in extensions:
            try:
                idx = int(name)
                if idx >= start_index:
                    # Check if corresponding segmented image exists
                    seg_exists = any(
                        osp.exists(osp.join(seg_folder, f"{name}_segment{e}"))
                        for e in extensions
                    )
                    if seg_exists:
                        indices.append(idx)
            except ValueError:
                continue
    
    return sorted(indices)


def create_data_loaders(data_root, start_index, end_index, train_split, 
                        im_size, batch_size, num_workers, aug_prob, seed):
    """
    Create train and test data loaders.
    """
    # Get available indices
    all_indices = get_available_indices(data_root, start_index)
    
    if end_index is not None:
        all_indices = [i for i in all_indices if i <= end_index]
    
    if len(all_indices) == 0:
        raise ValueError(f"No valid image pairs found in {data_root} starting from index {start_index}")
    
    print(f"Found {len(all_indices)} valid image pairs (indices {min(all_indices)} to {max(all_indices)})")
    
    # Shuffle and split
    random.seed(seed)
    random.shuffle(all_indices)
    
    split_idx = int(len(all_indices) * train_split)
    train_indices = sorted(all_indices[:split_idx])
    test_indices = sorted(all_indices[split_idx:])
    
    print(f"Training set: {len(train_indices)} images")
    print(f"Test set: {len(test_indices)} images")
    
    # Create datasets
    train_dataset = FIVESDataset(
        data_root=data_root,
        indices=train_indices,
        im_size=im_size,
        is_train=True,
        aug_prob=aug_prob
    )
    
    test_dataset = FIVESDataset(
        data_root=data_root,
        indices=test_indices,
        im_size=im_size,
        is_train=False,
        aug_prob=0.0
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, test_loader, train_indices, test_indices


def compare_op(metric):
    """Return comparison operator and initial value for metric"""
    if metric == 'auc':
        return operator.gt, 0
    elif metric == 'dice':
        return operator.gt, 0
    elif metric == 'loss':
        return operator.lt, np.inf
    else:
        raise NotImplementedError(f"Unknown metric: {metric}")


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def run_one_epoch(loader, model, criterion, optimizer=None, scheduler=None,
                  grad_acc_steps=0, assess=False):
    """Run one epoch of training or evaluation"""
    device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'
    train = optimizer is not None
    
    model.train() if train else model.eval()
    
    if assess:
        logits_all, labels_all = [], []
    
    n_elems, running_loss, tr_lr = 0, 0, 0
    
    for i_batch, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        if train:
            logits = model(inputs)
        else:
            with torch.no_grad():
                logits = model(inputs)
        
        if isinstance(logits, tuple):  # wnet
            logits_aux, logits = logits
            if model.n_classes == 1:
                loss_aux = criterion(logits_aux, labels.unsqueeze(dim=1).float())
                loss = loss_aux + criterion(logits, labels.unsqueeze(dim=1).float())
            else:
                loss_aux = criterion(logits_aux, labels)
                loss = loss_aux + criterion(logits, labels)
        else:
            if model.n_classes == 1:
                loss = criterion(logits, labels.unsqueeze(dim=1).float())
            else:
                loss = criterion(logits, labels)
        
        if train:
            (loss / (grad_acc_steps + 1)).backward()
            tr_lr = get_lr(optimizer)
            if i_batch % (grad_acc_steps + 1) == 0:
                optimizer.step()
                for _ in range(grad_acc_steps + 1):
                    scheduler.step()
                optimizer.zero_grad()
        
        if assess:
            logits_all.extend(logits)
            labels_all.extend(labels)
        
        running_loss += loss.item() * inputs.size(0)
        n_elems += inputs.size(0)
    
    run_loss = running_loss / n_elems
    
    if assess:
        return logits_all, labels_all, run_loss, tr_lr
    return None, None, run_loss, tr_lr


def train_one_cycle(train_loader, model, criterion, optimizer, scheduler, 
                    grad_acc_steps=0, cycle=0):
    """Train for one cycle"""
    model.train()
    optimizer.zero_grad()
    cycle_len = scheduler.cycle_lens[cycle]
    
    with tqdm(range(cycle_len), desc=f"Cycle {cycle + 1}") as t:
        for epoch in t:
            assess = (epoch == cycle_len - 1)
            tr_logits, tr_labels, tr_loss, tr_lr = run_one_epoch(
                train_loader, model, criterion,
                optimizer=optimizer, scheduler=scheduler,
                grad_acc_steps=grad_acc_steps, assess=assess
            )
            t.set_postfix(tr_loss=f"{tr_loss:.4f}", lr=f"{tr_lr:.6f}")
    
    return tr_logits, tr_labels, tr_loss


def train_model(model, optimizer, criterion, train_loader, val_loader, 
                scheduler, grad_acc_steps, metric, exp_path):
    """Train the model"""
    n_cycles = len(scheduler.cycle_lens)
    best_auc, best_dice, best_cycle = 0, 0, 0
    is_better, best_monitoring_metric = compare_op(metric)
    
    for cycle in range(n_cycles):
        print(f'\nCycle {cycle + 1}/{n_cycles}')
        print('-' * 50)
        
        # Train one cycle
        tr_logits, tr_labels, tr_loss = train_one_cycle(
            train_loader, model, criterion, optimizer, scheduler, grad_acc_steps, cycle
        )
        
        # Evaluate at end of cycle
        print('\nEvaluating...')
        tr_auc, tr_dice = evaluate(tr_logits, tr_labels, model.n_classes)
        del tr_logits, tr_labels
        
        with torch.no_grad():
            vl_logits, vl_labels, vl_loss, _ = run_one_epoch(
                val_loader, model, criterion, assess=True
            )
            vl_auc, vl_dice = evaluate(vl_logits, vl_labels, model.n_classes)
            del vl_logits, vl_labels
        
        print(f'Train Loss: {tr_loss:.4f} | Val Loss: {vl_loss:.4f}')
        print(f'Train AUC: {tr_auc:.4f} | Val AUC: {vl_auc:.4f}')
        print(f'Train DICE: {tr_dice:.4f} | Val DICE: {vl_dice:.4f}')
        print(f'Learning Rate: {get_lr(optimizer):.6f}')
        
        # Check if best
        if metric == 'auc':
            monitoring_metric = vl_auc
        elif metric == 'loss':
            monitoring_metric = vl_loss
        elif metric == 'dice':
            monitoring_metric = vl_dice
        
        if is_better(monitoring_metric, best_monitoring_metric):
            print(f'\n*** Best {metric} improved: {100*best_monitoring_metric:.2f}% -> {100*monitoring_metric:.2f}% ***')
            best_auc, best_dice, best_cycle = vl_auc, vl_dice, cycle + 1
            best_monitoring_metric = monitoring_metric
            if exp_path is not None:
                print('Saving checkpoint...')
                save_model(exp_path, model, optimizer)
    
    del model
    torch.cuda.empty_cache()
    return best_auc, best_dice, best_cycle


def test_model(model, test_loader, criterion):
    """Test the model and return metrics"""
    print('\n' + '=' * 50)
    print('TESTING')
    print('=' * 50)
    
    model.eval()
    
    with torch.no_grad():
        test_logits, test_labels, test_loss, _ = run_one_epoch(
            test_loader, model, criterion, assess=True
        )
        test_auc, test_dice = evaluate(test_logits, test_labels, model.n_classes)
    
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test AUC: {test_auc:.4f} ({100*test_auc:.2f}%)')
    print(f'Test DICE: {test_dice:.4f} ({100*test_dice:.2f}%)')
    
    return test_auc, test_dice, test_loss


def main():
    args = parser.parse_args()
    
    # Device setup
    if args.device.startswith("cuda"):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device.split(":", 1)[1]
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available!")
        print(f'* Using device: {args.device}')
        device = torch.device("cuda")
    else:
        device = torch.device(args.device)
        print(f'* Using device: {args.device}')
    
    # Reproducibility
    set_seeds(args.seed, args.device.startswith("cuda"))
    
    # Parse cycle lens
    cycle_lens = args.cycle_lens.split('/')
    cycle_lens = list(map(int, cycle_lens))
    if len(cycle_lens) == 2:
        cycle_lens = cycle_lens[0] * [cycle_lens[1]]
    
    # Setup experiment path
    do_not_save = str2bool(args.do_not_save)
    if not do_not_save:
        save_path = args.save_path
        if save_path == 'date_time':
            save_path = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        experiment_path = osp.join('experiments', save_path)
        os.makedirs(experiment_path, exist_ok=True)
        
        # Save config
        config_file_path = osp.join(experiment_path, 'config.cfg')
        with open(config_file_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        print(f'* Experiment path: {experiment_path}')
    else:
        experiment_path = None
    
    # Create data loaders
    print('\n* Creating data loaders...')
    train_loader, test_loader, train_indices, test_indices = create_data_loaders(
        data_root=args.data_root,
        start_index=args.start_index,
        end_index=args.end_index,
        train_split=args.train_split,
        im_size=args.im_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        aug_prob=args.aug_prob,
        seed=args.seed
    )
    
    # Save train/test split info
    if experiment_path is not None:
        split_info = {
            'train_indices': train_indices,
            'test_indices': test_indices
        }
        with open(osp.join(experiment_path, 'data_split.json'), 'w') as f:
            json.dump(split_info, f, indent=2)
    
    # Create model
    n_classes = 1  # Binary segmentation
    print(f'\n* Creating {args.model_name} model...')
    model = get_arch(args.model_name, in_c=args.in_c, n_classes=n_classes)
    model = model.to(device)
    print(f"  Total params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.max_lr)
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cycle_lens[0] * len(train_loader),
        eta_min=args.min_lr
    )
    setattr(optimizer, 'max_lr', args.max_lr)
    setattr(scheduler, 'cycle_lens', cycle_lens)
    
    # Loss function
    criterion = torch.nn.BCEWithLogitsLoss()
    print(f'* Loss function: {criterion}')
    
    # Train
    print('\n* Starting training...')
    print('=' * 50)
    best_auc, best_dice, best_cycle = train_model(
        model, optimizer, criterion,
        train_loader, test_loader,
        scheduler, args.grad_acc_steps,
        args.metric, experiment_path
    )
    
    print('\n' + '=' * 50)
    print('TRAINING COMPLETE')
    print('=' * 50)
    print(f'Best Validation AUC: {100*best_auc:.2f}%')
    print(f'Best Validation DICE: {100*best_dice:.2f}%')
    print(f'Best Cycle: {best_cycle}')
    
    # Load best model and test
    if experiment_path is not None:
        print('\n* Loading best model for final testing...')
        model = get_arch(args.model_name, in_c=args.in_c, n_classes=n_classes)
        model, _, _ = load_model(model, experiment_path, device=device)
        model = model.to(device)
        
        test_auc, test_dice, test_loss = test_model(model, test_loader, criterion)
        
        # Save final results
        with open(osp.join(experiment_path, 'final_results.txt'), 'w') as f:
            f.write(f'Training Results\n')
            f.write(f'================\n')
            f.write(f'Best Validation AUC: {100*best_auc:.2f}%\n')
            f.write(f'Best Validation DICE: {100*best_dice:.2f}%\n')
            f.write(f'Best Cycle: {best_cycle}\n')
            f.write(f'\nTest Results\n')
            f.write(f'============\n')
            f.write(f'Test AUC: {100*test_auc:.2f}%\n')
            f.write(f'Test DICE: {100*test_dice:.2f}%\n')
            f.write(f'Test Loss: {test_loss:.4f}\n')
        
        print(f'\n* Results saved to: {experiment_path}')


if __name__ == '__main__':
    """
    Example usage:
    
    # Basic usage (expects images in D:/DRIVE/FIVES/Original and D:/DRIVE/FIVES/Segmented)
    python train_fives.py --data_root D:/DRIVE/FIVES --start_index 801
    
    # With custom settings
    python train_fives.py --data_root D:/DRIVE/FIVES --start_index 801 --batch_size 2 --im_size 1024 --save_path my_fives_experiment
    
    # For cloud compute (expects images in ./data/Original and ./data/Segmented)
    python train_fives.py --data_root ./data --start_index 801 --device cuda:0
    """
    main()
