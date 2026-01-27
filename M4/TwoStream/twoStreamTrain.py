# ==========================================================
# IMPROVED TWO-STREAM RNN TRAINING
# ==========================================================
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
import cv2
import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from collections import defaultdict, Counter

from twoStreamModel import TwoStreamActivityRNN
import random



def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
set_seed(42)


# ==========================================================
# CONFIG
# ==========================================================
DATA_ROOT = "path_to_activity_dataset/dataset_all_2"
CHECKPOINT_DIR = "/home/user/checkpoint_dir/..."
LOG_DIR = "/home/user/log_dir/..."

# Training hyperparameters
SEQ_LENGTH = 16
BATCH_SIZE = 8  
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# Early stopping
EARLY_STOPPING_PATIENCE = 15
MIN_DELTA = 0.001  

# Model hyperparameters
BACKBONE = 'resnet50' # 'resnet18', 'resnet34', 'resnet50'
LSTM_HIDDEN_SIZE = 256
LSTM_LAYERS = 2
DROPOUT = 0.5
FUSION_STRATEGY = 'concat' # 'concat', 'add', 'max'

# Data split
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Class balancing
USE_WEIGHTED_LOSS = True
USE_OVERSAMPLING = True
MIN_CLASS_SAMPLES = 30  # Mindestanzahl pro Klasse beim Training

USE_GRADIENT_CLIPPING = True
GRAD_CLIP_NORM = 1.0

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 Using device: {DEVICE}")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# ==========================================================
# DATASET WITH CLIP-BASED SPLITTING
# ==========================================================
class TwoStreamVideoDataset(Dataset):
    """Dataset mit Clip-based Organization"""

    def __init__(self, root_dir, seq_length=16, transform=None, exclude_classes=None):
        self.root_dir = Path(root_dir)
        self.seq_length = seq_length
        self.transform = transform
        self.exclude_classes = exclude_classes or ['unknown']  # Default: exclude 'unknown'

        self.samples = []
        self.classes = []
        self.class_to_idx = {}
        self.clip_to_samples = defaultdict(list)  # Mapping: clip_id -> [sample_indices]

        # Load dataset
        for class_folder in sorted(self.root_dir.iterdir()):
            if not class_folder.is_dir():
                continue

            class_name = class_folder.name
            
            # Skip excluded classes
            if class_name.lower() in [c.lower() for c in self.exclude_classes]:
                print(f"  ⚠️  Skipping class: {class_name}")
                continue
            
            if class_name not in self.class_to_idx:
                self.class_to_idx[class_name] = len(self.classes)
                self.classes.append(class_name)

            class_idx = self.class_to_idx[class_name]

            # Iterate through clip folders
            for clip_folder in sorted(class_folder.iterdir()):
                if not clip_folder.is_dir():
                    continue

                clip_id = f"{class_name}_{clip_folder.name}"
                bbox_video = clip_folder / "bbox.mp4"
                roi_video = clip_folder / "roi.mp4"

                if bbox_video.exists() and roi_video.exists():
                    sample_idx = len(self.samples)
                    self.samples.append({
                        "bbox_video": str(bbox_video),
                        "roi_video": str(roi_video),
                        "label": class_idx,
                        "class_name": class_name,
                        "clip_id": clip_id
                    })
                    self.clip_to_samples[clip_id].append(sample_idx)

        print(f"  Found {len(self.samples)} video pairs from {len(self.clip_to_samples)} clips")
        print(f"  Classes ({len(self.classes)}): {self.classes}")

        # Class distribution
        label_counts = Counter([s['label'] for s in self.samples])
        print("\n  📊 Class distribution:")
        for cls_name, cls_idx in self.class_to_idx.items():
            count = label_counts[cls_idx]
            percentage = 100 * count / len(self.samples)
            print(f"    {cls_name:15s}: {count:4d} samples ({percentage:5.1f}%)")
        
        # Warning for underrepresented classes
        min_samples = min(label_counts.values()) if label_counts else 0
        if min_samples < 10:
            print(f"\n  ⚠️  WARNING: Smallest class has only {min_samples} samples!")
            print(f"     Consider: 1) Collecting more data")
            print(f"              2) Strong augmentation")
            print(f"              3) Leave-one-out validation for tiny classes")

    def get_clip_ids(self):
        """Returns all unique clip IDs"""
        return list(self.clip_to_samples.keys())

    def get_samples_by_clips(self, clip_ids):
        """Returns sample indices for given clip IDs"""
        indices = []
        for clip_id in clip_ids:
            indices.extend(self.clip_to_samples[clip_id])
        return indices

    def __len__(self):
        return len(self.samples)

    def load_video_frames(self, video_path):
        """Load frames from video"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        return frames

    def sample_frames(self, frames):
        """Sample seq_length frames uniformly"""
        total_frames = len(frames)
        if total_frames >= self.seq_length:
            indices = np.linspace(0, total_frames - 1, self.seq_length, dtype=int)
        else:
            indices = list(range(total_frames))
            while len(indices) < self.seq_length:
                indices.append(total_frames - 1)
        return [frames[i] for i in indices]

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load and sample frames
        bbox_frames = self.sample_frames(self.load_video_frames(sample['bbox_video']))
        roi_frames = self.sample_frames(self.load_video_frames(sample['roi_video']))

        # Convert to tensor
        bbox_tensor = torch.stack([
            torch.from_numpy(f).permute(2, 0, 1).float() / 255.0
            for f in bbox_frames
        ])
        roi_tensor = torch.stack([
            torch.from_numpy(f).permute(2, 0, 1).float() / 255.0
            for f in roi_frames
        ])

        # Apply transforms (frame-wise)
        if self.transform:
            bbox_tensor = torch.stack([self.transform(frame) for frame in bbox_tensor])
            roi_tensor = torch.stack([self.transform(frame) for frame in roi_tensor])

        label = torch.tensor(sample['label'], dtype=torch.long)
        return bbox_tensor, roi_tensor, label


# ==========================================================
# CLIP-BASED SPLITTING
# ==========================================================
def clip_based_split(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Split dataset by clips, not samples"""
    
    # Get all clips grouped by class
    class_clips = defaultdict(list)
    for clip_id in dataset.get_clip_ids():
        class_name = clip_id.split('_')[0]
        class_clips[class_name].append(clip_id)

    train_clips, val_clips, test_clips = [], [], []

    # Split each class separately to maintain class distribution
    for class_name, clips in class_clips.items():
        np.random.shuffle(clips)
        n = len(clips)
        
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        
        train_clips.extend(clips[:n_train])
        val_clips.extend(clips[n_train:n_train + n_val])
        test_clips.extend(clips[n_train + n_val:])

    # Convert clips to sample indices
    train_indices = dataset.get_samples_by_clips(train_clips)
    val_indices = dataset.get_samples_by_clips(val_clips)
    test_indices = dataset.get_samples_by_clips(test_clips)

    print(f"\n📊 Clip-based split:")
    print(f"  Train: {len(train_clips)} clips → {len(train_indices)} samples")
    print(f"  Val:   {len(val_clips)} clips → {len(val_indices)} samples")
    print(f"  Test:  {len(test_clips)} clips → {len(test_indices)} samples")

    return train_indices, val_indices, test_indices


# ==========================================================
# WEIGHTED SAMPLER FOR CLASS BALANCING
# ==========================================================
def create_weighted_sampler(dataset, indices, oversample_factor=3.0):
    """Create weighted sampler to balance classes
    
    Args:
        dataset: The dataset
        indices: Sample indices for this split
        oversample_factor: How much more to sample minority classes (default: 3x)
    """
    
    # Count samples per class
    labels = [dataset.samples[i]['label'] for i in indices]
    class_counts = Counter(labels)
    
    print(f"\n  📊 Class distribution in split:")
    for cls_idx, count in sorted(class_counts.items()):
        cls_name = dataset.classes[cls_idx]
        print(f"    {cls_name}: {count} samples")
    
    # Calculate weights (inverse frequency with boost for tiny classes)
    max_count = max(class_counts.values())
    class_weights = {}
    
    for cls, count in class_counts.items():
        # Base weight: inverse frequency
        base_weight = max_count / count
        
        # Extra boost for very small classes (< 10 samples)
        if count < 10:
            boost = oversample_factor
            print(f"    ⚡ Boosting {dataset.classes[cls]} by {boost}x (only {count} samples)")
        else:
            boost = 1.0
        
        class_weights[cls] = base_weight * boost
    
    # Assign weight to each sample
    sample_weights = [class_weights[label] for label in labels]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    return sampler


# ==========================================================
# DATA AUGMENTATION (Enhanced for small classes)
# ==========================================================
def get_train_transforms(strong_aug=False):
    """Augmentation for training
    
    Args:
        strong_aug: If True, apply stronger augmentation (useful for small classes)
    """
    if strong_aug:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomAffine(degrees=15, translate=(0.15, 0.15), scale=(0.85, 1.15)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.Resize((224, 224)),
        ])
    else:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.Resize((224, 224)),
        ])

def get_val_transforms():
    """No augmentation for validation"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
    ])


# ==========================================================
# EARLY STOPPING
# ==========================================================
class EarlyStopping:
    """Early stopping handler"""
    
    def __init__(self, patience=10, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'max':
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:  # min
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            return True
        
        return False


# ==========================================================
# TRAINING FUNCTIONS
# ==========================================================
def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training")
    for bbox_frames, roi_frames, labels in pbar:
        bbox_frames = bbox_frames.to(device)
        roi_frames = roi_frames.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(bbox_frames, roi_frames)
        loss = criterion(outputs, labels)
        
        loss.backward()
        
        if USE_GRADIENT_CLIPPING:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=GRAD_CLIP_NORM
            )
        
        optimizer.step()


        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            'loss': running_loss / (pbar.n + 1),
            'acc': 100. * correct / total
        })

    return running_loss / len(dataloader), 100. * correct / total


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for bbox_frames, roi_frames, labels in tqdm(dataloader, desc="Validation"):
            bbox_frames = bbox_frames.to(device)
            roi_frames = roi_frames.to(device)
            labels = labels.to(device)

            outputs = model(bbox_frames, roi_frames)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return running_loss / len(dataloader), 100. * correct / total, all_preds, all_labels


# ==========================================================
# MAIN
# ==========================================================
def main():
    print("\n" + "=" * 70)
    print("🎬 IMPROVED TWO-STREAM RNN TRAINING")
    print("=" * 70)

    # Load dataset
    print("\n📂 Loading dataset...")
    full_dataset = TwoStreamVideoDataset(
        root_dir=DATA_ROOT,
        seq_length=SEQ_LENGTH,
        transform=None  # Will be set per split
    )

    NUM_CLASSES = len(full_dataset.classes)

    # Clip-based split
    train_indices, val_indices, test_indices = clip_based_split(
        full_dataset, TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT
    )

    # Create subset datasets with transforms
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    
    # Check if we have very small classes - use strong augmentation if so
    train_labels = [full_dataset.samples[i]['label'] for i in train_indices]
    min_class_count = min(Counter(train_labels).values())
    use_strong_aug = min_class_count < 5
    
    if use_strong_aug:
        print("\n⚡ Using STRONG augmentation due to very small classes")
    
    train_dataset.dataset.transform = get_train_transforms(strong_aug=use_strong_aug)
    
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    val_dataset.dataset.transform = get_val_transforms()
    
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    test_dataset.dataset.transform = get_val_transforms()

    # Create dataloaders
    if USE_OVERSAMPLING:
        train_sampler = create_weighted_sampler(full_dataset, train_indices, oversample_factor=5.0)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                 sampler=train_sampler, num_workers=4)
        print("\n✅ Using weighted sampling for class balancing (5x boost for tiny classes)")
    else:
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                 shuffle=True, num_workers=4)

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                           shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=4)

    # Create model
    print("\n🔨 Building model...")
    model = TwoStreamActivityRNN(
        num_classes=NUM_CLASSES,
        backbone=BACKBONE,
        lstm_hidden_size=LSTM_HIDDEN_SIZE,
        lstm_layers=LSTM_LAYERS,
        dropout=DROPOUT,
        fusion_strategy=FUSION_STRATEGY
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # Weighted loss for class imbalance
    if USE_WEIGHTED_LOSS:
        train_labels = [full_dataset.samples[i]['label'] for i in train_indices]
        class_counts = Counter(train_labels)
        class_weights = torch.tensor([
            1.0 / class_counts[i] for i in range(NUM_CLASSES)
        ], dtype=torch.float32).to(DEVICE)
        class_weights = class_weights / class_weights.sum() * NUM_CLASSES
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"\n✅ Using weighted loss: {class_weights.cpu().numpy()}")
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    
    # ReduceLROnPlateau: reduces LR when validation plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )

    # Early stopping
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, 
                                   min_delta=MIN_DELTA, mode='max')

    # Training loop
    print("\n🚀 Starting training...")
    best_val_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(NUM_EPOCHS):
        print(f"\n{'=' * 70}")
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] - LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"{'=' * 70}")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, DEVICE)

        # Update scheduler
        scheduler.step(val_acc)

        # Log history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"\n📊 Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"📊 Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'classes': full_dataset.classes
            }, Path(CHECKPOINT_DIR) / "best_model.pth")
            print(f"✅ Saved best model (Val Acc: {val_acc:.2f}%)")

        # Early stopping check
        if early_stopping(val_acc):
            print(f"\n⏹️  Early stopping triggered at epoch {epoch + 1}")
            break

    # Final evaluation on test set
    print("\n" + "=" * 70)
    print("📊 FINAL TEST EVALUATION")
    print("=" * 70)

    checkpoint = torch.load(Path(CHECKPOINT_DIR) / "best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_acc, test_preds, test_labels = validate(
        model, test_loader, criterion, DEVICE
    )

    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print("\n" + classification_report(test_labels, test_preds,
                                       target_names=full_dataset.classes))

    # Save plots
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Progress')

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy Progress')

    plt.tight_layout()
    plt.savefig(Path(LOG_DIR) / "training_curves.png")
    print(f"\n✅ Saved training curves")

    print("\n🎉 Training complete!")


if __name__ == "__main__":
    main()