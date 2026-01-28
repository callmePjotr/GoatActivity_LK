import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from model import ActivityRecognitionModel, LightweightActivityModel, create_model



class ActivityDataset(Dataset):
    """
    - Dataset entsprechend der Ordnerstruktur aus sliding_frame_generator.py laden
    """

    def __init__(self, data_root, split='train', transform=None):
        """
        Args:
            data_root: Pfad zu normalized_sliding/
            split: 'train', 'val' oder 'test'
        """
        self.data_root = Path(data_root)
        self.transform = transform
        self.samples = []

        # Sammle alle .npz files und ihre Labels
        for activity_dir in sorted(self.data_root.iterdir()):
            if not activity_dir.is_dir():
                continue

            activity_name = activity_dir.name

            for clip_dir in activity_dir.iterdir():
                if not clip_dir.is_dir():
                    continue

                # Sammle alle windows dieses clips
                windows = sorted(clip_dir.glob("window_*.npz"))

                for window_path in windows:
                    self.samples.append({
                        'path': window_path,
                        'activity': activity_name,
                        'clip': clip_dir.name
                    })

        # Erstelle Label-Mapping
        self.activities = sorted(set(s['activity'] for s in self.samples))
        self.activity_to_idx = {act: i for i, act in enumerate(self.activities)}
        self.idx_to_activity = {i: act for act, i in self.activity_to_idx.items()}

        print(f"   {split.upper()} Dataset")
        print(f"   Total samples: {len(self.samples)}")
        print(f"   Activities: {self.activities}")
        print(f"   Samples per activity:")
        for act in self.activities:
            count = sum(1 for s in self.samples if s['activity'] == act)
            print(f"      {act}: {count}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Lade .npz file
        data = np.load(sample['path'])
        frames = data['frames']  # (T, H, W, C) uint8

        # Normalisiere zu [0, 1] und konvertiere zu (T, C, H, W)
        frames = frames.astype(np.float32) / 255.0
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # (T, C, H, W)

        # Optional: Data Augmentation
        if self.transform:
            frames = self.transform(frames)

        label = self.activity_to_idx[sample['activity']]

        return frames, label, sample['clip']


class ActivityDataAugmentation:
    """
    Simple Augmentations für Videos
    """

    def __init__(self, temporal_crop_ratio=0.9, horizontal_flip_prob=0.5):
        self.temporal_crop_ratio = temporal_crop_ratio
        self.horizontal_flip_prob = horizontal_flip_prob

    def __call__(self, frames):
        # Temporal Random Crop
        T = frames.shape[0]
        crop_len = int(T * self.temporal_crop_ratio)
        if crop_len < T:
            start = torch.randint(0, T - crop_len + 1, (1,)).item()
            frames = frames[start:start + crop_len]
            # Pad zurück auf Original-Länge
            if frames.shape[0] < T:
                pad = T - frames.shape[0]
                frames = torch.cat([frames, frames[-1:].repeat(pad, 1, 1, 1)], dim=0)

        # Random Horizontal Flip
        if torch.rand(1) < self.horizontal_flip_prob:
            frames = torch.flip(frames, dims=[3])  # Flip width

        return frames


def split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    - Splitted Dataset in Train/Val/Test
    - der Split erfolgt je Clip und nicht über Windows
    - ansonsten können sehr nahezu identische Samples in Train und Val landen
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Gruppierung nach Clips
    clip_to_samples = {}
    for idx, sample in enumerate(dataset.samples):
        clip_id = sample['clip']
        if clip_id not in clip_to_samples:
            clip_to_samples[clip_id] = []
        clip_to_samples[clip_id].append(idx)

    # Clips mischen
    clip_ids = list(clip_to_samples.keys())
    np.random.shuffle(clip_ids)

    # dann aufspalten nach geg. Ratio
    n_clips = len(clip_ids)
    n_train = int(n_clips * train_ratio)
    n_val = int(n_clips * val_ratio)

    train_clips = clip_ids[:n_train]
    val_clips = clip_ids[n_train:n_train + n_val]
    test_clips = clip_ids[n_train + n_val:]

    # indices der clips holen
    train_indices = [idx for clip in train_clips for idx in clip_to_samples[clip]]
    val_indices = [idx for clip in val_clips for idx in clip_to_samples[clip]]
    test_indices = [idx for clip in test_clips for idx in clip_to_samples[clip]]

    return train_indices, val_indices, test_indices


def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None):

    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(dataloader, desc="Training")
    for frames, labels, _ in pbar:
        frames = frames.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Mixed Precision Training für effiezienz und weniger speicherverbrauch
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(frames)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(frames)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()

        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average='macro')

    return epoch_loss, epoch_acc, epoch_f1


def validate(model, dataloader, criterion, device):
    """Validation"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for frames, labels, _ in tqdm(dataloader, desc="Validation"):
            frames = frames.to(device)
            labels = labels.to(device)

            outputs = model(frames)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average='macro')

    return epoch_loss, epoch_acc, epoch_f1, all_preds, all_labels


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest')
    plt.colorbar()

    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    plt.yticks(range(len(class_names)), class_names)

    for i in range(len(cm)):
        for j in range(len(cm)):
            plt.text(j, i, cm[i, j], ha='center', va='center')

    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()



def plot_class_distribution(dataset, class_weights, save_path):
    """
    Visualisiert Klassenverteilung und Weights
    """
    # Count samples per class
    class_counts = {}
    for sample in dataset.samples:
        activity = sample['activity']
        class_counts[activity] = class_counts.get(activity, 0) + 1

    # Sortiere nach activities
    activities = dataset.activities
    counts = [class_counts[act] for act in activities]
    weights = class_weights.cpu().numpy() if isinstance(class_weights, torch.Tensor) else class_weights

    # Create figure mit 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Sample Distribution
    colors = plt.cm.viridis(np.linspace(0, 1, len(activities)))
    bars1 = ax1.bar(range(len(activities)), counts, color=colors, alpha=0.7)
    ax1.set_xlabel('Activity Class', fontsize=12)
    ax1.set_ylabel('Number of Samples', fontsize=12)
    ax1.set_title('Class Distribution', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(activities)))
    ax1.set_xticklabels(activities, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)

    # Zeige counts auf bars
    for i, (bar, count) in enumerate(zip(bars1, counts)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{int(count)}',
                 ha='center', va='bottom', fontsize=10)

    # Plot 2: Class Weights
    bars2 = ax2.bar(range(len(activities)), weights, color=colors, alpha=0.7)
    ax2.set_xlabel('Activity Class', fontsize=12)
    ax2.set_ylabel('Class Weight', fontsize=12)
    ax2.set_title('Loss Weights (Higher = More Important)', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(activities)))
    ax2.set_xticklabels(activities, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)

    # Zeige weights auf bars
    for i, (bar, weight) in enumerate(zip(bars2, weights)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{weight:.3f}',
                 ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f" Class distribution plot saved to {save_path}")


def compute_class_weights(dataset, method='inverse_freq'):
    """
    Berechnet Class Weights basierend auf Klassenverteilung

    Args:
        dataset: ActivityDataset
        method: 'inverse_freq', 'effective_num', oder 'balanced'

    Returns:
        torch.Tensor mit weights für jede Klasse
    """
    # Zähle Samples pro Klasse
    class_counts = {}
    for sample in dataset.samples:
        activity = sample['activity']
        class_counts[activity] = class_counts.get(activity, 0) + 1

    # Sortiere nach activity_to_idx
    num_classes = len(dataset.activities)
    counts = np.array([class_counts[act] for act in dataset.activities])

    if method == 'inverse_freq':
        # Inverse Frequency: weight = 1 / count
        weights = 1.0 / counts
        weights = weights / weights.sum() * num_classes  # Normalisiere

    elif method == 'effective_num':
        # Effective Number of Samples (besser für extreme Imbalance)
        # Paper: "Class-Balanced Loss Based on Effective Number of Samples"
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * num_classes

    elif method == 'balanced':
        # Sklearn-style balanced weights
        weights = len(dataset.samples) / (num_classes * counts)

    else:
        raise ValueError(f"Unknown method: {method}")

    return torch.FloatTensor(weights)


def train_model(
        data_root,
        num_epochs=50,
        batch_size=8,
        lr=1e-4,
        model_type='standard',
        save_dir='checkpoints',
        device='cuda',
        use_class_weights=True,
        class_weight_method='effective_num'
):
    """
    Haupttraining-Funktion

    Args:
        use_class_weights: Wenn True, verwendet gewichtete Loss-Funktion
        class_weight_method: 'inverse_freq', 'effective_num', oder 'balanced'
    """
    print("=" * 70)
    print(" Activity Recognition Training")
    print("=" * 70)

    # Setup directories
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    # Load full dataset
    full_dataset = ActivityDataset(data_root, split='full')

    # Split in Train/Val/Test
    train_indices, val_indices, test_indices = split_dataset(full_dataset)

    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

    # Augmentation nur für Training
    # train_dataset.dataset.transform = ActivityDataAugmentation()

    print(f"\n Dataset Split:")
    print(f"   Train: {len(train_dataset)}")
    print(f"   Val: {len(val_dataset)}")
    print(f"   Test: {len(test_dataset)}")

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)

    # Model
    num_classes = len(full_dataset.activities)
    model = create_model(num_classes, model_type=model_type, device=device)

    # Class Weights
    if use_class_weights:
        print(f"\n Computing Class Weights ({class_weight_method})...")
        class_weights = compute_class_weights(full_dataset, method=class_weight_method)

        print(f"   Class weights:")
        for i, (activity, weight) in enumerate(zip(full_dataset.activities, class_weights)):
            count = sum(1 for s in full_dataset.samples if s['activity'] == activity)
            print(f"      {activity:20s}: weight={weight:.4f} (n={count})")

        # Plot class distribution
        plot_class_distribution(full_dataset, class_weights, save_dir / 'class_distribution.png')

        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        print(f"\n Using uniform class weights")
        criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )

    # Mixed Precision Training
    scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None

    # Training Loop
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print(f"\n Starting Training...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 70)

        # Train
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )

        # Validate
        val_loss, val_acc, val_f1, val_preds, val_labels = validate(
            model, val_loader, criterion, device
        )

        # Scheduler step
        scheduler.step(val_acc)

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
                'class_mapping': full_dataset.activity_to_idx
            }, save_dir / 'best_model.pth')
            print(f" Saved best model! (Val Acc: {val_acc:.4f})")

    # Test final model
    print("\n" + "=" * 70)
    print(" Testing on Test Set...")
    test_loss, test_acc, test_f1, test_preds, test_labels = validate(
        model, test_loader, criterion, device
    )
    print(f"Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f}")

    # Confusion Matrix
    plot_confusion_matrix(
        test_labels, test_preds,
        full_dataset.activities,
        save_dir / 'confusion_matrix.png'
    )
    print(f" Confusion matrix saved to {save_dir / 'confusion_matrix.png'}")

    # Save training history
    with open(save_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print("\n Training complete!")
    print("=" * 70)

    return model, history


# ============================================================================
# Usage Example
# ============================================================================

if __name__ == "__main__":
    DATA_ROOT = r"L:\Uni\Master\Master Thesis\RE_ID_MODELL\YOLO_DATASET_4_PREDICTIONS\detection\track_crops_corrected\normalized_sliding_2act"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, history = train_model(
        data_root=DATA_ROOT,
        num_epochs=50,
        batch_size=1,  # abhängig vom Vram
        lr=1e-4,
        model_type='standard',  # 'standard' oder 'lightweight'
        save_dir='checkpoints',
        device=device,
        use_class_weights=True,  # Class Weights sehr wichtig
        class_weight_method='effective_num'  # 'inverse_freq', 'effective_num', 'balanced'
    )