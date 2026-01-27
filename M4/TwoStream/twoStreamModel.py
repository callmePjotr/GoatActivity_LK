import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import numpy as np


# ==========================================================
# TWO-STREAM RNN MODEL
# ==========================================================

class TwoStreamActivityRNN(nn.Module):
    """
    Two-Stream Architektur für Activity Recognition:
    - Stream 1: BBox-Crop (Fokus auf Zieltier)
    - Stream 2: ROI-Crop (Kontext + Nachbarn)
    """

    def __init__(
            self,
            num_classes,
            backbone='resnet18',
            lstm_hidden_size=256,
            lstm_layers=2,
            dropout=0.5,
            fusion_strategy='concat'  # 'concat', 'add', 'max'
    ):
        super().__init__()

        self.fusion_strategy = fusion_strategy

        # ===== BBox-Stream (Fokus) =====
        if backbone == 'resnet18':
            bbox_backbone = models.resnet18(pretrained=True)
            self.bbox_features = nn.Sequential(*list(bbox_backbone.children())[:-1])
            bbox_feat_dim = 512
        elif backbone == 'resnet34':
            bbox_backbone = models.resnet34(pretrained=True)
            self.bbox_features = nn.Sequential(*list(bbox_backbone.children())[:-1])
            bbox_feat_dim = 512
        elif backbone == 'resnet50':
            bbox_backbone = models.resnet50(pretrained=True)
            self.bbox_features = nn.Sequential(*list(bbox_backbone.children())[:-1])
            bbox_feat_dim = 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # ===== ROI-Stream (Kontext) =====
        if backbone == 'resnet18':
            roi_backbone = models.resnet18(pretrained=True)
            self.roi_features = nn.Sequential(*list(roi_backbone.children())[:-1])
            roi_feat_dim = 512
        elif backbone == 'resnet34':
            roi_backbone = models.resnet34(pretrained=True)
            self.roi_features = nn.Sequential(*list(roi_backbone.children())[:-1])
            roi_feat_dim = 512
        elif backbone == 'resnet50':
            roi_backbone = models.resnet50(pretrained=True)
            self.roi_features = nn.Sequential(*list(roi_backbone.children())[:-1])
            roi_feat_dim = 2048

        # ===== Feature Fusion =====
        if fusion_strategy == 'concat':
            lstm_input_size = bbox_feat_dim + roi_feat_dim
        else:  # add, max
            lstm_input_size = bbox_feat_dim  # same dimension required
            assert bbox_feat_dim == roi_feat_dim, "For add/max fusion, both streams must have same dim"

        # ===== LSTM =====
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=False
        )

        # ===== Classifier =====
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden_size, num_classes)
        )

    def extract_features(self, bbox_frames, roi_frames):
        """
        Extrahiert Features aus beiden Streams
        bbox_frames: (batch, seq_len, C, H, W)
        roi_frames: (batch, seq_len, C, H, W)
        """
        batch_size, seq_len, C, H, W = bbox_frames.shape

        # Reshape für CNN: (batch * seq_len, C, H, W)
        bbox_flat = bbox_frames.view(batch_size * seq_len, C, H, W)
        roi_flat = roi_frames.view(batch_size * seq_len, C, H, W)

        # CNN Features
        bbox_feat = self.bbox_features(bbox_flat)  # (batch*seq, feat_dim, 1, 1)
        roi_feat = self.roi_features(roi_flat)  # (batch*seq, feat_dim, 1, 1)

        # Flatten spatial dimensions
        bbox_feat = bbox_feat.view(batch_size, seq_len, -1)  # (batch, seq, feat_dim)
        roi_feat = roi_feat.view(batch_size, seq_len, -1)

        # Feature Fusion
        if self.fusion_strategy == 'concat':
            fused_feat = torch.cat([bbox_feat, roi_feat], dim=-1)
        elif self.fusion_strategy == 'add':
            fused_feat = bbox_feat + roi_feat
        elif self.fusion_strategy == 'max':
            fused_feat = torch.max(bbox_feat, roi_feat)
        else:
            raise ValueError(f"Unknown fusion: {self.fusion_strategy}")

        return fused_feat

    def forward(self, bbox_frames, roi_frames):
        """
        bbox_frames: (batch, seq_len, 3, H, W)
        roi_frames: (batch, seq_len, 3, H, W)
        """
        # Extract and fuse features
        features = self.extract_features(bbox_frames, roi_frames)

        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(features)

        # Nimm letzten hidden state
        final_hidden = lstm_out[:, -1, :]  # (batch, hidden_size)

        # Classification
        logits = self.classifier(final_hidden)

        return logits


# ==========================================================
# DATASET
# ==========================================================

class TwoStreamVideoDataset(Dataset):
    """
    Dataset für Two-Stream Videos
    Erwartet Struktur:
    root/
      class_1/
        clip_001/
          subclip_000_bbox.mp4
          subclip_000_roi.mp4
        clip_002/
          ...
    """

    def __init__(
            self,
            root_dir,
            seq_length=16,
            transform=None,
            target_size=(224, 224)
    ):
        self.root_dir = root_dir
        self.seq_length = seq_length
        self.transform = transform
        self.target_size = target_size

        self.samples = []
        self.classes = []

        # Lade alle Clips
        for class_name in sorted(os.listdir(root_dir)):
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path):
                continue

            class_idx = len(self.classes)
            self.classes.append(class_name)

            for clip_name in os.listdir(class_path):
                clip_path = os.path.join(class_path, clip_name)
                if not os.path.isdir(clip_path):
                    continue

                # Finde alle subclip-Paare
                for filename in os.listdir(clip_path):
                    if filename.endswith('_bbox.mp4'):
                        base_name = filename.replace('_bbox.mp4', '')
                        bbox_path = os.path.join(clip_path, f"{base_name}_bbox.mp4")
                        roi_path = os.path.join(clip_path, f"{base_name}_roi.mp4")

                        if os.path.exists(bbox_path) and os.path.exists(roi_path):
                            self.samples.append({
                                'bbox_video': bbox_path,
                                'roi_video': roi_path,
                                'label': class_idx,
                                'class_name': class_name
                            })

    def __len__(self):
        return len(self.samples)

    def load_video_frames(self, video_path):
        """Lädt Frames aus Video"""
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
        """Sampelt seq_length Frames gleichmäßig"""
        total_frames = len(frames)

        if total_frames >= self.seq_length:
            # Uniform sampling
            indices = np.linspace(0, total_frames - 1, self.seq_length, dtype=int)
        else:
            # Repeat last frame if too short
            indices = list(range(total_frames))
            while len(indices) < self.seq_length:
                indices.append(total_frames - 1)

        sampled = [frames[i] for i in indices]
        return sampled

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load frames
        bbox_frames = self.load_video_frames(sample['bbox_video'])
        roi_frames = self.load_video_frames(sample['roi_video'])

        # Sample frames
        bbox_frames = self.sample_frames(bbox_frames)
        roi_frames = self.sample_frames(roi_frames)

        # Convert to tensor
        bbox_tensor = torch.stack([
            torch.from_numpy(f).permute(2, 0, 1).float() / 255.0
            for f in bbox_frames
        ])  # (seq_len, 3, H, W)

        roi_tensor = torch.stack([
            torch.from_numpy(f).permute(2, 0, 1).float() / 255.0
            for f in roi_frames
        ])

        # Apply transforms
        if self.transform:
            bbox_tensor = self.transform(bbox_tensor)
            roi_tensor = self.transform(roi_tensor)

        label = torch.tensor(sample['label'], dtype=torch.long)

        return bbox_tensor, roi_tensor, label


# ==========================================================
# TRAINING EXAMPLE
# ==========================================================

def train_example():
    """Beispiel-Training"""

    # Hyperparameters
    num_classes = 10  # Anzahl Aktivitäten
    seq_length = 16  # Frames pro Clip
    batch_size = 8
    num_epochs = 50
    learning_rate = 1e-4

    # Model
    model = TwoStreamActivityRNN(
        num_classes=num_classes,
        backbone='resnet18',
        lstm_hidden_size=256,
        lstm_layers=2,
        dropout=0.5,
        fusion_strategy='concat'
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Dataset & DataLoader
    train_dataset = TwoStreamVideoDataset(
        root_dir='D:\temp_data\dataset_all',
        seq_length=seq_length
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for bbox_frames, roi_frames, labels in train_loader:
            bbox_frames = bbox_frames.to(device)
            roi_frames = roi_frames.to(device)
            labels = labels.to(device)

            # Forward
            optimizer.zero_grad()
            outputs = model(bbox_frames, roi_frames)
            loss = criterion(outputs, labels)

            # Backward
            loss.backward()
            optimizer.step()

            # Stats
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        scheduler.step()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total

        print(f"Epoch [{epoch + 1}/{num_epochs}] "
              f"Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}%")

    return model


if __name__ == "__main__":
    # Test model
    model = TwoStreamActivityRNN(num_classes=4, backbone='resnet18')

    # Dummy input
    bbox = torch.randn(2, 16, 3, 224, 224)  # (batch=2, seq=16, C=3, H=224, W=224)
    roi = torch.randn(2, 16, 3, 224, 224)

    output = model(bbox, roi)
    print(f"Output shape: {output.shape}")  # (2, 10)
    print("✅ Model test successful!")