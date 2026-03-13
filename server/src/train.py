from collections import Counter
import torch
import torch.nn as nn
import torch.onnx
from torch.amp import autocast, GradScaler
from datasets import load_dataset
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torch.optim.lr_scheduler import (
    OneCycleLR,
    ReduceLROnPlateau,
    CosineAnnealingWarmRestarts,
)
import argparse
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import numpy as np


# ============== Mixup & CutMix ==============


def mixup_data(x, y, alpha=0.2):
    """Mixup data augmentation: blends two samples with random lambda."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute loss for mixed samples."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class CutMix:
    """CutMix data augmentation wrapper."""

    def __init__(self, beta=1.0):
        self.beta = beta

    def __call__(self, img, label):
        if np.random.rand() > 0.5:
            return img, label, 1.0

        beta = self.beta
        lam = np.random.beta(beta, beta) if beta > 0 else 1.0
        return img, label, lam


def cutmix(x, y, alpha=1.0):
    """CutMix: replaces a random region with another sample's region."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    r_x = torch.zeros_like(x)
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    r_x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # Adjust lambda based on actual cut area
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))

    y_a, y_b = y, y[index]
    return r_x, y_a, y_b, lam


def rand_bbox(size, lam):
    """Generate random bounding box for CutMix."""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Randomly select center point
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: TensorBoard not available. Install with: pip install tensorboard")


def parse_args():
    parser = argparse.ArgumentParser(description="Fatigue Driving Model Training")
    parser.add_argument(
        "--dataset",
        type=str,
        default="n7i5x9/driver-drowsiness-dataset",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--img-size", type=int, default=224, help="Input image size")
    parser.add_argument("--workers", type=int, default=4, help="Data loading workers")
    parser.add_argument(
        "--patience", type=int, default=15, help="Early stopping patience"
    )
    parser.add_argument(
        "--grad-clip", type=float, default=1.0, help="Gradient clipping value"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output-dir", type=str, default="./public/models", help="Output directory"
    )
    parser.add_argument(
        "--no-mixed-precision",
        action="store_true",
        help="Disable mixed precision training",
    )
    parser.add_argument(
        "--no-tensorboard", action="store_true", help="Disable TensorBoard logging"
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        default=True,
        help="Freeze backbone during initial training",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.1,
        help="Label smoothing factor",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["onecycle", "cosine", "plateau"],
        help="Learning rate scheduler type",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=5,
        help="Number of warmup epochs for cosine scheduler",
    )
    parser.add_argument(
        "--mixup",
        default=True,
        action="store_true",
        help="Use Mixup data augmentation",
    )
    parser.add_argument(
        "--mixup-alpha",
        type=float,
        default=0.2,
        help="Mixup alpha parameter",
    )
    parser.add_argument(
        "--cutmix",
        default=False,
        action="store_true",
        help="Use CutMix data augmentation",
    )
    parser.add_argument(
        "--cutmix-alpha",
        type=float,
        default=1.0,
        help="CutMix alpha parameter",
    )
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(args):
    """Load and prepare dataset"""
    print("Loading dataset from Hugging Face...")

    try:
        dataset = load_dataset(args.dataset)
        train_data = dataset["train"]
        val_data = dataset["validation"]
        test_data = dataset["test"]

        print(
            f"Data split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}"
        )

    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

    return train_data, val_data, test_data


def create_transforms(img_size: int):
    """Create data augmentation transforms for training and validation."""
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # Enhanced training augmentations for better generalization
    train_transforms = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.3, hue=0.15
            ),
            transforms.RandomRotation(degrees=20),
            transforms.RandomAffine(
                degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(
                p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value="random"
            ),  # Random occlusion for robustness
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0)),
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    return train_transforms, val_transforms


def collate_fn(examples, transform_func):
    """Efficient collate function for data loading"""
    images = []
    labels = []
    for x in examples:
        img = x["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        img_tensor = transform_func(img)
        images.append(img_tensor)
        labels.append(int(x["label"]))

    return torch.stack(images, dim=0), torch.tensor(labels, dtype=torch.long)


def create_dataloaders(train_data, val_data, args, train_transforms, val_transforms):
    """Create optimized DataLoaders"""
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, train_transforms),
        num_workers=args.workers,
        pin_memory=True,
        prefetch_factor=2 if args.workers > 0 else None,
        persistent_workers=args.workers > 0,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, val_transforms),
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    return train_loader, val_loader


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        # Channel attention
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        # Spatial attention
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        # Channel Attention: aggregate spatial info via pooling
        b, c, _, _ = x.size()
        avg_out = F.adaptive_avg_pool2d(x, 1).view(b, c)
        max_out = F.adaptive_max_pool2d(x, 1).view(b, c)

        # Combine avg and max pooled features through FC layers
        avg_att = self.fc(avg_out)
        max_att = self.fc(max_out)
        channel_att = torch.sigmoid(avg_att + max_att).view(b, c, 1, 1)

        x = x * channel_att

        # Spatial Attention: concatenate channel-wise avg and max, then convolve
        avg_s = torch.mean(x, dim=1, keepdim=True)
        max_s, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = torch.sigmoid(self.conv_spatial(torch.cat([avg_s, max_s], dim=1)))

        return x * spatial_att


class DDDFatigueNet(nn.Module):
    """Fatigue detection model based on MobileNetV3 with CBAM attention."""

    def __init__(self, num_classes=2, dropout_rate=0.5):
        super().__init__()
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
        base_model = models.mobilenet_v3_large(weights=weights)

        # Extract backbone (feature extractor)
        self.backbone = base_model.features
        backbone_output_channels = 960

        # Add CBAM attention after backbone
        self.attention = CBAM(backbone_output_channels)

        # Classifier: global pooling followed by FC layers
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone_output_channels, 512),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(256, num_classes),
        )

        self._init_classifier()

    def _init_classifier(self):
        """Initialize classifier weights with Kaiming initialization."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.backbone(x)
        x = self.attention(x)
        x = self.pool(x)  # 池化
        return self.classifier(x)


class FocalLoss(nn.Module):
    """Focal Loss with label smoothing for handling class imbalance."""

    def __init__(self, alpha=1, gamma=2, reduction="mean", label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        # Apply label smoothing
        if self.label_smoothing > 0:
            n_classes = inputs.size(1)
            targets = F.one_hot(targets, n_classes) * (1 - self.label_smoothing)
            targets = targets + self.label_smoothing / n_classes

        ce_loss = F.cross_entropy(
            inputs, targets, reduction="none", label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        return focal_loss.sum()


def train_one_epoch(
    device_type,
    model,
    loader,
    optimizer,
    criterion,
    device,
    scaler,
    grad_clip,
    writer,
    epoch,
    use_mixup=False,
    mixup_alpha=0.2,
    use_cutmix=False,
    cutmix_alpha=1.0,
):
    """Train for one epoch with mixed precision and optional Mixup/CutMix."""
    model.train()
    model.to(device)
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Train]")
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)

        # Apply Mixup or CutMix augmentation
        if use_mixup:
            images, labels_a, labels_b, lam = mixup_data(images, labels, mixup_alpha)
            loss = mixup_criterion(criterion, model(images), labels_a, labels_b, lam)
        elif use_cutmix:
            images, labels_a, labels_b, lam = cutmix(images, labels, cutmix_alpha)
            loss = mixup_criterion(criterion, model(images), labels_a, labels_b, lam)
        else:
            # Standard training without augmentation
            with autocast(device_type=device_type, enabled=(scaler is not None)):
                outputs = model(images)
                loss = criterion(outputs, labels)

        optimizer.zero_grad()

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        running_loss += loss.item()

        # Calculate accuracy
        if use_mixup or use_cutmix:
            # For mixed samples, count as correct if prediction matches either label
            with torch.no_grad():
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                correct_mask = (predicted == labels_a) | (predicted == labels_b)
                correct += correct_mask.sum().item()
        else:
            with torch.no_grad():
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()

        total += labels.size(0)

        # Update progress bar with current metrics
        pbar.set_postfix(
            {"loss": f"{loss.item():.4f}", "acc": f"{100 * correct / total:.2f}%"}
        )

        # Log batch metrics to TensorBoard
        if writer and batch_idx % 10 == 0:
            global_step = epoch * len(loader) + batch_idx
            writer.add_scalar("Loss/batch", loss.item(), global_step)
            writer.add_scalar("Accuracy/batch", 100 * correct / total, global_step)

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, loader, criterion, device, writer, epoch):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Val]  ")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Update progress bar with current metrics
        pbar.set_postfix(
            {"loss": f"{loss.item():.4f}", "acc": f"{100 * correct / total:.2f}%"}
        )

    val_loss = running_loss / len(loader)
    val_acc = 100 * correct / total

    # Log validation metrics to TensorBoard
    if writer:
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)

    return val_loss, val_acc


def save_checkpoint(model, optimizer, scheduler, epoch, val_acc, args, filename="best"):
    """Save model checkpoint with metadata"""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "val_accuracy": val_acc,
        "args": vars(args),
        "timestamp": datetime.now().isoformat(),
    }

    # Save full checkpoint
    checkpoint_path = output_dir / f"ddd_fatigue_v3_{filename}_checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)

    # Export to ONNX - create a temporary CPU copy without affecting training model
    # Use state_dict to avoid disrupting CUDA context
    dummy_input = torch.randn(1, 3, args.img_size, args.img_size)
    onnx_path = output_dir / f"ddd_fatigue_v3_{filename}.onnx"

    try:
        # Create a new model instance on CPU for export
        model_for_export = DDDFatigueNet(num_classes=2)
        model_for_export.load_state_dict(model.state_dict())
        model_for_export.eval()

        torch.onnx.export(
            model_for_export,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            verbose=False,
            dynamo=False,
        )
        print(f"  -> ONNX exported to {onnx_path}")
    except Exception as e:
        print(f"  -> ONNX export failed: {e}")

    return checkpoint_path


@torch.no_grad()
def evaluate_test_set(model, loader, device, output_dir):
    """Evaluate model on test set and save detailed metrics."""
    model.eval()
    all_preds = []
    all_labels = []

    print("\n[Analysis] Running full evaluation on Test Set...")
    for images, labels in tqdm(loader, desc="Evaluating"):
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

    # Generate classification report (Precision, Recall, F1)
    report = classification_report(
        all_labels, all_preds, target_names=["Normal", "Fatigue"]
    )
    print("\nFinal Classification Report:")
    print(report)

    # Save report to file
    with open(Path(output_dir) / "test_report.txt", "w") as f:
        f.write(report)

    # Generate confusion matrix visualization
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Fatigue"],
        yticklabels=["Normal", "Fatigue"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Fatigue Detection")

    plot_path = Path(output_dir) / "confusion_matrix.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"  -> Analysis plots saved to {output_dir}")

    return f1_score(all_labels, all_preds)


def main():
    args = parse_args()
    set_seed(args.seed)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"Mixed precision: {'Disabled' if args.no_mixed_precision else 'Enabled'}"
        )

    # Setup TensorBoard
    writer = None
    if TENSORBOARD_AVAILABLE and not args.no_tensorboard:
        log_dir = f"./logs/train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        writer = SummaryWriter(log_dir)
        print(f"TensorBoard logs: {log_dir}")

    # Load data
    train_data, val_data, test_data = load_data(args)
    train_transforms, val_transforms = create_transforms(args.img_size)
    train_loader, val_loader = create_dataloaders(
        train_data, val_data, args, train_transforms, val_transforms
    )

    # Create model
    model = DDDFatigueNet(num_classes=2).to(device)

    # Optionally freeze backbone for initial training
    if args.freeze_backbone:
        print("Freezing backbone layers...")
        for param in model.backbone.parameters():
            param.requires_grad = False
        # Only train attention and classifier modules
        for param in model.attention.parameters():
            param.requires_grad = True
        for param in model.classifier.parameters():
            param.requires_grad = True

    # Optimizer with better defaults
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # Learning rate scheduler based on args
    steps_per_epoch = len(train_loader)
    if args.scheduler == "onecycle":
        scheduler = OneCycleLR(
            optimizer,
            max_lr=args.lr,
            epochs=args.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.3,
            anneal_strategy="cos",
            cycle_momentum=True,
        )
        print(f"\n[Scheduler] Using OneCycleLR (max_lr={args.lr})")
    elif args.scheduler == "cosine":
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=args.lr / 100
        )
        print(f"\n[Scheduler] Using CosineAnnealingWarmRestarts (T_0=10)")
    else:  # plateau
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-7
        )
        print(f"\n[Scheduler] Using ReduceLROnPlateau (factor=0.5, patience=5)")

    # Focal Loss with label smoothing and class weights
    class_counts = [count for label, count in Counter(train_data["label"]).items()]
    alpha = max(class_counts) / min(class_counts)  # Weight for minority class
    criterion = FocalLoss(alpha=alpha, gamma=2, label_smoothing=args.label_smoothing)

    # Mixed precision scaler
    scaler = (
        GradScaler()
        if (torch.cuda.is_available() and not args.no_mixed_precision)
        else None
    )

    # Training tracking
    best_val_acc = 0.0
    patience_counter = 0
    early_stop_threshold = args.patience

    # Data verification: check first batch
    print("\n[Debug] Verifying data loader...")
    sample_batch = next(iter(train_loader))
    sample_images, sample_labels = sample_batch
    print(f"  Batch shape: {sample_images.shape}")
    print(f"  Labels: {sample_labels.tolist()}")
    print(f"  Label distribution in first batch: {Counter(sample_labels.tolist())}")
    print(
        f"  Image value range: [{sample_images.min():.3f}, {sample_images.max():.3f}]"
    )
    print("-" * 80)

    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Batch size: {args.batch_size}, LR: {args.lr}, Image size: {args.img_size}")
    print(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")
    print("-" * 80)

    # Two-stage training: if backbone is frozen, train classifier first
    if args.freeze_backbone:
        print("\n[Stage 1] Training classifier only (backbone frozen)...")
        freeze_epochs = min(10, args.epochs)
        for epoch in range(freeze_epochs):
            train_loss, train_acc = train_one_epoch(
                device_type,
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                scaler,
                args.grad_clip,
                writer,
                epoch,
                use_mixup=args.mixup,
                mixup_alpha=args.mixup_alpha,
                use_cutmix=args.cutmix,
                cutmix_alpha=args.cutmix_alpha,
            )

            val_loss, val_acc = validate(
                model, val_loader, criterion, device, writer, epoch
            )

            print(
                f"Epoch [{epoch+1:3d}/{freeze_epochs}] | "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:6.2f}% | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:6.2f}%"
            )

        # Unfreeze backbone for fine-tuning
        print("\n[Stage 2] Unfreezing backbone for fine-tuning...")
        for param in model.backbone.parameters():
            param.requires_grad = True

        # Reduce learning rate for fine-tuning
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr / 10

        print(f"  Learning rate adjusted to: {args.lr / 10}")
        start_epoch = freeze_epochs
    else:
        start_epoch = 0

    # Full training / fine-tuning
    print(f"\n[Final Training] Starting from epoch {start_epoch + 1}...")
    print(f"[Data Aug] Mixup: {args.mixup}, CutMix: {args.cutmix}")
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss, train_acc = train_one_epoch(
            device_type,
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler,
            args.grad_clip,
            writer,
            epoch,
            use_mixup=args.mixup,
            mixup_alpha=args.mixup_alpha,
            use_cutmix=args.cutmix,
            cutmix_alpha=args.cutmix_alpha,
        )

        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, writer, epoch
        )

        # Log epoch summary
        print(
            f"Epoch [{epoch+1:3d}/{args.epochs}] | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:6.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:6.2f}%"
        )

        # Update learning rate scheduler
        if args.scheduler == "plateau":
            scheduler.step(val_loss)
        elif args.scheduler == "cosine":
            scheduler.step()
        else:  # onecycle
            scheduler.step()

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, scheduler, epoch, val_acc, args, "best")
            print(f"  -> New best model! (Val Acc: {val_acc:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_threshold:
                print(f"\nEarly stopping triggered after {epoch+1} epochs.")
                print(f"  Best Val Acc: {best_val_acc:.2f}%")
                break

        # Log learning rate
        if writer:
            current_lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar("LearningRate", current_lr, epoch)

        # Print current learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"  -> Current LR: {current_lr:.6f}")

    print("-" * 80)
    print(f"Training finished. Best Validation Accuracy: {best_val_acc:.2f}%")

    # Load best model for final export
    best_checkpoint_path = Path(args.output_dir) / "ddd_fatigue_v3_best_checkpoint.pt"
    if best_checkpoint_path.exists():
        checkpoint = torch.load(best_checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded best model from epoch {checkpoint['epoch']}")

    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda x: collate_fn(
            x, val_transforms
        ),  # Use val transforms for test
        num_workers=args.workers,
        pin_memory=True,
    )

    # Run full evaluation
    final_f1 = evaluate_test_set(model, test_loader, device, log_dir)
    print(f"\n[Success] Evaluation Complete. Final Test F1-Score: {final_f1:.4f}")

    # Final ONNX export - create fresh CPU model to avoid CUDA context issues
    print("\nExporting final model to ONNX...")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "ddd_fatigue_v3.onnx"

    # Create a new CPU model for export
    model_for_export = DDDFatigueNet(num_classes=2)
    model_for_export.load_state_dict(model.state_dict())
    model_for_export.eval()

    dummy_input = torch.randn(1, 3, args.img_size, args.img_size)

    try:
        torch.onnx.export(
            model_for_export,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            verbose=False,
            dynamo=False,
        )
        print(f"Final model exported to {output_path}")
    except Exception as e:
        print(f"Error during ONNX export: {e}")

    if writer:
        writer.close()
        print(f"\nTensorBoard logs available at: ./logs/")


if __name__ == "__main__":
    main()
