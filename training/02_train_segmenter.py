import os
import random
from pathlib import Path
from typing import Set

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def prepare_segmentation_crops(
    raw_dataset_dir: Path,
    output_dir: Path,
    train_seqs: Set[str],
    val_seqs: Set[str],
):
    train_img_dir = output_dir / "train" / "images"
    train_mask_dir = output_dir / "train" / "masks"
    val_img_dir = output_dir / "val" / "images"
    val_mask_dir = output_dir / "val" / "masks"

    if train_img_dir.exists() and any(train_img_dir.iterdir()):
        print(f"[Dataset] Using existing patch dataset at: {output_dir}")
        return

    for p in [train_img_dir, train_mask_dir, val_img_dir, val_mask_dir]:
        p.mkdir(parents=True, exist_ok=True)

    all_seqs = train_seqs | val_seqs
    found_seq_dirs = []

    for root, _, _ in os.walk(raw_dataset_dir):
        inst_dir = Path(root)
        folder_name = inst_dir.name
        if folder_name in all_seqs and inst_dir.parent.name == "instances":
            parts = list(inst_dir.parts)
            idx = parts.index("instances")
            parts[idx] = "images"
            img_dir = Path(*parts)
            if img_dir.exists():
                found_seq_dirs.append((folder_name, inst_dir, img_dir))

    print(f"[Dataset] Found {len(found_seq_dirs)} sequences. Extracting crops...")
    crop_stats = {"train": 0, "val": 0}

    for folder_name, inst_dir, img_dir in tqdm(found_seq_dirs, desc="Sequences"):
        split = "val" if folder_name in val_seqs else "train"
        dst_imgs = val_img_dir if split == "val" else train_img_dir
        dst_masks = val_mask_dir if split == "val" else train_mask_dir

        mask_files = sorted(list(inst_dir.glob("*.png")))
        for mask_p in mask_files:
            img_p = img_dir / mask_p.name
            if not img_p.exists():
                img_p = img_dir / f"{mask_p.stem}.jpg"
            if not img_p.exists():
                continue

            mask_np = np.array(Image.open(mask_p))
            img_pil = Image.open(img_p).convert("RGB")
            img_w, img_h = img_pil.size

            unique_objs = np.unique(mask_np)
            for obj_id in unique_objs:
                if obj_id == 0 or obj_id >= 10000:
                    continue

                pos = np.where(mask_np == obj_id)
                if len(pos[0]) == 0:
                    continue

                y1, y2 = np.min(pos[0]), np.max(pos[0])
                x1, x2 = np.min(pos[1]), np.max(pos[1])
                bw = (x2 - x1) + 1
                bh = (y2 - y1) + 1

                if bw < 6 or bh < 6:
                    continue

                pad_x = int(bw * 0.15)
                pad_y = int(bh * 0.15)
                cx1 = max(0, x1 - pad_x)
                cy1 = max(0, y1 - pad_y)
                cx2 = min(img_w, x2 + pad_x)
                cy2 = min(img_h, y2 + pad_y)

                crop_img = img_pil.crop((cx1, cy1, cx2, cy2)).resize((128, 128), Image.BILINEAR)
                binary_mask = (mask_np[cy1:cy2, cx1:cx2] == obj_id).astype(np.uint8) * 255
                crop_mask = Image.fromarray(binary_mask).resize((128, 128), Image.NEAREST)

                fname = f"seq{folder_name}_{mask_p.stem}_obj{obj_id}.png"
                crop_img.save(dst_imgs / fname)
                crop_mask.save(dst_masks / fname)
                crop_stats[split] += 1

    print(f"[Dataset] Ready: Train crops: {crop_stats['train']} | Val crops: {crop_stats['val']}")


class ApplePatchDataset(Dataset):
    def __init__(self, root_dir: Path, is_train: bool = True):
        self.img_dir = root_dir / "images"
        self.mask_dir = root_dir / "masks"
        self.files = sorted([f.name for f in self.img_dir.glob("*.png")])
        self.is_train = is_train

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        img = Image.open(self.img_dir / f).convert("RGB")
        mask = Image.open(self.mask_dir / f).convert("L")

        if self.is_train:
            if np.random.rand() > 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
            if np.random.rand() > 0.5:
                img = TF.vflip(img)
                mask = TF.vflip(mask)
            if np.random.rand() > 0.5:
                angle = float(np.random.uniform(-25, 25))
                img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR)
                mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)
            if np.random.rand() > 0.4:
                img = TF.adjust_brightness(img, float(np.random.uniform(0.75, 1.25)))
                img = TF.adjust_contrast(img, float(np.random.uniform(0.75, 1.25)))
                img = TF.adjust_saturation(img, float(np.random.uniform(0.7, 1.3)))

        img_t = TF.to_tensor(img)
        img_t = TF.normalize(img_t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        mask_np = np.array(mask, dtype=np.float32) / 255.0
        mask_t = torch.from_numpy(mask_np).unsqueeze(0)
        return img_t, mask_t


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class AppleUNet128(nn.Module):
    def __init__(self):
        super().__init__()
        self.inc = DoubleConv(3, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 256))

        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(128 + 256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(64 + 128, 64)
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(32 + 64, 32)
        self.up4 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(32 + 32, 32)
        self.outc = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        d1 = self.up1(x5)
        d1 = self.conv1(torch.cat([d1, x4], dim=1))
        d2 = self.up2(d1)
        d2 = self.conv2(torch.cat([d2, x3], dim=1))
        d3 = self.up3(d2)
        d3 = self.conv3(torch.cat([d3, x2], dim=1))
        d4 = self.up4(d3)
        d4 = self.conv4(torch.cat([d4, x1], dim=1))
        return self.outc(d4)


class DiceBCELoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.weight = bce_weight

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        smooth = 1.0
        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice_loss = 1.0 - ((2.0 * intersection + smooth) / (union + smooth)).mean()
        return self.weight * bce_loss + (1.0 - self.weight) * dice_loss


def calculate_iou(logits, targets, thresh=0.5):
    preds = (torch.sigmoid(logits) > thresh).float()
    inter = (preds * targets).sum(dim=(2, 3))
    union = (preds + targets).clamp(0, 1).sum(dim=(2, 3))
    iou = (inter + 1e-6) / (union + 1e-6)
    return iou.mean().item()


def plot_metrics(history: dict, output_path: Path):
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(7, 6))

    plt.subplot(2, 1, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss", color="#1f77b4", linewidth=1.7)
    plt.plot(epochs, history["val_loss"], label="Val Loss", color="#ff7f0e", linewidth=1.7)
    plt.ylabel("Loss", fontsize=11)
    plt.grid(True, linestyle="-", alpha=0.6)
    plt.legend(loc="upper right", frameon=True)

    plt.subplot(2, 1, 2)
    plt.plot(epochs, [v * 100 for v in history["train_iou"]], label="Train IoU", color="#2ca02c", linewidth=1.7)
    plt.plot(epochs, [v * 100 for v in history["val_iou"]], label="Val IoU", color="#d62728", linewidth=1.7)
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("IoU (%)", fontsize=11)
    plt.grid(True, linestyle="-", alpha=0.6)
    plt.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Plot] Metrics plot saved to: {output_path}")


def main():
    seed = 42
    set_seed(seed)
    print(f"[Reproducibility] Random seed fixed to: {seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Hardware] Execution device: {device}")

    # Sequence-level split without temporal data leakage
    train_seqs = {"0000", "0001", "0002", "0003", "0004"}
    val_seqs = {"0005"}
    print(f"[Split] Train sequences: {sorted(list(train_seqs))}")
    print(f"[Split] Val sequences:   {sorted(list(val_seqs))}")

    base_dir = Path(__file__).resolve().parent.parent
    data_source_dir = base_dir / "data" / "APPLE_MOTS"
    crop_dir = base_dir / "data" / "processed_segmentation"
    weights_dir = base_dir / "weights"
    assets_dir = base_dir / "assets"

    weights_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    prepare_segmentation_crops(data_source_dir, crop_dir, train_seqs, val_seqs)

    train_ds = ApplePatchDataset(crop_dir / "train", is_train=True)
    val_ds = ApplePatchDataset(crop_dir / "val", is_train=False)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=32,
        shuffle=True,
        num_workers=2,
        pin_memory=True if device.type == "cuda" else False,
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=True if device.type == "cuda" else False,
        worker_init_fn=seed_worker,
    )

    model = AppleUNet128().to(device)

    # Verification check
    test_dummy = torch.randn(2, 3, 128, 128, device=device)
    with torch.no_grad():
        out_dummy = model(test_dummy)
    assert out_dummy.shape == (2, 1, 128, 128), f"Dimension mismatch: {out_dummy.shape}"
    print(f"[Model] Dimension check passed: {tuple(test_dummy.shape)} -> {tuple(out_dummy.shape)}")

    criterion = DiceBCELoss(bce_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-5)

    epochs = 15
    best_val_iou = 0.0
    best_model_path = weights_dir / "apple_crop_unet_best.pth"

    history = {"train_loss": [], "val_loss": [], "train_iou": [], "val_iou": []}

    print("\n" + "=" * 60)
    print("           STARTING SEGMENTER TRAINING (15 EPOCHS)           ")
    print("=" * 60)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_iou = 0.0, 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs} [Train]")
        for imgs, masks in pbar:
            imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_iou += calculate_iou(logits, masks)
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        train_loss /= len(train_loader)
        train_iou /= len(train_loader)
        scheduler.step()

        model.eval()
        val_loss, val_iou = 0.0, 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)
                logits = model(imgs)
                loss = criterion(logits, masks)
                val_loss += loss.item()
                val_iou += calculate_iou(logits, masks)

        val_loss /= len(val_loader)
        val_iou /= len(val_loader)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_iou"].append(train_iou)
        history["val_iou"].append(val_iou)

        print(
            f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f}, IoU: {train_iou*100:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou*100:.2f}%"
        )

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), best_model_path)


    print("\n" + "=" * 60)
    print(f"Training Complete! Peak Val IoU: {best_val_iou*100:.2f}%")
    print(f"Best model weights saved to: {best_model_path}")
    print("=" * 60)

    plot_metrics(history, assets_dir / "segmenter_training_metrics.png")


if __name__ == "__main__":
    main()