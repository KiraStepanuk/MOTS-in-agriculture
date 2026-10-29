import os
import math
import shutil
import random
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms as T
from torchvision.models import resnet18, ResNet18_Weights


SEED = 42


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def set_epoch_seed(seed: int, epoch: int):
    ep_seed = seed + epoch * 1000
    random.seed(ep_seed)
    np.random.seed(ep_seed)
    torch.manual_seed(ep_seed)
    torch.cuda.manual_seed_all(ep_seed)


set_seed(SEED)


def extract_context_crop(img: np.ndarray, box: np.ndarray, scale: float = 2.2) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    bw, bh = (x2 - x1), (y2 - y1)
    nw, nh = bw * scale, bh * scale
    nx1 = max(0, int(cx - nw * 0.5))
    ny1 = max(0, int(cy - nh * 0.5))
    nx2 = min(w, int(cx + nw * 0.5))
    ny2 = min(h, int(cy + nh * 0.5))
    if (nx2 - nx1) < 10 or (ny2 - ny1) < 10:
        return img[int(y1):int(y2), int(x1):int(x2)]
    return img[ny1:ny2, nx1:nx2]


def generate_crops(sequences, target_dir: Path, min_crops_per_id: int = 4):
    id_counts = defaultdict(int)
    for seq, inst_dir, img_dir in sequences:
        mask_files = sorted([f for f in os.listdir(inst_dir) if f.endswith(".png")])
        for f in tqdm(mask_files, desc=f"Seq {seq}"):
            fid = os.path.splitext(f)[0]
            src_img_p = os.path.join(img_dir, f"{fid}.png")
            if not os.path.exists(src_img_p):
                src_img_p = os.path.join(img_dir, f"{fid}.jpg")
            if not os.path.exists(src_img_p):
                continue

            img = cv2.imread(src_img_p)
            if img is None:
                continue

            mask = np.array(Image.open(os.path.join(inst_dir, f)))
            for obj_id in np.unique(mask):
                if obj_id == 0 or obj_id >= 10000:
                    continue
                pos = np.where(mask == obj_id)
                if len(pos[0]) == 0:
                    continue

                ymin, ymax = float(np.min(pos[0])), float(np.max(pos[0]))
                xmin, xmax = float(np.min(pos[1])), float(np.max(pos[1]))
                if (xmax - xmin) < 6 or (ymax - ymin) < 6:
                    continue

                box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
                crop = extract_context_crop(img, box, scale=2.2)
                if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                    continue

                global_id = f"seq_{seq}_id_{obj_id}"
                save_subfolder = target_dir / global_id
                save_subfolder.mkdir(parents=True, exist_ok=True)
                crop_path = save_subfolder / f"{fid}.jpg"
                cv2.imwrite(str(crop_path), crop)
                id_counts[global_id] += 1

    valid_classes = {k for k, v in id_counts.items() if v >= min_crops_per_id}
    for folder in list(target_dir.iterdir()):
        if folder.is_dir() and folder.name not in valid_classes:
            shutil.rmtree(folder)
    return valid_classes


def setup_reid_data(raw_data_dir: Path, crops_root: Path):
    train_dir = crops_root / "train"
    val_dir = crops_root / "val"

    if train_dir.exists() and any(train_dir.iterdir()):
        print(f"[Dataset] Found existing Re-ID crops at: {crops_root}")
        return train_dir, val_dir

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    train_seqs = {"0000", "0001", "0002", "0003", "0004"}
    val_seqs = {"0005"}
    strict_exclude = {"0006", "0007", "0008"}

    found_train = []
    found_val = []

    for root, dirs, files in os.walk(raw_data_dir):
        folder_name = os.path.basename(root)
        if folder_name in strict_exclude:
            continue
        if folder_name in train_seqs or folder_name in val_seqs:
            if os.path.basename(os.path.dirname(root)) == "instances":
                inst_dir = root
                img_dir = root.replace(f"/instances/{folder_name}", f"/images/{folder_name}").replace(f"\\instances\\{folder_name}", f"\\images\\{folder_name}")
                if os.path.exists(img_dir):
                    if folder_name in train_seqs:
                        found_train.append((folder_name, inst_dir, img_dir))
                    elif folder_name in val_seqs:
                        found_val.append((folder_name, inst_dir, img_dir))

    found_train.sort(key=lambda x: x[0])
    found_val.sort(key=lambda x: x[0])

    print("[Dataset] Generating train identity crops...")
    train_classes = generate_crops(found_train, train_dir, min_crops_per_id=4)
    print("[Dataset] Generating val identity crops...")
    val_classes = generate_crops(found_val, val_dir, min_crops_per_id=4)
    print(f"[Dataset] Ready: Train IDs: {len(train_classes)} | Val IDs: {len(val_classes)}")
    return train_dir, val_dir


class AppleReIDDataset(Dataset):
    def __init__(self, root_dir: Path, transform=None):
        self.samples = []
        self.transform = transform
        self.class_to_indices = defaultdict(list)
        subdirs = sorted([d for d in Path(root_dir).iterdir() if d.is_dir()])
        self.class_to_idx = {d.name: idx for idx, d in enumerate(subdirs)}
        idx_counter = 0

        for d in subdirs:
            c_idx = self.class_to_idx[d.name]
            for img_path in sorted(d.glob("*.jpg")):
                digits = "".join(filter(str.isdigit, img_path.stem))
                frame_idx = int(digits) if digits else 0
                self.samples.append((str(img_path), c_idx, frame_idx))
                self.class_to_indices[c_idx].append(idx_counter)
                idx_counter += 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, frame_idx = self.samples[idx]
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((128, 128, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)
        return img, label, frame_idx


class BalancedIdentitySampler(Sampler):
    def __init__(self, dataset: AppleReIDDataset, num_classes_per_batch=16, num_instances_per_class=4):
        self.dataset = dataset
        self.p = num_classes_per_batch
        self.k = num_instances_per_class
        self.batch_size = self.p * self.k
        self.classes = sorted(list(self.dataset.class_to_indices.keys()))

    def __iter__(self):
        class_pool = list(self.classes)
        random.shuffle(class_pool)
        class_idx_ptrs = {c: 0 for c in self.classes}
        class_indices_shuffled = {c: list(self.dataset.class_to_indices[c]) for c in self.classes}
        for c in self.classes:
            random.shuffle(class_indices_shuffled[c])

        batches = []
        while len(class_pool) >= self.p:
            selected_classes = class_pool[:self.p]
            class_pool = class_pool[self.p:]
            batch = []
            for c in selected_classes:
                indices = class_indices_shuffled[c]
                ptr = class_idx_ptrs[c]
                if ptr + self.k <= len(indices):
                    selected = indices[ptr:ptr + self.k]
                    class_idx_ptrs[c] += self.k
                else:
                    selected = indices[ptr:]
                    random.shuffle(indices)
                    needed = self.k - len(selected)
                    selected += indices[:needed]
                    class_idx_ptrs[c] = needed
                batch.extend(selected)
            batches.extend(batch)
        return iter(batches)

    def __len__(self):
        return (len(self.classes) // self.p) * self.batch_size


class AppleFaceNet(nn.Module):
    def __init__(self, emb_dim: int = 128):
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.projector = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, emb_dim),
            nn.BatchNorm1d(emb_dim, affine=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x).flatten(1)
        emb = self.projector(feat)
        return F.normalize(emb, p=2, dim=1)


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features=128, out_features=100, s=16.0, m=0.25):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, emb: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(emb, F.normalize(self.weight, p=2, dim=1))
        sine = torch.sqrt(torch.clamp(1.0 - cosine ** 2, min=1e-7))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1.0)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return output * self.s


class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin=0.25):
        super().__init__()
        self.margin = margin

    def forward(self, embs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        dist_mat = torch.clamp(2.0 - 2.0 * torch.matmul(embs, embs.t()), min=1e-8).sqrt()
        is_pos = labels.unsqueeze(0) == labels.unsqueeze(1)
        is_neg = labels.unsqueeze(0) != labels.unsqueeze(1)
        hard_pos = (dist_mat * is_pos.float()).max(dim=1)[0]
        hard_neg = dist_mat.clone()
        hard_neg[~is_neg] = 1e5
        hard_neg = hard_neg.min(dim=1)[0]
        loss = F.relu(hard_pos - hard_neg + self.margin)
        return loss.mean()


@torch.no_grad()
def evaluate_reid_on_val_strict(model, loader, device, temporal_gap=5):
    model.eval()
    all_embs, all_labels, all_fids = [], [], []

    for batch in loader:
        imgs, labels, fids = batch
        imgs = imgs.to(device)
        embs = model(imgs)
        all_embs.append(embs.cpu().numpy())
        all_labels.append(labels.numpy())
        all_fids.append(fids.numpy())

    all_embs = np.concatenate(all_embs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_fids = np.concatenate(all_fids, axis=0)

    sim_mat = np.dot(all_embs, all_embs.T)
    total = len(all_labels)
    rank1_hits = 0
    valid_evals = 0

    for i in range(total):
        same_id_close_frame = (all_labels == all_labels[i]) & (np.abs(all_fids - all_fids[i]) < temporal_gap)
        sim_row = sim_mat[i].copy()
        sim_row[same_id_close_frame] = -1.0

        valid_positives = (all_labels == all_labels[i]) & (np.abs(all_fids - all_fids[i]) >= temporal_gap)
        if not np.any(valid_positives):
            continue

        valid_evals += 1
        top1_match = np.argmax(sim_row)
        if all_labels[top1_match] == all_labels[i]:
            rank1_hits += 1

    return (rank1_hits / valid_evals) * 100.0 if valid_evals > 0 else 0.0


def plot_metrics(history: dict, output_path: Path):
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(7, 6))

    plt.subplot(2, 1, 1)
    plt.plot(epochs, history["train_loss"], label="Train Triplet Loss", color="#1f77b4", linewidth=1.7)
    plt.plot(epochs, history["val_loss"], label="Val Triplet Loss", color="#ff7f0e", linewidth=1.7)
    plt.ylabel("Triplet Loss", fontsize=11)
    plt.grid(True, linestyle="-", alpha=0.6)
    plt.legend(loc="upper right", frameon=True)

    plt.subplot(2, 1, 2)
    plt.plot(epochs, history["train_acc"], label="Train Acc (%)", color="#2ca02c", linewidth=1.7)
    plt.plot(epochs, history["val_rank1"], label="Val Rank-1 (%)", color="#d62728", linewidth=1.7)
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Accuracy / Rank-1 (%)", fontsize=11)
    plt.grid(True, linestyle="-", alpha=0.6)
    plt.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Plot] Metrics plot saved to: {output_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Hardware] Execution device: {device}")

    base_dir = Path.cwd()
    raw_data_dir = base_dir / "data" / "APPLE_MOTS"
    crops_root = base_dir / "data" / "apple_reid_crops"
    weights_dir = base_dir / "weights"
    assets_dir = base_dir / "assets"

    weights_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    train_dir, val_dir = setup_reid_data(raw_data_dir, crops_root)

    train_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((128, 128)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=180),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.RandomErasing(p=0.3, scale=(0.02, 0.2), value="random"),
    ])

    val_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((128, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = AppleReIDDataset(train_dir, transform=train_transform)
    val_ds = AppleReIDDataset(val_dir, transform=val_transform)
    num_train_classes = len(train_ds.class_to_idx)

    train_sampler = BalancedIdentitySampler(train_ds, num_classes_per_batch=16, num_instances_per_class=4)
    train_loader = DataLoader(train_ds, batch_size=64, sampler=train_sampler, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

    model = AppleFaceNet(emb_dim=128).to(device)
    metric_head = ArcMarginProduct(in_features=128, out_features=num_train_classes, s=16.0, m=0.25).to(device)

    triplet_criterion = BatchHardTripletLoss(margin=0.25)
    ce_criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": 1e-4, "weight_decay": 1e-4},
        {"params": model.projector.parameters(), "lr": 5e-4, "weight_decay": 1e-4},
        {"params": metric_head.parameters(), "lr": 5e-4, "weight_decay": 1e-4},
    ])

    EPOCHS = 20
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_val_rank1 = 0.0
    best_weights_path = weights_dir / "apple_faceid_metric.pth"
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_rank1": []}

    print("\n" + "=" * 60)
    print(f"      STARTING RE-ID TRAINING ({EPOCHS} EPOCHS, {num_train_classes} CLASSES)      ")
    print("=" * 60)

    for epoch in range(1, EPOCHS + 1):
        set_epoch_seed(SEED, epoch)
        model.train()
        metric_head.train()
        total_train_triplet_loss = 0.0
        total_correct = 0
        total_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{EPOCHS} [Train]")
        for imgs, labels, _ in pbar:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            embs = model(imgs)
            logits = metric_head(embs, labels)

            loss_ce = ce_criterion(logits, labels)
            loss_triplet = triplet_criterion(embs, labels)
            loss = loss_ce + 1.0 * loss_triplet

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
            optimizer.step()

            batch_sz = imgs.size(0)
            total_train_triplet_loss += loss_triplet.item() * batch_sz
            

            with torch.no_grad():
                clean_logits = F.linear(embs, F.normalize(metric_head.weight, p=2, dim=1))
                total_correct += (clean_logits.argmax(dim=1) == labels).sum().item()

            total_samples += batch_sz
            pbar.set_postfix({"TripLoss": f"{loss_triplet.item():.4f}", "Total": f"{loss.item():.4f}"})

        scheduler.step()
        

        epoch_train_loss = total_train_triplet_loss / total_samples
        epoch_train_acc = (total_correct / total_samples) * 100.0

        model.eval()
        val_triplet_loss, val_count = 0.0, 0
        with torch.no_grad():
            for v_imgs, v_labels, _ in val_loader:
                v_imgs = v_imgs.to(device, non_blocking=True)
                v_labels = v_labels.to(device, non_blocking=True)
                v_embs = model(v_imgs)
                val_triplet_loss += triplet_criterion(v_embs, v_labels).item() * v_imgs.size(0)
                val_count += v_imgs.size(0)
        epoch_val_loss = val_triplet_loss / max(1, val_count)


        val_rank1 = evaluate_reid_on_val_strict(model, val_loader, device, temporal_gap=5)

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_rank1"].append(val_rank1)

        print(
            f"Epoch {epoch:02d} | Train Loss (Trip): {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}% | "
            f"Val Loss (Trip): {epoch_val_loss:.4f}, Val Rank-1: {val_rank1:.2f}%"
        )

        if val_rank1 > best_val_rank1:
            best_val_rank1 = val_rank1
            torch.save(model.state_dict(), best_weights_path)

    print("\n" + "=" * 60)
    print(f"Training Complete! Best Val Rank-1: {best_val_rank1:.2f}%")
    print(f"Best model weights saved to: {best_weights_path}")
    print("=" * 60)

    plot_metrics(history, assets_dir / "reid_training_metrics.png")


if __name__ == "__main__":
    main()