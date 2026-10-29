import os
import random
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from ultralytics import RTDETR
from ultralytics.utils.torch_utils import strip_optimizer


def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_detection_dataset(raw_dataset_dir: Path, output_dir: Path) -> Path:
    splits = ["train", "val", "test"]
    for s in splits:
        (output_dir / "images" / s).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / s).mkdir(parents=True, exist_ok=True)

    yaml_path = output_dir / "apple_detection.yaml"
    labels_train = output_dir / "labels" / "train"
    if yaml_path.exists() and any(labels_train.iterdir() if labels_train.exists() else []):
        print(f"[Dataset] Found existing prepared dataset at {output_dir}")
        return yaml_path

    train_seqs = {"0000", "0001", "0002", "0003", "0004"}
    val_seqs = {"0005"}
    test_seqs = {"0006", "0007", "0008"}

    found_sequences = []
    for root, _, _ in os.walk(raw_dataset_dir):
        folder_name = os.path.basename(root)
        if folder_name in train_seqs | val_seqs | test_seqs:
            if os.path.basename(os.path.dirname(root)) == "instances":
                inst_dir = Path(root)
                img_dir = Path(
                    str(root)
                    .replace(f"/instances/{folder_name}", f"/images/{folder_name}")
                    .replace(f"\\instances\\{folder_name}", f"\\images\\{folder_name}")
                )
                if img_dir.exists():
                    if folder_name in train_seqs:
                        split = "train"
                    elif folder_name in val_seqs:
                        split = "val"
                    else:
                        split = "test"
                    found_sequences.append((folder_name, split, inst_dir, img_dir))

    print(f"[Dataset] Found {len(found_sequences)} sequences. Converting masks to bounding boxes...")

    for seq, split, inst_dir, img_dir in tqdm(found_sequences, desc="Processing sequences"):
        mask_files = sorted(list(inst_dir.glob("*.png")))
        dst_img_dir = output_dir / "images" / split
        dst_lbl_dir = output_dir / "labels" / split

        for mask_path in mask_files:
            fid = mask_path.stem
            src_img = img_dir / f"{fid}.png"
            if not src_img.exists():
                src_img = img_dir / f"{fid}.jpg"
            if not src_img.exists():
                continue

            mask = np.array(Image.open(mask_path))
            h_img, w_img = mask.shape[:2]
            records = []

            for obj_id in np.unique(mask):
                if obj_id == 0 or obj_id >= 10000:
                    continue
                pos = np.where(mask == obj_id)
                if len(pos[0]) == 0:
                    continue

                ymin, ymax = float(np.min(pos[0])), float(np.max(pos[0]))
                xmin, xmax = float(np.min(pos[1])), float(np.max(pos[1]))
                bw, bh = (xmax - xmin) + 1.0, (ymax - ymin) + 1.0

                if bw < 4 or bh < 4:
                    continue

                xc = np.clip(((xmin + xmax) / 2.0) / w_img, 0.0, 1.0)
                yc = np.clip(((ymin + ymax) / 2.0) / h_img, 0.0, 1.0)
                w_norm = np.clip(bw / w_img, 0.0, 1.0)
                h_norm = np.clip(bh / h_img, 0.0, 1.0)
                records.append(f"0 {xc:.6f} {yc:.6f} {w_norm:.6f} {h_norm:.6f}")

            prefix = f"seq_{seq}_{fid}"
            shutil.copy(src_img, dst_img_dir / f"{prefix}{src_img.suffix}")
            with open(dst_lbl_dir / f"{prefix}.txt", "w") as f_lbl:
                f_lbl.write("\n".join(records))

    with open(yaml_path, "w") as f_yaml:
        f_yaml.write(
            f"path: {output_dir.resolve().as_posix()}\n"
            f"train: images/train\n"
            f"val: images/val\n"
            f"test: images/test\n"
            f"names:\n"
            f"  0: apple\n"
        )
    return yaml_path


def plot_and_save_metrics(csv_path: Path, output_plot_path: Path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    epochs = df["epoch"]

    train_box = df["train/giou_loss"] if "train/giou_loss" in df.columns else df.get("train/box_loss", 0.0)
    train_cls = df.get("train/cls_loss", 0.0)
    train_l1 = df.get("train/l1_loss", 0.0)
    train_loss = train_box + train_cls + train_l1

    val_box = df["val/giou_loss"] if "val/giou_loss" in df.columns else df.get("val/box_loss", 0.0)
    val_cls = df.get("val/cls_loss", 0.0)
    val_l1 = df.get("val/l1_loss", 0.0)
    val_loss = val_box + val_cls + val_l1

    map50 = df["metrics/mAP50(B)"]
    recall = df["metrics/recall(B)"]

    plt.figure(figsize=(7, 6))

    plt.subplot(2, 1, 1)
    plt.plot(epochs, train_loss, label="Train", color="#1f77b4", linewidth=1.7)
    plt.plot(epochs, val_loss, label="Validation", color="#ff7f0e", linewidth=1.7)
    plt.ylabel("Loss", fontsize=11)
    plt.grid(True, linestyle="-", alpha=0.6)
    plt.legend(loc="upper right", frameon=True)

    plt.subplot(2, 1, 2)
    plt.plot(epochs, map50, label="mAP50", color="#d62728", linewidth=1.7)
    plt.plot(epochs, recall, label="recall", color="#9467bd", linewidth=1.7)
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Metrics", fontsize=11)
    plt.grid(True, linestyle="-", alpha=0.6)
    plt.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    output_plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_plot_path, dpi=300)
    plt.close()
    print(f"[Plot] Metrics plot saved to: {output_plot_path}")


def save_metrics_table(eval_results, output_csv_path: Path):
    metrics_data = {
        "Metric": ["mAP50", "mAP50-95", "Precision", "Recall"],
        "Value": [
            eval_results.box.map50,
            eval_results.box.map,
            eval_results.box.mp,
            eval_results.box.mr
        ],
        "Percentage": [
            f"{eval_results.box.map50 * 100:.2f}%",
            f"{eval_results.box.map * 100:.2f}%",
            f"{eval_results.box.mp * 100:.2f}%",
            f"{eval_results.box.mr * 100:.2f}%"
        ]
    }
    df_metrics = pd.DataFrame(metrics_data)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_metrics.to_csv(output_csv_path, index=False)
    print(f"[Table] Evaluation metrics table saved to: {output_csv_path}")


def run_tracking_on_test(detector_weights: Path, raw_dataset_dir: Path, output_tracking_dir: Path):
    print("\n" + "=" * 60)
    print("      STARTING TRACKING ON INDEPENDENT TEST SET (0006-0008)      ")
    print("=" * 60)

    output_tracking_dir.mkdir(parents=True, exist_ok=True)
    test_seqs = ["0006", "0007", "0008"]
    model = RTDETR(str(detector_weights))

    for seq in test_seqs:
        seq_img_dir = None
        for root, _, _ in os.walk(raw_dataset_dir):
            if os.path.basename(root) == seq and os.path.basename(os.path.dirname(root)) == "images":
                seq_img_dir = Path(root)
                break

        if not seq_img_dir or not seq_img_dir.exists():
            print(f"[Tracking Warning] Sequence {seq} not found in {raw_dataset_dir}")
            continue

        images = sorted([f for f in seq_img_dir.iterdir() if f.suffix.lower() in [".png", ".jpg", ".jpeg"]])
        mot_records = []
        out_file = output_tracking_dir / f"{seq}.txt"

        print(f"[Tracking] Processing sequence {seq} ({len(images)} frames)...")

        for frame_idx, img_path in enumerate(tqdm(images, desc=f"Seq {seq}"), start=1):
            persist_flag = False if frame_idx == 1 else True

            results = model.track(
                source=str(img_path),
                persist=persist_flag,
                tracker="bytetrack.yaml",
                conf=0.28,
                iou=0.60,
                imgsz=960,
                verbose=False
            )

            res = results[0]
            if res.boxes is not None and res.boxes.id is not None:
                boxes = res.boxes.xyxy.cpu().numpy()
                track_ids = res.boxes.id.int().cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()

                for box, tid, conf in zip(boxes, track_ids, confs):
                    x1, y1, x2, y2 = box
                    w = x2 - x1
                    h = y2 - y1
                    mot_records.append(f"{frame_idx},{tid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.4f},-1,-1,-1")

        with open(out_file, "w") as f:
            f.write("\n".join(mot_records) + "\n")
        print(f"[Tracking] Results for {seq} saved to: {out_file}")


def main():
    seed_val = 42
    set_seed(seed_val)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Hardware] Execution device: {device}")

    base_dir = Path.cwd()
    data_source_dir = base_dir / "data" / "APPLE_MOTS"
    dataset_proc_dir = base_dir / "data" / "processed_detection"
    runs_dir = base_dir / "runs" / "detector_train"
    weights_dir = base_dir / "weights"
    assets_dir = base_dir / "assets"
    tracking_out_dir = base_dir / "runs" / "test_tracking_results"

    weights_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    data_yaml = prepare_detection_dataset(data_source_dir, dataset_proc_dir)

    print("\n[Model] Initializing RT-DETR-L...")
    model = RTDETR("rtdetr-l.pt")

    print("\n" + "=" * 60)
    print("                STARTING DETECTOR TRAINING                ")
    print("=" * 60)

    train_results = model.train(
        data=str(data_yaml),
        epochs=15,
        imgsz=960,
        batch=8,
        device=0 if device == "cuda" else "cpu",
        project=str(runs_dir),
        name="rtdetr_honest_run",
        exist_ok=True,
        save=True,
        save_period=1,
        val=True,
        patience=0,
        freeze=10,
        optimizer="auto",
        cos_lr=True,
        lrf=0.01,
        warmup_epochs=2.0,
        weight_decay=0.0005,
        dropout=0.05,
        box=7.5,
        cls=0.65,
        mosaic=0.0,
        close_mosaic=0,
        mixup=0.0,
        scale=0.5,
        degrees=5.0,
        fliplr=0.5,
        plots=False,
        seed=seed_val,
        deterministic=True
    )

    csv_log = Path(train_results.save_dir) / "results.csv"
    if csv_log.exists():
        plot_and_save_metrics(csv_log, assets_dir / "detector_training_metrics.png")

    best_val_ckpt = Path(train_results.save_dir) / "weights" / "best.pt"
    export_target = weights_dir / "honest_best_detector.pt"

    if best_val_ckpt.exists():
        shutil.copy(best_val_ckpt, export_target)
        try:
            strip_optimizer(str(export_target))
        except Exception:
            pass
        print(f"\n[Model] Best validation weights (seq 0005) saved to: {export_target}")

    print("\n" + "=" * 60)
    print("      FINAL EVALUATION ON TEST SET (FOR PUBLICATION)      ")
    print("=" * 60)

    test_model = RTDETR(str(export_target))
    test_eval_results = test_model.val(
        data=str(data_yaml),
        split="test",
        imgsz=960,
        batch=8,
        conf=0.28,
        iou=0.60,
        device=0 if device == "cuda" else "cpu",
        plots=False
    )

    save_metrics_table(test_eval_results, assets_dir / "test_evaluation_metrics.csv")

    print("\n>>> DETECTION RESULTS ON TEST SET (0006-0008) FOR PUBLICATION TABLE:")
    print(f"mAP50:    {test_eval_results.box.map50 * 100:.2f}%")
    print(f"mAP50-95: {test_eval_results.box.map * 100:.2f}%")
    print(f"Precision:{test_eval_results.box.mp * 100:.2f}%")
    print(f"Recall:   {test_eval_results.box.mr * 100:.2f}%")

    run_tracking_on_test(export_target, data_source_dir, tracking_out_dir)

    print("\n" + "=" * 60)
    print("                 ALL STAGES SUCCESSFULLY COMPLETED                 ")
    print("=" * 60)


if __name__ == "__main__":
    main()