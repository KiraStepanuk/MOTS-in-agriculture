import os
import sys
import time
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import pycocotools.mask as rletools
from tqdm import tqdm

import torch
from ultralytics import RTDETR


BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from configs.tracking_config import PipelineConfig
from tracking.run_tracker import (
    AppleFaceNet,
    AppleUNet128,
    AppleFaceIDTracker,
    extract_clean_detections,
    MOTS_TOOLS_DIR,
    EVAL_SCRIPT,
)


FALLBACK_IMG_MAP = {"0010": "0006", "0011": "0007", "0012": "0008"}

def resolve_sequence_paths(seq_name: str, cfg: PipelineConfig) -> Tuple[Optional[Path], Optional[Path]]:
    splits = ["testing", "test", "train", "val"]
    found_img_dir = None
    found_gt_dir = None


    for sp in splits:
        gt_candidates = [
            cfg.data_dir / sp / "instances" / seq_name,
            cfg.base_dir / "data" / "raw_apple_mots" / "APPLE_MOTS" / sp / "instances" / seq_name,
            cfg.base_dir / "data" / "raw_apple_mots" / sp / "instances" / seq_name,
        ]
        for p in gt_candidates:
            if p.exists():
                found_gt_dir = p
                break
        if found_gt_dir:
            break

    img_names_to_try = [seq_name, FALLBACK_IMG_MAP.get(seq_name, seq_name)]
    for img_name in img_names_to_try:
        for sp in splits:
            img_candidates = [
                cfg.data_dir / sp / "images" / img_name,
                cfg.base_dir / "data" / "raw_apple_mots" / "APPLE_MOTS" / sp / "images" / img_name,
                cfg.base_dir / "data" / "raw_apple_mots" / sp / "images" / img_name,
            ]
            for p in img_candidates:
                if p.exists():
                    found_img_dir = p
                    break
            if found_img_dir:
                break
        if found_img_dir:
            break

    return found_img_dir, found_gt_dir


def run_ignore_regions_eval(cfg: Optional[PipelineConfig] = None):
    if cfg is None:
        cfg = PipelineConfig()

    seqs_ignore = ["0010", "0011", "0012"]
    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 85)
    print(f"[*] Hardware check: PyTorch {torch.__version__}")
    print(f"[*] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[*] Execution device: {torch.cuda.get_device_name(0)}")
        cfg.device = torch.device("cuda:0")
    else:
        print("[!] CRITICAL WARNING: Running on CPU! Execution will be slow.")
        cfg.device = torch.device("cpu")
    print("=" * 85)


    seq_paths = {}
    for s in seqs_ignore:
        img_p, gt_p = resolve_sequence_paths(s, cfg)
        assert gt_p is not None, f"GT ignore-region annotations for {s} not found!"
        assert img_p is not None, f"Images directory for {s} (or fallback {FALLBACK_IMG_MAP.get(s)}) not found!"
        seq_paths[s] = {"images": img_p, "instances": gt_p}

    gt_instances_root = seq_paths["0010"]["instances"].parent
    print("[*] RESOLVED DIRECTORIES (IGNORE REGIONS):")
    for s, p in seq_paths.items():
        print(f"  • {s} | Images: {p['images'].name} | GT: {p['instances'].name}")
    print(f"[*] Root GT instances directory: {gt_instances_root}")
    print("=" * 85)


    seqmap_ignore_path = out_dir / "test_ignore.seqmap"
    seqmap_lines = []
    for s in seqs_ignore:
        frames = sorted([f for f in os.listdir(seq_paths[s]["images"]) if f.endswith((".png", ".jpg"))])
        assert len(frames) > 0, f"Image folder {seq_paths[s]['images']} is empty!"
        seqmap_lines.append(f"{s} empty 0 {len(frames) - 1}\n")
    
    with open(seqmap_ignore_path, "w") as f_sm:
        f_sm.writelines(seqmap_lines)
    print(f"[+] Generated {seqmap_ignore_path.name}")

    # Load Models
    print(f"[*] Loading RT-DETR weights from: {cfg.detector_weights}")
    detector_model = RTDETR(str(cfg.detector_weights))
    if hasattr(detector_model, "fuse"):
        try:
            detector_model.fuse()
        except Exception:
            pass
    if cfg.device.type == "cuda":
        detector_model.to("cuda")

    reid_net = None
    if cfg.reid_weights.exists():
        print(f"[*] Loading AppleFaceNet weights from: {cfg.reid_weights}")
        reid_net = AppleFaceNet(emb_dim=128).to(cfg.device)
        reid_checkpoint = torch.load(str(cfg.reid_weights), map_location=cfg.device, weights_only=False)
        reid_net.load_state_dict(reid_checkpoint)
        reid_net.eval()

    seg_net = None
    if cfg.seg_weights.exists():
        print(f"[*] Loading AppleUNet128 weights from: {cfg.seg_weights}")
        seg_net = AppleUNet128().to(cfg.device)
        seg_checkpoint = torch.load(str(cfg.seg_weights), map_location=cfg.device, weights_only=False)
        seg_net.load_state_dict(seg_checkpoint)
        seg_net.eval()


    total_frames = 0
    start_total = time.perf_counter()

    for s in seqs_ignore:
        img_dir = seq_paths[s]["images"]
        frames = sorted([f for f in os.listdir(img_dir) if f.endswith((".png", ".jpg"))])
        
        tracker = AppleFaceIDTracker(cfg=cfg)
        buffered = []
        orig_h, orig_w = None, None
        seq_start = time.perf_counter()

        pbar = tqdm(frames, desc=f"Track {s} (ignore regions)", unit="frame")
        for f_idx, f_name in enumerate(pbar):
            img_p = str(img_dir / f_name)
            curr_img = cv2.imread(img_p)
            if orig_h is None:
                orig_h, orig_w = curr_img.shape[:2]

            results = detector_model.predict(
                source=curr_img,
                conf=cfg.detector.conf_thresh,
                iou=cfg.detector.iou_thresh,
                imgsz=cfg.detector.imgsz,
                verbose=False,
                device=0 if cfg.device.type == "cuda" else "cpu",
            )[0]

            detections = extract_clean_detections(
                results, (orig_h, orig_w), curr_img, cfg, seg_net=seg_net, reid_net=reid_net
            )
            active_tracks = tracker.step(curr_img, detections)
            for t in active_tracks:
                buffered.append((f_idx, t.id, t.crop_box, t.mask_crop))

            if cfg.device.type == "cuda" and f_idx % 20 == 0:
                torch.cuda.empty_cache()

        seq_elapsed = time.perf_counter() - seq_start
        total_frames += len(frames)
        fps = len(frames) / seq_elapsed if seq_elapsed > 0 else 0
        print(f"[{s}] Processed {len(frames)} frames | Speed: {fps:.2f} FPS")

        # Save predictions in RLE MOTS format as {s}.txt (e.g., 0010.txt)
        out_txt = out_dir / f"{s}.txt"
        canvas = np.zeros((orig_h, orig_w), dtype=np.uint8, order="F")
        with open(out_txt, "w") as f_out:
            for f_idx, t_id, (y1, y2, x1, x2), crop in buffered:
                canvas[y1:y2, x1:x2] = crop
                rle = rletools.encode(canvas)
                rle_str = rle["counts"].decode("utf-8")
                f_out.write(f"{f_idx} {t_id} 1 {orig_h} {orig_w} {rle_str}\n")
                canvas[y1:y2, x1:x2] = 0

    total_fps = total_frames / (time.perf_counter() - start_total)
    print(f"\n[+] Inference finished. Average Speed: {total_fps:.2f} FPS")


    print("\n" + "=" * 85)
    print("[*] RUNNING OFFICIAL MOTS EVALUATION (0010 - 0012 WITH IGNORE REGIONS)...")
    print("=" * 85)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(MOTS_TOOLS_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [
        sys.executable,
        str(EVAL_SCRIPT),
        str(out_dir),
        str(gt_instances_root),
        str(seqmap_ignore_path),
    ]

    process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    print(process.stdout)


    metrics = {
        "Split": f"Test {seqs_ignore} (Ignore Regions)",
        "Speed (FPS) ↑": round(total_fps, 2),
        "sMOTSA (%) ↑": 0.0,
        "MOTSA (%) ↑": 0.0,
        "MOTSP (%) ↑": 0.0,
        "Recall (%) ↑": 0.0,
        "Prec (%) ↑": 0.0,
        "F1 (%) ↑": 0.0,
        "ID Switches ↓": 0,
    }

    is_target = False
    for line in process.stdout.splitlines():
        if any(k in line.lower() for k in ["cars", "apples"]):
            is_target = True
            continue
        if "pedestrians" in line.lower():
            is_target = False
        if is_target and line.strip().startswith("all"):
            nums = re.findall(r"[-+]?\d*\.\d+|\d+", line)
            if len(nums) >= 17:
                metrics["sMOTSA (%) ↑"] = float(nums[0])
                metrics["MOTSA (%) ↑"] = float(nums[1])
                metrics["MOTSP (%) ↑"] = float(nums[2])
                metrics["Recall (%) ↑"] = float(nums[6])
                metrics["Prec (%) ↑"] = float(nums[7])
                metrics["F1 (%) ↑"] = float(nums[8])
                metrics["ID Switches ↓"] = int(nums[16])
            break

    return metrics


if __name__ == "__main__":
    cfg = PipelineConfig()
    test_metrics = run_ignore_regions_eval(cfg=cfg)

    df = pd.DataFrame([test_metrics])
    print("\n" + "=" * 80)
    print("FINAL BENCHMARK RESULTS (TEST SPLIT WITH IGNORE REGIONS: 0010 - 0012)")
    print("=" * 80)
    print(df.to_string(index=False))