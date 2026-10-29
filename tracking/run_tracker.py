import os
import sys
import time
import re
import subprocess
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import pycocotools.mask as rletools
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision.models import resnet18
from ultralytics import RTDETR


BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from configs.tracking_config import PipelineConfig


MOTS_TOOLS_DIR = BASE_DIR / "tools" / "mots_tools"
EVAL_SCRIPT = MOTS_TOOLS_DIR / "mots_eval" / "eval.py"

if not EVAL_SCRIPT.exists():
    print("[Setup] Cloning official mots_tools benchmark...")
    MOTS_TOOLS_DIR.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "https://github.com/VisualComputingInstitute/mots_tools", str(MOTS_TOOLS_DIR)],
        check=True,
    )


class AppleFaceNet(nn.Module):
    def __init__(self, emb_dim: int = 128):
        super().__init__()
        base = resnet18(weights=None)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.projector = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, emb_dim),
            nn.BatchNorm1d(emb_dim, affine=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x).flatten(1)
        emb = self.projector(feat)
        return F.normalize(emb, p=2, dim=1)

class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        d1 = self.conv1(torch.cat([self.up1(x5), x4], dim=1))
        d2 = self.conv2(torch.cat([self.up2(d1), x3], dim=1))
        d3 = self.conv3(torch.cat([self.up3(d2), x2], dim=1))
        d4 = self.conv4(torch.cat([self.up4(d3), x1], dim=1))
        return self.outc(d4)


reid_transforms = T.Compose([
    T.ToPILImage(),
    T.Resize((128, 128)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

seg_transforms = T.Compose([
    T.ToPILImage(),
    T.Resize((128, 128), interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_cmc_transform(
    prev_img: Optional[np.ndarray],
    curr_img: Optional[np.ndarray],
    cfg: PipelineConfig,
) -> np.ndarray:
    identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    if prev_img is None or curr_img is None:
        return identity
    h, w = prev_img.shape[:2]
    scale = cfg.cmc.scale_down_size / max(h, w)
    sw, sh = int(w * scale), int(h * scale)
    p_gray = cv2.cvtColor(cv2.resize(prev_img, (sw, sh)), cv2.COLOR_BGR2GRAY)
    c_gray = cv2.cvtColor(cv2.resize(curr_img, (sw, sh)), cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(
        p_gray,
        maxCorners=cfg.cmc.max_corners,
        qualityLevel=cfg.cmc.quality_level,
        minDistance=cfg.cmc.min_distance,
        blockSize=cfg.cmc.block_size,
    )
    if p0 is None or len(p0) < cfg.cmc.min_points_ransac:
        return identity
    p1, st, _ = cv2.calcOpticalFlowPyrLK(p_gray, c_gray, p0, None)
    if st is None:
        return identity
    good_p0 = p0[st.ravel() == 1]
    good_p1 = p1[st.ravel() == 1]
    if len(good_p0) < cfg.cmc.min_points_ransac:
        return identity
    M, _ = cv2.estimateAffinePartial2D(
        good_p0, good_p1, method=cv2.RANSAC, ransacReprojThreshold=3.0
    )
    if M is None:
        return identity
    M[0, 2] /= scale
    M[1, 2] /= scale
    return M.astype(np.float32)

def get_context_crop(img: np.ndarray, box: np.ndarray, scale: float = 2.2) -> np.ndarray:
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

def compute_nwd(box1: np.ndarray, box2: np.ndarray, constant: float = 12.0) -> float:
    cx1, cy1 = (box1[0] + box1[2]) * 0.5, (box1[1] + box1[3]) * 0.5
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    cx2, cy2 = (box2[0] + box2[2]) * 0.5, (box2[1] + box2[3]) * 0.5
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    center_dist2 = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2
    scale_dist2 = 0.10 * ((w1 - w2) ** 2 + (h1 - h2) ** 2)
    w_dist = np.sqrt(center_dist2 + scale_dist2)
    return float(np.exp(-w_dist / constant))

def compute_star_graphs(
    boxes: List[np.ndarray], radius_scale: float = 3.5, max_neighbors: int = 8
) -> Tuple[List[np.ndarray], np.ndarray]:
    N = len(boxes)
    if N == 0:
        return [], np.zeros((0,), dtype=np.float32)
    boxes_arr = np.array(boxes, dtype=np.float32)
    centers = (boxes_arr[:, :2] + boxes_arr[:, 2:]) * 0.5
    diams = np.sqrt(
        (boxes_arr[:, 2] - boxes_arr[:, 0]) * (boxes_arr[:, 3] - boxes_arr[:, 1])
    ) + 1e-6
    diff = centers[None, :, :] - centers[:, None, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    star_graphs = []
    confidences = np.zeros(N, dtype=np.float32)
    for i in range(N):
        r_limit = radius_scale * diams[i]
        neighbor_mask = (dist[i] <= r_limit) & (np.arange(N) != i)
        neighbor_indices = np.where(neighbor_mask)[0]
        if len(neighbor_indices) == 0:
            star_graphs.append(np.zeros((0, 2), dtype=np.float32))
            confidences[i] = 0.0
            continue
        if len(neighbor_indices) > max_neighbors:
            sorted_idx = np.argsort(dist[i, neighbor_indices])[:max_neighbors]
            neighbor_indices = neighbor_indices[sorted_idx]
        rel_vectors = diff[i, neighbor_indices] / diams[i]
        star_graphs.append(rel_vectors.astype(np.float32))
        confidences[i] = float(np.clip(len(neighbor_indices) / 3.0, 0.0, 1.0))
    return star_graphs, confidences

def compute_pairwise_star_chamfer(
    tracks_stars: List[np.ndarray], dets_stars: List[np.ndarray], cutoff: float = 1.5
) -> np.ndarray:
    n_t = len(tracks_stars)
    n_d = len(dets_stars)
    cost_mat = np.zeros((n_t, n_d), dtype=np.float32)
    for t_idx in range(n_t):
        A = tracks_stars[t_idx]
        len_A = len(A)
        for d_idx in range(n_d):
            B = dets_stars[d_idx]
            len_B = len(B)
            if len_A == 0 and len_B == 0:
                cost_mat[t_idx, d_idx] = 0.0
                continue
            if len_A == 0 or len_B == 0:
                cost_mat[t_idx, d_idx] = 1.0
                continue
            pair_diff = A[:, None, :] - B[None, :, :]
            pair_dist = np.sqrt(np.sum(pair_diff ** 2, axis=-1))
            dist_A_to_B = np.minimum(np.min(pair_dist, axis=1), cutoff)
            dist_B_to_A = np.minimum(np.min(pair_dist, axis=0), cutoff)
            chamfer = 0.5 * (np.mean(dist_A_to_B) + np.mean(dist_B_to_A))
            cost_mat[t_idx, d_idx] = float(chamfer / cutoff)
    return cost_mat

def extract_clean_detections(
    results, orig_shape: Tuple[int, int], curr_img: np.ndarray, cfg: PipelineConfig,
    seg_net: Optional[nn.Module] = None, reid_net: Optional[nn.Module] = None
) -> List[dict]:
    if results.boxes is None or len(results.boxes) == 0:
        return []
    orig_h, orig_w = orig_shape
    confs_np = results.boxes.conf.cpu().numpy().astype(np.float32)
    boxes = results.boxes.xyxy.cpu().numpy().astype(np.float32)
    centers = results.boxes.xywh[:, :2].cpu().numpy().astype(np.float32)

    valid_candidates = []
    for idx in range(len(boxes)):
        bx1, by1, bx2, by2 = boxes[idx]
        bw, bh = bx2 - bx1, by2 - by1
        if bw < 5 or bh < 5:
            continue
        aspect = min(bw, bh) / (max(bw, bh) + 1e-6)
        if aspect < cfg.scale_prior.min_aspect_ratio:
            continue
        valid_candidates.append({
            "idx": idx,
            "box": np.array([bx1, by1, bx2, by2], dtype=np.float32),
            "score": confs_np[idx],
            "center": centers[idx],
            "bw": bw,
            "bh": bh,
        })
    if not valid_candidates:
        return []
    valid_candidates.sort(key=lambda x: x["score"], reverse=True)

    survived_dets = []
    for cand in valid_candidates:
        is_dup = False
        for surv in survived_dets:
            c_scale = min(cand["bw"] + cand["bh"], surv["bw"] + surv["bh"]) * 0.5
            nwd_sim = compute_nwd(cand["box"], surv["box"], constant=max(c_scale, 5.0))
            if nwd_sim > cfg.scale_prior.nwd_nms_thresh:
                is_dup = True
                break
        if not is_dup:
            survived_dets.append(cand)

    unet_crops, crop_coords = [], []
    for item in survived_dets:
        bx1, by1, bx2, by2 = item["box"]
        bw, bh = item["bw"], item["bh"]
        pad_x, pad_y = int(bw * 0.15), int(bh * 0.15)
        cx1 = max(0, int(bx1) - pad_x)
        cy1 = max(0, int(by1) - pad_y)
        cx2 = min(orig_w, int(bx2) + pad_x)
        cy2 = min(orig_h, int(by2) + pad_y)
        patch = curr_img[cy1:cy2, cx1:cx2]
        unet_crops.append(seg_transforms(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)))
        crop_coords.append((cx1, cy1, cx2, cy2))

    if seg_net is not None and len(unet_crops) > 0:
        batch_t = torch.stack(unet_crops).to(cfg.device)
        with torch.no_grad():
            logits = seg_net(batch_t)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
    else:
        probs = np.ones((len(survived_dets), 128, 128), dtype=np.float32)

    full_assign = np.zeros((orig_h, orig_w), dtype=np.int32)
    assignment_weight_canvas = np.zeros((orig_h, orig_w), dtype=np.float32)
    for det_rank, item in enumerate(survived_dets):
        det_id = det_rank + 1
        item["det_id"] = det_id
        cx1, cy1, cx2, cy2 = crop_coords[det_rank]
        target_w, target_h = cx2 - cx1, cy2 - cy1
        if target_w <= 0 or target_h <= 0:
            continue
        prob_map = cv2.resize(probs[det_rank], (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        fruit_mask = prob_map > 0.46
        x_shift = np.arange(target_w, dtype=np.float32) - (item["center"][0] - cx1)
        y_shift = np.arange(target_h, dtype=np.float32) - (item["center"][1] - cy1)
        sigma_x = max(item["bw"] * 0.45, 3.0)
        sigma_y = max(item["bh"] * 0.45, 3.0)
        spatial_prior = np.exp(-0.5 * ((x_shift[None, :] / sigma_x) ** 2 + (y_shift[:, None] / sigma_y) ** 2))
        weighted_map = prob_map * (0.60 + 0.40 * spatial_prior)
        curr_canvas_slice = assignment_weight_canvas[cy1:cy2, cx1:cx2]
        win_mask = fruit_mask & (weighted_map > curr_canvas_slice)
        full_assign[cy1:cy2, cx1:cx2][win_mask] = det_id
        assignment_weight_canvas[cy1:cy2, cx1:cx2][win_mask] = weighted_map[win_mask]

    boxes_out, crops_ctx, raw_dets = [], [], []
    for det_rank, item in enumerate(survived_dets):
        det_id = item["det_id"]
        cx1, cy1, cx2, cy2 = crop_coords[det_rank]
        full_patch_mask = (full_assign[cy1:cy2, cx1:cx2] == det_id).astype(np.uint8)
        y_idxs, x_idxs = np.where(full_patch_mask > 0)
        if len(y_idxs) < 10:
            continue
        real_y1 = cy1 + int(np.min(y_idxs))
        real_y2 = cy1 + int(np.max(y_idxs)) + 1
        real_x1 = cx1 + int(np.min(x_idxs))
        real_x2 = cx1 + int(np.max(x_idxs)) + 1
        mask_crop = (full_assign[real_y1:real_y2, real_x1:real_x2] == det_id).astype(np.uint8)
        b = item["box"]
        crops_ctx.append(get_context_crop(curr_img, b, scale=cfg.topology.context_crop_scale))
        boxes_out.append(b)
        raw_dets.append({
            "id_in_frame": det_id,
            "crop_box": (real_y1, real_y2, real_x1, real_x2),
            "mask_crop": mask_crop,
            "box": b,
            "centroid": item["center"],
            "score": item["score"],
        })

    if len(raw_dets) == 0:
        return []

    if reid_net is not None and len(crops_ctx) > 0:
        batch = torch.stack([reid_transforms(c) for c in crops_ctx]).to(cfg.device)
        with torch.no_grad():
            vis_embs = reid_net(batch).cpu().numpy()
    else:
        vis_embs = np.zeros((len(raw_dets), 128), dtype=np.float32)

    star_graphs, star_confs = compute_star_graphs(
        boxes_out,
        radius_scale=cfg.topology.radius_scale,
        max_neighbors=cfg.topology.max_neighbors,
    )
    for i, d in enumerate(raw_dets):
        d["vis_emb"] = vis_embs[i]
        d["topo_emb"] = star_graphs[i]
        d["topo_conf"] = float(star_confs[i])

    return raw_dets

# ----------------- Tracker Components -----------------
class Track:
    def __init__(
        self, uid: int, mask_crop: np.ndarray, crop_box: tuple, box: np.ndarray,
        centroid: np.ndarray, score: float, vis_emb: np.ndarray, topo_emb: np.ndarray,
        cfg: PipelineConfig, topo_conf: float = 1.0,
    ):
        self.cfg = cfg
        self.uid = uid
        self.id = None
        self.mask_crop = mask_crop
        self.crop_box = crop_box
        self.box = np.array(box, dtype=np.float32)
        self.centroid = np.array(centroid, dtype=np.float32)
        self.score = float(score)
        self.vis_emb = vis_emb
        self.topo_emb = topo_emb
        self.topo_conf = float(topo_conf)
        self.state = "Tentative"
        self.hits = 1
        self.time_since_update = 0
        self.min_hits = cfg.assoc.min_hits_to_activate

        bw = max(float(self.box[2] - self.box[0]), 1.0)
        bh = max(float(self.box[3] - self.box[1]), 1.0)
        scale = np.sqrt(bw * bh)
        std_p = getattr(cfg.kalman, "std_weight_pos", 0.05) * scale
        std_v = getattr(cfg.kalman, "std_weight_vel", 0.00625) * scale
        min_clip = getattr(cfg.kalman, "min_score_clip", 0.10)

        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32
        )
        self.kf.transitionMatrix = np.array(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        self.kf.processNoiseCov = np.diag(
            [std_p**2, std_p**2, std_v**2, std_v**2]
        ).astype(np.float32)
        r_pos = (std_p**2) / max(self.score, min_clip)
        self.kf.measurementNoiseCov = np.diag([r_pos, r_pos]).astype(np.float32)
        self.kf.errorCovPost = np.diag([
            (2.0 * std_p) ** 2,
            (2.0 * std_p) ** 2,
            (10.0 * std_v) ** 2,
            (10.0 * std_v) ** 2,
        ]).astype(np.float32)
        self.kf.statePost = np.array(
            [[self.centroid[0]], [self.centroid[1]], [0.0], [0.0]],
            dtype=np.float32,
        )

    def update(
        self, mask_crop: np.ndarray, crop_box: tuple, box: np.ndarray,
        centroid: np.ndarray, score: float, vis_emb: np.ndarray, topo_emb: np.ndarray,
        topo_conf: float = 1.0,
    ):
        bw = max(float(box[2] - box[0]), 1.0)
        bh = max(float(box[3] - box[1]), 1.0)
        scale = np.sqrt(bw * bh)
        std_p = getattr(self.cfg.kalman, "std_weight_pos", 0.05) * scale
        std_v = getattr(self.cfg.kalman, "std_weight_vel", 0.00625) * scale
        min_clip = getattr(self.cfg.kalman, "min_score_clip", 0.10)

        self.kf.processNoiseCov = np.diag(
            [std_p**2, std_p**2, std_v**2, std_v**2]
        ).astype(np.float32)
        r_pos = (std_p**2) / max(float(score), min_clip)
        self.kf.measurementNoiseCov = np.diag([r_pos, r_pos]).astype(np.float32)
        meas = np.array([[centroid[0]], [centroid[1]]], dtype=np.float32)
        self.kf.correct(meas)
        kf_cx = float(self.kf.statePost[0, 0])
        kf_cy = float(self.kf.statePost[1, 0])
        self.centroid = np.array([kf_cx, kf_cy], dtype=np.float32)
        self.box = np.array([
            kf_cx - bw * 0.5,
            kf_cy - bh * 0.5,
            kf_cx + bw * 0.5,
            kf_cy + bh * 0.5,
        ], dtype=np.float32)
        self.mask_crop = mask_crop
        self.crop_box = crop_box
        self.score = float(score)
        self.time_since_update = 0
        self.hits += 1

        alpha = self.cfg.assoc.alpha_embedding_update
        self.vis_emb = alpha * self.vis_emb + (1.0 - alpha) * vis_emb
        self.vis_emb /= np.linalg.norm(self.vis_emb) + 1e-6
        self.topo_emb = topo_emb
        self.topo_conf = alpha * self.topo_conf + (1.0 - alpha) * float(topo_conf)

    def apply_cmc(self, A_t: np.ndarray):
        pos = np.array([self.kf.statePost[0, 0], self.kf.statePost[1, 0], 1.0], dtype=np.float32)
        new_pos = A_t.dot(pos)
        R = A_t[:2, :2].astype(np.float32)
        vel = np.array([self.kf.statePost[2, 0], self.kf.statePost[3, 0]], dtype=np.float32)
        new_vel = R.dot(vel)
        self.kf.statePost[0, 0], self.kf.statePost[1, 0] = new_pos[0], new_pos[1]
        self.kf.statePost[2, 0], self.kf.statePost[3, 0] = new_vel[0], new_vel[1]
        J = np.zeros((4, 4), dtype=np.float32)
        J[:2, :2] = R
        J[2:, 2:] = R
        self.kf.errorCovPost = J.dot(self.kf.errorCovPost).dot(J.T)
        zoom_factor = float(np.sqrt(max(np.abs(np.linalg.det(R)), 1e-4)))
        bw = (self.box[2] - self.box[0]) * zoom_factor
        bh = (self.box[3] - self.box[1]) * zoom_factor
        self.centroid = new_pos[:2].astype(np.float32)
        self.box = np.array([
            self.centroid[0] - bw * 0.5,
            self.centroid[1] - bh * 0.5,
            self.centroid[0] + bw * 0.5,
            self.centroid[1] + bh * 0.5,
        ], dtype=np.float32)

    def predict(self):
        pred = self.kf.predict()
        bw = self.box[2] - self.box[0]
        bh = self.box[3] - self.box[1]
        self.centroid = np.array([pred[0, 0], pred[1, 0]], dtype=np.float32)
        self.box = np.array([
            self.centroid[0] - bw * 0.5,
            self.centroid[1] - bh * 0.5,
            self.centroid[0] + bw * 0.5,
            self.centroid[1] + bh * 0.5,
        ], dtype=np.float32)
        self.time_since_update += 1

class AppleFaceIDTracker:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.tracks: List[Track] = []
        self.next_mots_id = 1
        self.next_uid = 1
        self.prev_img = None

    def _hybrid_match(
        self, tracks: List[Track], dets: List[dict], use_reid: bool, max_cost: float
    ):
        if len(tracks) == 0 or len(dets) == 0:
            return tracks, set()
        t_boxes = np.array([t.box for t in tracks], dtype=np.float32)
        d_boxes = np.array([d["box"] for d in dets], dtype=np.float32)

        tl = np.maximum(t_boxes[:, None, :2], d_boxes[None, :, :2])
        br = np.minimum(t_boxes[:, None, 2:], d_boxes[None, :, 2:])
        wh = np.maximum(0.0, br - tl)
        inter = wh[:, :, 0] * wh[:, :, 1]
        a_a = (t_boxes[:, 2] - t_boxes[:, 0]) * (t_boxes[:, 3] - t_boxes[:, 1])
        a_b = (d_boxes[:, 2] - d_boxes[:, 0]) * (d_boxes[:, 3] - d_boxes[:, 1])
        iou_mat = inter / (a_a[:, None] + a_b[None, :] - inter + 1e-6)
        t_c = np.array([t.centroid for t in tracks], dtype=np.float32)
        d_c = np.array([d["centroid"] for d in dets], dtype=np.float32)
        dist_mat = np.linalg.norm(t_c[:, None, :] - d_c[None, :, :], axis=2)
        scale_geo = np.sqrt(np.sqrt(a_a)[:, None] * np.sqrt(a_b)[None, :]) + 1e-6
        norm_dist_cost = np.clip(dist_mat / (1.5 * scale_geo), 0.0, 1.0)
        disp_ratio = dist_mat / scale_geo

        if use_reid:
            t_vis = np.array([t.vis_emb for t in tracks], dtype=np.float32)
            d_vis = np.array([d["vis_emb"] for d in dets], dtype=np.float32)
            cost_vis = np.clip((1.0 - np.dot(t_vis, d_vis.T)) * 0.5, 0.0, 1.0)
            t_stars = [t.topo_emb for t in tracks]
            d_stars = [d["topo_emb"] for d in dets]
            cost_topo = compute_pairwise_star_chamfer(
                t_stars, d_stars, cutoff=self.cfg.topology.chamfer_cutoff
            )
            t_conf = np.array([t.topo_conf for t in tracks], dtype=np.float32)[:, None]
            d_conf = np.array([d.get("topo_conf", 1.0) for d in dets], dtype=np.float32)[None, :]
            topo_conf = np.minimum(t_conf, d_conf)
            w_topo = self.cfg.assoc.reid_topo_weight * np.clip(topo_conf, 0.0, 1.0)
            cost_appearance = (1.0 - w_topo) * cost_vis + w_topo * cost_topo
            gate = self.cfg.assoc.iou_reid_gate
            alpha_reid = self.cfg.assoc.w_reid_max * np.clip((gate - iou_mat) / gate, 0.0, 1.0)
            w_spatial = 1.0 - alpha_reid
            w_iou = w_spatial * 0.75
            w_dist = w_spatial * 0.25
            cost_mat = (
                w_iou * (1.0 - iou_mat)
                + alpha_reid * cost_appearance
                + w_dist * norm_dist_cost
            )
        else:
            cost_mat = (1.0 - iou_mat) * 0.70 + norm_dist_cost * 0.30

        tau_high = getattr(self.cfg.assoc, "high_iou_lock_thresh", 0.45)
        iou_weight = getattr(self.cfg.assoc, "high_iou_cost_weight", 0.35)
        dominant_mask = iou_mat >= tau_high
        cost_mat[dominant_mask] = (1.0 - iou_mat[dominant_mask]) * iou_weight

        INF_COST = 1e5
        gamma_disp = (
            getattr(self.cfg.assoc, "gate_disp_round1", 1.45)
            if use_reid
            else getattr(self.cfg.assoc, "gate_disp_round2", 1.80)
        )
        delta_margin = getattr(self.cfg.assoc, "exclusivity_margin", 0.12)
        gating_mask = disp_ratio > gamma_disp
        gating_mask |= (iou_mat == 0.0) & (norm_dist_cost > 0.80)
        max_iou_per_det = np.max(iou_mat, axis=0)
        best_track_per_det = np.argmax(iou_mat, axis=0)
        for d_idx in range(len(dets)):
            if max_iou_per_det[d_idx] >= tau_high:
                owner_t = best_track_per_det[d_idx]
                for t_idx in range(len(tracks)):
                    if t_idx != owner_t and (max_iou_per_det[d_idx] - iou_mat[t_idx, d_idx]) > delta_margin:
                        gating_mask[t_idx, d_idx] = True
        cost_mat[gating_mask] = INF_COST

        row_ind, col_ind = linear_sum_assignment(cost_mat)
        matched_t, matched_d = set(), set()
        for r, c in zip(row_ind, col_ind):
            if cost_mat[r, c] < max_cost:
                tracks[r].update(
                    dets[c]["mask_crop"],
                    dets[c]["crop_box"],
                    dets[c]["box"],
                    dets[c]["centroid"],
                    dets[c]["score"],
                    dets[c]["vis_emb"],
                    dets[c]["topo_emb"],
                    dets[c].get("topo_conf", 1.0),
                )
                matched_t.add(r)
                matched_d.add(dets[c]["uid"])
        unmatched_tracks = [t for i, t in enumerate(tracks) if i not in matched_t]
        return unmatched_tracks, matched_d

    def step(self, curr_img: np.ndarray, detections: List[dict]):
        A_t = get_cmc_transform(self.prev_img, curr_img, self.cfg)
        self.prev_img = curr_img.copy()
        for t in self.tracks:
            t.apply_cmc(A_t)
            t.predict()

        high_th = self.cfg.assoc.high_score_thresh
        dets_high = [d for d in detections if d["score"] >= high_th]
        dets_low = [d for d in detections if d["score"] < high_th]
        for idx, d in enumerate(dets_high + dets_low):
            d["uid"] = idx

        fresh_tracks = [t for t in self.tracks if t.time_since_update <= 3]
        lost_tracks = [t for t in self.tracks if t.state == "Active" and t.time_since_update > 3]

        unm_fresh, m_uids1 = self._hybrid_match(
            fresh_tracks, dets_high, use_reid=True, max_cost=self.cfg.assoc.match_cost_round1
        )
        matched_uids = set(m_uids1)

        active_unm = [t for t in unm_fresh if t.state == "Active" or t.time_since_update <= 1]
        rem_low = [d for d in dets_low if d["uid"] not in matched_uids]
        _, m_uids2 = self._hybrid_match(
            active_unm, rem_low, use_reid=False, max_cost=self.cfg.assoc.match_cost_round2
        )
        matched_uids.update(m_uids2)

        rem_high = [d for d in dets_high if d["uid"] not in matched_uids]
        if len(lost_tracks) > 0 and len(rem_high) > 0:
            t_vis = np.array([t.vis_emb for t in lost_tracks])
            d_vis = np.array([d["vis_emb"] for d in rem_high])
            cost_vis = (1.0 - np.dot(t_vis, d_vis.T)) * 0.5
            t_stars = [t.topo_emb for t in lost_tracks]
            d_stars = [d["topo_emb"] for d in rem_high]
            cost_topo = compute_pairwise_star_chamfer(
                t_stars, d_stars, cutoff=self.cfg.topology.chamfer_cutoff
            )
            t_conf = np.array([t.topo_conf for t in lost_tracks], dtype=np.float32)[:, None]
            d_conf = np.array([d.get("topo_conf", 1.0) for d in rem_high], dtype=np.float32)[None, :]
            topo_confidence = np.minimum(t_conf, d_conf)
            w_topo_dyn = self.cfg.assoc.reid_topo_weight * np.clip(topo_confidence, 0.0, 1.0)
            w_vis_dyn = 1.0 - w_topo_dyn
            cost_reid = w_vis_dyn * cost_vis + w_topo_dyn * cost_topo
            t_pos = np.array([t.centroid for t in lost_tracks])
            d_pos = np.array([d["centroid"] for d in rem_high])
            spat_dist = np.linalg.norm(t_pos[:, None, :] - d_pos[None, :, :], axis=2)
            t_boxes_lost = np.array([t.box for t in lost_tracks], dtype=np.float32)
            t_diams = np.sqrt(
                (t_boxes_lost[:, 2] - t_boxes_lost[:, 0]) * (t_boxes_lost[:, 3] - t_boxes_lost[:, 1])
            )[:, None]
            delta_t = np.array([t.time_since_update for t in lost_tracks], dtype=np.float32)[:, None]
            drift_factor = np.clip(
                1.0 + self.cfg.assoc.drift_temporal_factor * delta_t,
                1.0,
                self.cfg.assoc.max_drift_multiplier,
            )
            max_allowed = t_diams * drift_factor
            cost_reid[spat_dist > max_allowed] = 1e5
            r_idx, c_idx = linear_sum_assignment(cost_reid)
            for r, c in zip(r_idx, c_idx):
                if cost_reid[r, c] < self.cfg.assoc.match_cost_lost:
                    lost_tracks[r].update(
                        rem_high[c]["mask_crop"],
                        rem_high[c]["crop_box"],
                        rem_high[c]["box"],
                        rem_high[c]["centroid"],
                        rem_high[c]["score"],
                        rem_high[c]["vis_emb"],
                        rem_high[c]["topo_emb"],
                        rem_high[c].get("topo_conf", 1.0),
                    )
                    matched_uids.add(rem_high[c]["uid"])

        for t in self.tracks:
            if t.state == "Tentative" and t.hits >= t.min_hits:
                t.state = "Active"
                if t.id is None:
                    t.id = self.next_mots_id
                    self.next_mots_id += 1

        for d in dets_high:
            if d["uid"] not in matched_uids:
                new_t = Track(
                    self.next_uid,
                    d["mask_crop"],
                    d["crop_box"],
                    d["box"],
                    d["centroid"],
                    d["score"],
                    d["vis_emb"],
                    d["topo_emb"],
                    self.cfg,
                    d.get("topo_conf", 1.0),
                )
                self.tracks.append(new_t)
                self.next_uid += 1

        max_age_tentative = getattr(self.cfg.assoc, "max_age_tentative", 2)
        valid_tracks = []
        for t in self.tracks:
            if t.state == "Tentative":
                if t.time_since_update <= max_age_tentative:
                    valid_tracks.append(t)
            elif t.state == "Active":
                if t.time_since_update <= self.cfg.assoc.max_age_active:
                    valid_tracks.append(t)
        self.tracks = valid_tracks

        return [t for t in self.tracks if (t.state == "Active" and t.time_since_update == 0)]

# ----------------- Evaluation Pipeline -----------------
def generate_seqmap(seq_list: List[str], img_root: Path, out_path: Path):
    lines = []
    for seq in seq_list:
        seq_folder = img_root / seq
        frames = sorted([f for f in os.listdir(seq_folder) if f.endswith((".png", ".jpg"))])
        lines.append(f"{seq} empty 0 {len(frames) - 1}\n")
    with open(out_path, "w") as f:
        f.writelines(lines)

def run_evaluation(
    seq_list: List[str], split_name: str = "testing", cfg: Optional[PipelineConfig] = None
) -> dict:
    if cfg is None:
        cfg = PipelineConfig()

    print("=" * 80)
    print(f"[*] Hardware check: PyTorch {torch.__version__}")
    print(f"[*] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[*] Execution device: {torch.cuda.get_device_name(0)}")
        cfg.device = torch.device("cuda:0")
    else:
        print("[!] CRITICAL WARNING: Running on CPU! This will be extremely slow.")
        cfg.device = torch.device("cpu")
    print("=" * 80)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    images_root = cfg.data_dir / split_name / "images"
    gt_instances = cfg.data_dir / split_name / "instances"
    
    assert images_root.exists(), f"Images directory not found: {images_root}"
    assert gt_instances.exists(), f"GT directory not found: {gt_instances}"

    # Generate sequence map
    seqmap_path = cfg.output_dir / f"{split_name}.seqmap"
    generate_seqmap(seq_list, images_root, seqmap_path)

    # Initialize neural networks
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
    else:
        print(f"[!] Warning: ReID weights not found at {cfg.reid_weights}")

    seg_net = None
    if cfg.seg_weights.exists():
        print(f"[*] Loading AppleUNet128 weights from: {cfg.seg_weights}")
        seg_net = AppleUNet128().to(cfg.device)
        seg_checkpoint = torch.load(str(cfg.seg_weights), map_location=cfg.device, weights_only=False)
        seg_net.load_state_dict(seg_checkpoint)
        seg_net.eval()
    else:
        print(f"[!] Warning: Segmentation weights not found at {cfg.seg_weights}")

    print("=" * 80)
    print(f"[*] START INFERENCE: {split_name.upper()} | Sequences: {seq_list}")
    print(f"[*] Images directory: {images_root}")
    print(f"[*] Ground truth: {gt_instances}")
    print("=" * 80)

    total_frames = 0
    start_total = time.perf_counter()

    for seq in seq_list:
        seq_path = images_root / seq
        frames = sorted([f for f in os.listdir(seq_path) if f.endswith((".png", ".jpg"))])
        if not frames:
            continue

        tracker = AppleFaceIDTracker(cfg=cfg)
        buffered = []
        orig_h, orig_w = None, None
        seq_start = time.perf_counter()

        pbar = tqdm(frames, desc=f"Track {seq}", unit="frame")
        for f_idx, f_name in enumerate(pbar):
            img_p = str(seq_path / f_name)
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
        print(f"[{seq}] Processed {len(frames)} frames | Speed: {fps:.2f} FPS")

        # Save predictions in RLE MOTS format
        out_txt = cfg.output_dir / f"{seq}.txt"
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

    # Run official eval.py benchmark
    print("\n" + "=" * 80)
    print(f"[*] RUNNING MOTS EVALUATION FOR {split_name.upper()}...")
    print("=" * 80)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(MOTS_TOOLS_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [
        sys.executable,
        str(EVAL_SCRIPT),
        str(cfg.output_dir),
        str(gt_instances),
        str(seqmap_path),
    ]

    process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    print(process.stdout)

    metrics = {
        "Split": f"Test {seq_list}",
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
    test_metrics = run_evaluation(seq_list=cfg.test_seqs, split_name="testing", cfg=cfg)

    df = pd.DataFrame([test_metrics])
    print("\n" + "=" * 80)
    print("FINAL BENCHMARK RESULTS (TEST SPLIT: 0006 - 0008)")
    print("=" * 80)
    print(df.to_string(index=False))