# MOTS in Agriculture: Multi-Object Tracking and Segmentation of Homogeneous Fruits in Complex Orchard Environments

Official implementation of the multi-stage **Multi-Object Tracking and Segmentation (MOTS)** pipeline tailored for apple orchards, developed within the **Lviv Data Science Summer School 2026** (*"Advancing AI-Driven and Data-Intensive Agriculture: From Genomics and Phenotyping to Digital Twins and Decision Support Systems"*), co-organized by the **Ukrainian Catholic University (UCU)** and the **U.S. National Academies of Sciences, Engineering, and Medicine (NASEM)**.

---

## Authors & Project Team

- **Kira Stepanuk** (BSc Student) — Odessa National Maritime University
- **Denys Klekots** (PhD Student) — Taras Shevchenko National University of Kyiv
- **Svyatoslav Borysyuk** (BSc Student) — Ivan Franko National University of Lviv
- **Project Mentor: Dr. João Valente** (Senior Scientist) — SAID Lab, CSIC, Spain
---


## Repository Structure

```text
MOTS-in-agriculture/
├── assets/                          # Metric plots, loss curves, and evaluation tables
├── configs/
│   └── tracking_config.py           # Dataclass configs for tracker, CMC, Kalman, and thresholds
├── data/
│   └── APPLE_MOTS/                  # Benchmark dataset directory
├── tools/                           # mots_tools submodule for official KITTI/APPLE MOTS evaluation
├── tracking/
│   ├── run_tracker.py               # Main inference script (Standard test sequences: 0006-0008)
│   └── run_tracker_ignore_regions.py# Inference script with background ignore masks (0010-0012)
├── training/
│   ├── 01_train_detector.py         # RT-DETR-L dataset conversion & training script
│   ├── 02_train_segmenter.py        # AppleUNet128 crop extraction & Dice/BCE training
│   └── 03_train_reid.py             # AppleFaceNet ArcFace + Triplet metric learning script
├── weights/                         # Pretrained model checkpoints (.pt and .pth)
├── requirements.txt
└── README.md
```

## 1. Setup and Environment

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/KiraStepanuk/MOTS-in-agriculture.git
cd MOTS-in-agriculture
pip install -r requirements.txt
```

A CUDA-compatible GPU is required for efficient training and tracking.

---

## 2. Download Pretrained Weights

Download the trained model checkpoints and place them directly into the `weights/` directory:

- **apple_detector.pt** (RT-DETR-L detector):  
  [Download via Google Drive](https://drive.google.com/file/d/13adZ5sDvUKY2q1NOV3PbKuaU_pEgwHVq/view?usp=sharing)
- **apple_crop_unet_best.pth** (AppleUNet128 instance segmenter):  
  [Download via Google Drive](https://drive.google.com/file/d/1WsfZPziPN8uXVRzP6MmK8sLUZ6IZoRTR/view?usp=sharing)
- **apple_faceid_metric.pth** (AppleFaceNet metric Re-ID):  
  [Download via Google Drive](https://drive.google.com/file/d/13ZBZuoLBXcuq1bJqnYJ4jw2W9ZfrUPbc/view?usp=sharing)

Ensure your `weights/` folder looks like this:
```text
weights/
├── apple_detector.pt
├── apple_crop_unet_best.pth
└── apple_faceid_metric.pth
```

---

## 3. Download the Dataset

The project is evaluated on the official **APPLE MOTS** benchmark dataset:

- **Zenodo Official Source:** [doi.org/10.5281/zenodo.5939726](https://doi.org/10.5281/zenodo.5939726)  
- **Google Drive Mirror:** [Download Dataset Archive](https://drive.google.com/file/d/1jcc207qWqBUv1zIGMTEw5Ik2CTJ49IQh/view?usp=sharing)

Extract the archive into `data/APPLE_MOTS/` so the directory structure is organized as:

```text
data/
└── APPLE_MOTS/
    ├── train/
    │   ├── images/
    │   └── instances/
    └── testing/
        ├── images/
        └── instances/
```

---

## 4. Running the Tracker & Evaluation

The tracking scripts automatically perform frame-by-frame inference, generate MOTS RLE predictions in `output_tracks/`, and execute the official `eval.py` benchmark.

### Standard Test Set (Sequences 0006 - 0008):
```bash
python tracking/run_tracker.py
```

### Official Evaluation with Ignore Regions (Sequences 0010 - 0012):
```bash
python tracking/run_tracker_ignore_regions.py
```

---

## 5. Training Models (Optional)

If you want to train individual components from scratch:

1. **Train the RT-DETR-L Detector (15 epochs):**
   ```bash
   python training/01_train_detector.py
   ```
2. **Train the AppleUNet128 Segmenter (15 epochs):**
   ```bash
   python training/02_train_segmenter.py
   ```
3. **Train the AppleFaceNet Re-ID Model (20 epochs):**
   ```bash
   python training/03_train_reid.py
   ```

Training curves and performance plots will be automatically exported to the `assets/` directory.

---


---

## 6.Overview & Key Contributions

Tracking and segmenting fruits in orchard environments suffers from four core domain bottlenecks: **extreme visual homogeneity** (identical red-green spheres), **heavy leaf/branch occlusions**, **dense clustering**, and **erratic camera shake/parallax** from UAVs or wearable sensors.

Rather than relying on single monolithic networks that underperform on small orchard datasets, this framework decouples the problem into a specialized 4-stage pipeline:

1. **Detection (RT-DETR-L):** End-to-end real-time transformer detector operating at $960\times960$, coupled with **Normalized Wasserstein Distance (NWD-NMS)** to reliably separate dense, touching fruit clusters without heuristic box dropping.
2. **Instance Segmentation (AppleUNet128):** A lightweight 4-stage U-Net segmenter on padded ($+15\%$) $128\times128$ RoI crops with 2D Gaussian spatial competition, ensuring sharp, non-overlapping binary masks.
3. **Contextual Metric Re-ID (AppleFaceNet):** ResNet-18 backbone with an ArcFace angular margin loss ($s=16.0, m=0.25$) and Batch-Hard Triplet Loss trained on **extended $2.2\times$ neighborhood context crops**, turning surrounding foliage, twigs, and bark patterns into a unique biometric signature.
4. **Kinematic & Topological Tracking Engine:**
   - **Camera Motion Compensation (CMC)**
   - **Scale-Adaptive 2D Kalman Filter**
   - **Topological "Fruit Constellation" Star Graphs**
   - **Cascaded 3-Stage Hungarian Association** 

### Benchmark Results (APPLE MOTS Test Set)

| Method / Configuration | sMOTSA (%) ↑ | MOTSA (%) ↑ | MOTSP (%) ↑ | ID Switches ↓ | Speed (FPS) ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| TrackR-CNN (Bounding Box) | 34.80 | 46.80 | 80.40 | 1427 | — |
| PointTrack (from scratch) | 31.80 | 46.00 | 80.00 | 469 | 3.06 |
| PointTrack (KITTI pre-trained) | 38.70 | 52.90 | 80.00 | 524 | 3.06 |
| **Ours (Standard Test 0006–0008)** | **47.00** | **59.10** | **81.40** | **43** | 1.51 |
| **Ours (With Ignore Regions 0010–0012)** | **47.30** | **59.20** | **81.60** | **25** | 1.93 |

---
## 7. Citation & Dataset Reference

```text
Jong, S. de ., Baja, H., & Valente, J. (2022). 
Dataset for benchmarking Multiple Object Tracking and Segmentation (MOTS) in an apple orchard field. 
Zenodo. https://doi.org/10.5281/zenodo.5939726
```
