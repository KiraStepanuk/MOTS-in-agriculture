# MOTS in Agriculture: Apple Tracking and Segmentation

Repository for the Multi-Object Tracking and Segmentation (MOTS) framework for apples in orchard environments. The pipeline combines an RT-DETR detector, a custom U-Net crop segmenter, an AppleFaceNet metric Re-ID network, and egocentric Star Graph topology matching.

---

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

## 6. Citation & Dataset Reference

```text
Jong, S. de ., Baja, H., & Valente, J. (2022). 
Dataset for benchmarking Multiple Object Tracking and Segmentation (MOTS) in an apple orchard field. 
Zenodo. https://doi.org/10.5281/zenodo.5939726
```
```
