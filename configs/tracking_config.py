from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import torch

BASE_DIR = Path(__file__).resolve().parent.parent

@dataclass
class DetectorConfig:
    conf_thresh: float = 0.36
    iou_thresh: float = 0.55
    imgsz: int = 960

@dataclass
class CMCConfig:
    scale_down_size: float = 640.0
    max_corners: int = 300
    quality_level: float = 0.005
    min_distance: float = 5.0
    block_size: int = 3
    min_points_ransac: int = 4

@dataclass
class ScalePriorConfig:
    min_conf_for_scene_scale: float = 0.50
    sigma_ratio: float = 0.35
    bg_probability_floor: float = 0.20
    min_aspect_ratio: float = 0.35
    min_visibility_ratio: float = 0.25
    nwd_nms_thresh: float = 0.60

@dataclass
class TopologyConfig:
    radius_scale: float = 3.5
    max_neighbors: int = 9
    chamfer_cutoff: float = 1.5
    context_crop_scale: float = 2.2

@dataclass
class KalmanConfig:
    std_weight_pos: float = 0.05
    std_weight_vel: float = 0.00625
    min_score_clip: float = 0.10

@dataclass
class AssociationConfig:
    high_score_thresh: float = 0.75
    w_iou_base: float = 0.60
    w_reid_max: float = 0.39
    w_dist_base: float = 0.20
    iou_reid_gate: float = 0.50
    reid_cutoff_on_zero_iou: float = 0.35
    scale_displacement_limit: float = 3.00
    match_cost_round1: float = 0.80
    match_cost_round2: float = 0.82
    match_cost_lost: float = 0.45
    reid_vis_weight: float = 0.65
    reid_topo_weight: float = 0.61
    drift_temporal_factor: float = 0.30
    max_drift_multiplier: float = 4.0
    min_hits_to_activate: int = 3
    max_age_active: int = 31
    max_age_tentative: int = 2
    alpha_embedding_update: float = 0.85
    gate_disp_round1: float = 1.45
    gate_disp_round2: float = 1.80
    high_iou_lock_thresh: float = 0.45
    exclusivity_margin: float = 0.12
    high_iou_cost_weight: float = 0.35

@dataclass
class PipelineConfig:
    base_dir: Path = field(default_factory=lambda: BASE_DIR)
    data_dir: Path = field(default_factory=lambda: BASE_DIR / "data" / "APPLE_MOTS")
    weights_dir: Path = field(default_factory=lambda: BASE_DIR / "weights")
    output_dir: Path = field(default_factory=lambda: BASE_DIR / "output_tracks")
    tools_dir: Path = field(default_factory=lambda: BASE_DIR / "tools")
    
    detector_weights_name: str = "apple_detector.pt"
    reid_weights_name: str = "apple_faceid_metric.pth"
    seg_weights_name: str = "apple_crop_unet_best.pth"
    
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    cmc: CMCConfig = field(default_factory=CMCConfig)
    scale_prior: ScalePriorConfig = field(default_factory=ScalePriorConfig)
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    kalman: KalmanConfig = field(default_factory=KalmanConfig)
    assoc: AssociationConfig = field(default_factory=AssociationConfig)
    
    train_seqs: List[str] = field(default_factory=lambda: ["0000", "0001", "0002"])
    val_seqs: List[str] = field(default_factory=lambda: ["0003", "0004", "0005"])
    test_seqs: List[str] = field(default_factory=lambda: ["0006", "0007", "0008"])
    
    device: torch.device = field(
        default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    @property
    def detector_weights(self) -> Path:
        return self.weights_dir / self.detector_weights_name

    @property
    def reid_weights(self) -> Path:
        return self.weights_dir / self.reid_weights_name

    @property
    def seg_weights(self) -> Path:
        return self.weights_dir / self.seg_weights_name