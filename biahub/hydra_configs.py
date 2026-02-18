"""
Hydra-compatible structured configuration definitions for biahub.

These dataclasses work with Hydra's structured configs while maintaining
compatibility with Pydantic for validation when needed.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class DetectPeaksConfig:
    threshold_abs: float = 110.0
    nms_distance: int = 16
    min_distance: int = 0
    block_size: List[int] = field(default_factory=lambda: [8, 8, 8])


@dataclass
class EdgeGraphConfig:
    method: Literal["knn", "radius", "full"] = "knn"
    k: Optional[int] = 5
    radius: Optional[float] = None


@dataclass
class CostMatrixConfig:
    weights: Dict[str, float] = field(default_factory=lambda: {
        "dist": 0.5,
        "edge_angle": 1.0,
        "edge_length": 1.0,
        "pca_dir": 0.0,
        "pca_aniso": 0.0,
        "edge_descriptor": 0.0,
    })
    normalize: bool = False


@dataclass
class FilterMatchesConfig:
    angle_threshold: float = 0.0
    direction_threshold: float = 0.0
    min_distance_quantile: float = 0.01
    max_distance_quantile: float = 0.95


@dataclass
class HungarianMatchConfig:
    distance_metric: Literal["euclidean", "cosine", "cityblock"] = "euclidean"
    cost_threshold: float = 0.10
    max_ratio: float = 0.8
    cross_check: bool = False
    edge_graph_settings: EdgeGraphConfig = field(default_factory=EdgeGraphConfig)
    cost_matrix_settings: CostMatrixConfig = field(default_factory=CostMatrixConfig)


@dataclass
class MatchDescriptorConfig:
    distance_metric: Literal["euclidean", "cosine", "cityblock"] = "euclidean"
    max_ratio: float = 0.8
    cross_check: bool = False


@dataclass
class QCBeadsRegistrationConfig:
    iterations: int = 1
    score_threshold: float = 0.40
    score_centroid_mask_radius: int = 6


@dataclass
class BeadsMatchConfig:
    algorithm: Literal["hungarian", "match_descriptor"] = "hungarian"
    t_reference: Literal["first", "previous"] = "first"
    source_peaks_settings: DetectPeaksConfig = field(default_factory=DetectPeaksConfig)
    target_peaks_settings: DetectPeaksConfig = field(default_factory=DetectPeaksConfig)
    match_descriptor_settings: MatchDescriptorConfig = field(default_factory=MatchDescriptorConfig)
    hungarian_match_settings: HungarianMatchConfig = field(default_factory=HungarianMatchConfig)
    filter_matches_settings: FilterMatchesConfig = field(default_factory=FilterMatchesConfig)
    qc_settings: QCBeadsRegistrationConfig = field(default_factory=QCBeadsRegistrationConfig)


@dataclass
class EvalTransformConfig:
    validation_window_size: int = 10
    validation_tolerance: float = 1000.0
    interpolation_window_size: int = 3
    interpolation_type: Literal["linear", "cubic"] = "linear"


@dataclass
class AffineTransformConfig:
    t_reference: Literal["first", "previous"] = "first"
    transform_type: Literal["euclidean", "similarity", "affine"] = "euclidean"
    approx_transform: List[List[float]] = field(default_factory=lambda: np.eye(4).tolist())
    use_prev_t_transform: bool = True


@dataclass
class AntsRegistrationConfig:
    crop: bool = False
    ref_mask_radius: float = 0.8
    clip: bool = True
    sobel_filter: bool = False


@dataclass
class ManualRegistrationConfig:
    time_index: int = 0
    affine_90degree_rotation: int = 0


@dataclass
class EstimateRegistrationConfig:
    """Configuration for estimate-registration command"""
    target_channel_name: str = MISSING
    source_channel_name: str = MISSING
    estimation_method: Literal["manual", "beads", "ants"] = "manual"
    beads_match_settings: Optional[BeadsMatchConfig] = None
    affine_transform_settings: AffineTransformConfig = field(default_factory=AffineTransformConfig)
    eval_transform_settings: Optional[EvalTransformConfig] = None
    ants_registration_settings: Optional[AntsRegistrationConfig] = None
    manual_registration_settings: Optional[ManualRegistrationConfig] = field(
        default_factory=ManualRegistrationConfig
    )
    verbose: bool = False


@dataclass
class DeskewConfig:
    """Configuration for deskew command"""
    pixel_size_um: float = MISSING
    ls_angle_deg: float = MISSING
    px_to_scan_ratio: Optional[float] = None
    scan_step_um: Optional[float] = None
    keep_overhang: bool = False
    average_n_slices: int = 3


@dataclass
class DataConfig:
    """Data paths configuration"""
    input_path: str = MISSING
    output_path: str = MISSING
    source_position_dirpaths: List[str] = field(default_factory=list)
    target_position_dirpaths: List[str] = field(default_factory=list)


@dataclass
class WorkflowConfig:
    """Workflow orchestration configuration"""
    name: str = "default"
    steps: List[str] = field(default_factory=list)


@dataclass
class BiahubConfig:
    """Main Biahub configuration"""
    verbose: bool = False
    output_dir: str = "${hydra:runtime.output_dir}"
    data: DataConfig = field(default_factory=DataConfig)
    registration: Optional[EstimateRegistrationConfig] = None
    deskew: Optional[DeskewConfig] = None
    workflow: Optional[WorkflowConfig] = None


# Register configs with Hydra's ConfigStore
cs = ConfigStore.instance()
cs.store(name="base_config", node=BiahubConfig)
cs.store(group="registration", name="base", node=EstimateRegistrationConfig)
cs.store(group="deskew", name="base", node=DeskewConfig)
cs.store(group="data", name="base", node=DataConfig)
cs.store(group="workflow", name="base", node=WorkflowConfig)
