"""Project-owned YOLO-family plate detector boundary."""

from anpr_engine.detection.config import DetectorConfig, load_detector_config
from anpr_engine.detection.data import DetectionDataset, DetectionManifestSample, load_samples
from anpr_engine.detection.model import TinyYoloDetector, decode_predictions

__all__ = [
    "DetectionDataset",
    "DetectionManifestSample",
    "DetectorConfig",
    "TinyYoloDetector",
    "decode_predictions",
    "load_detector_config",
    "load_samples",
]
