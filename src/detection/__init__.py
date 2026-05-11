"""检测模块"""

from .sam_wrapper import SAMWrapper
from .person_detector import PersonDetector
from .anime_face_detector import AnimeFaceDetector
from .yolo_face_detector import YOLOFaceDetector
from .face_detector_factory import FaceDetectorFactory, get_detector

__all__ = [
    'SAMWrapper',
    'PersonDetector',
    'AnimeFaceDetector',
    'YOLOFaceDetector',
    'FaceDetectorFactory',
    'get_detector'
]
