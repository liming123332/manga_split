"""人脸检测器工厂模块 - 支持多种检测方法"""

from typing import Optional
import logging

from .yolo_face_detector import YOLOFaceDetector
from .anime_face_detector import AnimeFaceDetector
from .sam_wrapper import SAMWrapper


logger = logging.getLogger(__name__)


class FaceDetectorFactory:
    """人脸检测器工厂类"""

    @staticmethod
    def create_detector(
        method: str = "yolo",
        **kwargs
    ):
        """
        创建人脸检测器

        Args:
            method: 检测方法
                - "yolo": YOLO 人脸检测（快速，推荐）
                - "sam": SAM + 几何裁剪（高精度）
            **kwargs: 传递给具体检测器的参数

        Returns:
            人脸检测器实例

        Raises:
            ValueError: 不支持的检测方法
        """
        method = method.lower()

        if method == "yolo":
            logger.info("使用 YOLO 人脸检测器")
            # YOLO 检测器接受的参数
            yolo_params = {}
            valid_params = [
                'model_path', 'confidence_threshold', 'iou_threshold',
                'max_detections', 'device', 'image_enhancer'
            ]
            for key in valid_params:
                if key in kwargs:
                    yolo_params[key] = kwargs[key]
            return YOLOFaceDetector(**yolo_params)

        elif method == "sam":
            logger.info("使用 SAM + 几何裁剪检测器")
            # 需要 sam_wrapper 参数
            if 'sam_wrapper' not in kwargs:
                raise ValueError("SAM 方法需要提供 sam_wrapper 参数")
            # SAM 检测器接受的参数
            sam_params = {}
            valid_params = [
                'sam_wrapper', 'face_crop_ratio', 'face_center_crop',
                'min_person_size', 'image_enhancer'
            ]
            for key in valid_params:
                if key in kwargs:
                    sam_params[key] = kwargs[key]
            return AnimeFaceDetector(**sam_params)

        else:
            raise ValueError(
                f"不支持的检测方法: {method}。"
                f"支持的选项: 'yolo', 'sam'"
            )


def get_detector(
    method: str = "yolo",
    sam_wrapper: Optional[SAMWrapper] = None,
    **kwargs
):
    """
    获取人脸检测器的便捷函数

    Args:
        method: 检测方法 ("yolo", "sam")
        sam_wrapper: SAM 包装器（仅当 method="sam" 时需要）
        **kwargs: 其他参数

    Returns:
        人脸检测器实例
    """
    # 为 SAM 方法添加 sam_wrapper
    if method.lower() == "sam" and sam_wrapper is not None:
        kwargs['sam_wrapper'] = sam_wrapper

    return FaceDetectorFactory.create_detector(method, **kwargs)
