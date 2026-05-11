"""基于 YOLO 的二次元人脸检测器模块"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import logging

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False


logger = logging.getLogger(__name__)


class YOLOFaceDetector:
    """基于 YOLO 的二次元人脸检测器"""

    def __init__(
        self,
        model_path: str = "yolov8n_animeface.pt",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        max_detections: int = 10,
        device: str = "cuda",
        image_enhancer: Optional['ImageEnhancer'] = None
    ):
        """
        初始化 YOLO 人脸检测器

        Args:
            model_path: YOLO 模型权重路径
            confidence_threshold: 置信度阈值 (0-1)
            iou_threshold: NMS IoU 阈值 (0-1)
            max_detections: 单张图像最大检测数量
            device: 运行设备 (cuda/cpu)
            image_enhancer: 可选的图像增强器（用于提高检测精度）
        """
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError(
                "需要安装 ultralytics 库。请运行: pip install ultralytics"
            )

        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.device = device
        self.image_enhancer = image_enhancer

        # 加载模型
        self.model = self._load_model()

        logger.info(
            f"YOLO 人脸检测器初始化成功: {model_path}, "
            f"置信度阈值={confidence_threshold}, 设备={device}"
        )

    def _load_model(self) -> 'YOLO':
        """
        加载 YOLO 模型

        Returns:
            YOLO 模型实例
        """
        model_path = Path(self.model_path)

        # 如果模型文件不存在，尝试下载预训练模型
        if not model_path.exists():
            logger.warning(f"模型文件不存在: {self.model_path}")
            logger.info("尝试使用 YOLOv8 预训练模型...")
            try:
                # 使用官方 YOLOv8n 模型（会自动下载）
                model = YOLO('yolov8n.pt')
                logger.info("已加载 YOLOv8n 预训练模型")
                return model
            except Exception as e:
                logger.error(f"加载预训练模型失败: {e}")
                raise

        # 加载自定义权重
        try:
            model = YOLO(str(model_path))
            logger.info(f"成功加载模型: {model_path}")
            return model
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise

    def detect_faces(
        self,
        image: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
        """
        检测图像中的人脸

        Args:
            image: BGR 格式的图像

        Returns:
            人脸边界框列表 [(x, y, w, h), ...]
        """
        # 应用图像增强（如果启用）
        detection_image = image
        if self.image_enhancer is not None:
            detection_image = self.image_enhancer.enhance_for_detection(image)
            logger.debug("已应用图像增强预处理")

        # YOLO 推理
        results = self.model(
            detection_image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            max_det=self.max_detections,
            device=self.device,
            verbose=False
        )

        # 提取边界框
        faces = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                for box in result.boxes:
                    # 获取坐标（xyxy 格式：x1, y1, x2, y2）
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    w = int(x2 - x1)
                    h = int(y2 - y1)
                    faces.append((int(x1), int(y1), w, h))

        logger.debug(f"检测到 {len(faces)} 个人脸")
        return faces

    def extract_face_images(
        self,
        image: np.ndarray,
        padding: int = 20
    ) -> List[Tuple[np.ndarray, Dict]]:
        """
        从图像中提取人脸图像

        Args:
            image: 原始图像
            padding: 边距（像素）

        Returns:
            [(人脸图像, 元数据), ...] 列表
        """
        # 检测人脸
        faces = self.detect_faces(image)

        if not faces:
            logger.debug("未检测到人脸")
            return []

        extracted = []

        for idx, (x, y, w, h) in enumerate(faces):
            # 添加边距
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image.shape[1], x + w + padding)
            y2 = min(image.shape[0], y + h + padding)

            # 裁剪人脸区域
            face_image = image[y1:y2, x1:x2].copy()

            # 元数据
            metadata = {
                'face_id': idx,
                'bbox': (x, y, w, h),
                'bbox_with_padding': (x1, y1, x2, y2),
                'image_shape': face_image.shape,
                'detector_type': 'YOLO',
                'confidence': 1.0  # YOLO 的置信度需要从 results 中获取
            }

            extracted.append((face_image, metadata))

        return extracted

    def detect_and_extract(
        self,
        image_path: str
    ) -> List[Tuple[np.ndarray, Dict]]:
        """
        从图像文件中检测并提取人脸

        Args:
            image_path: 图像文件路径

        Returns:
            [(人脸图像, 元数据), ...] 列表
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"无法读取图像: {image_path}")
            return []

        # 提取人脸
        extracted = self.extract_face_images(image)

        logger.info(f"从 {image_path} 提取了 {len(extracted)} 个人脸")
        return extracted

    @staticmethod
    def download_animeface_model(
        model_name: str = "yolov8n_animeface",
        output_dir: str = "./models"
    ) -> str:
        """
        下载二次元人脸检测模型

        Args:
            model_name: 模型名称
            output_dir: 输出目录

        Returns:
            模型文件路径
        """
        logger.info(f"下载二次元人脸检测模型: {model_name}")

        # 目前需要手动从 GitHub 下载预训练权重
        # YOLOv8-AnimeFace: https://github.com/Fuyucch1/yolov8_animeface
        # YOLOv5-Anime: https://github.com/zymk9/yolov5_anime

        logger.warning(
            "请手动下载预训练权重：\n"
            "1. YOLOv8-AnimeFace: "
            "https://github.com/Fuyucch1/yolov8_animeface\n"
            "2. YOLOv5-Anime: "
            "https://github.com/zymk9/yolov5_anime\n\n"
            "下载后将 .pt 文件放到 models/ 目录"
        )

        return ""


def detect_faces_batch(
    image_paths: List[str],
    detector: YOLOFaceDetector,
    progress_callback: Optional[callable] = None
) -> Dict[str, List[Tuple[np.ndarray, Dict]]]:
    """
    批量检测人脸

    Args:
        image_paths: 图像路径列表
        detector: 人脸检测器
        progress_callback: 进度回调函数

    Returns:
        {图像路径: [(人脸图像, 元数据), ...]} 字典
    """
    results = {}

    for idx, image_path in enumerate(image_paths):
        try:
            extracted = detector.detect_and_extract(image_path)
            if extracted:
                results[image_path] = extracted

            # 进度回调
            if progress_callback:
                progress_callback(idx + 1, len(image_paths), image_path)

        except Exception as e:
            logger.error(f"处理图像失败 {image_path}: {e}")
            continue

    return results
