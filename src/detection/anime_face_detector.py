"""二次元人脸检测器模块（SAM + 几何裁剪混合方案）"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import logging

from .sam_wrapper import SAMWrapper
from .person_detector import PersonDetector


logger = logging.getLogger(__name__)


class AnimeFaceDetector:
    """二次元人脸检测器（SAM + 几何裁剪混合方案）"""

    def __init__(
        self,
        sam_wrapper: SAMWrapper,
        face_crop_ratio: float = 0.5,
        face_center_crop: float = 0.6,
        min_person_size: int = 256,
        image_enhancer: Optional['ImageEnhancer'] = None
    ):
        """
        初始化二次元人脸检测器

        Args:
            sam_wrapper: SAM 模型包装器
            face_crop_ratio: 脸部上半部分占比（0-1）
            face_center_crop: 人脸中心裁剪比例（0-1）
            min_person_size: 人物最小尺寸
            image_enhancer: 可选的图像增强器
        """
        self.sam = sam_wrapper
        self.face_crop_ratio = face_crop_ratio
        self.face_center_crop = face_center_crop
        self.min_person_size = min_person_size

        # 初始化人物检测器（使用 SAM）
        self.person_detector = PersonDetector(
            sam_wrapper=sam_wrapper,
            min_person_size=min_person_size,
            prompt_strategy="center",  # 使用中心点提示，更可能检测到人物
            image_enhancer=image_enhancer  # 传递增强器
        )

        logger.info("二次元人脸检测器（SAM + 几何裁剪）初始化成功")

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
        # 使用 SAM 检测人物
        detections = self.person_detector.detect_persons(image)

        # 转换为人脸边界框
        face_boxes = []
        for _, (person_image, person_mask, det_metadata) in enumerate(
            self.person_detector.extract_person_images(image, detections)
        ):
            # 获取人物边界框
            bbox = det_metadata['bbox']
            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1

            # 计算人脸区域（上半部分的中心）
            face_height = int(h * self.face_crop_ratio)
            face_y = y1
            face_x = x1
            face_width = w

            # 进一步裁剪中心区域
            center_crop_height = int(face_height * self.face_center_crop)
            if center_crop_height < face_height:
                center_y = face_y + (face_height - center_crop_height) // 2
                face_y = center_y
                face_height = center_crop_height

            face_boxes.append((face_x, face_y, face_width, face_height))

        return face_boxes

    def extract_face_images(
        self,
        image: np.ndarray,
        padding: int = 20,
        crop_mode: str = "face",
        crop_padding_ratio: float = 0.5
    ) -> List[Tuple[np.ndarray, Dict]]:
        """
        从图像中提取人脸图像

        Args:
            image: 原始图像
            padding: 基础边距（像素）
            crop_mode: 裁剪模式
                - "face": 仅人脸
                - "upper_body": 上半身（推荐）
                - "full_body": 全身
            crop_padding_ratio: 额外边距比例

        Returns:
            [(人脸图像, 元数据), ...] 列表
        """
        # 检测人物
        detections = self.person_detector.detect_persons(image)

        if not detections:
            logger.warning("未检测到人物，无法提取人脸")
            return []

        extracted = []

        for idx, (person_image, person_mask, det_metadata) in enumerate(
            self.person_detector.extract_person_images(image, detections)
        ):
            # 获取原始图像中的位置
            bbox = det_metadata['bbox']
            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1

            # 根据裁剪模式决定提取区域
            if crop_mode == "face":
                # 仅人脸：使用原始的几何裁剪逻辑
                face_height = int(h * self.face_crop_ratio)
                face_y = y1
                face_x = x1

                # 进一步裁剪中心区域
                center_crop_height = int(face_height * self.face_center_crop)
                if center_crop_height < face_height:
                    center_y = face_y + (face_height - center_crop_height) // 2
                    face_y = center_y
                    face_height = center_crop_height

                # 添加边距
                face_x1 = max(0, face_x - padding)
                face_y1 = max(0, face_y - padding)
                face_x2 = min(image.shape[1], face_x + w + padding)
                face_y2 = min(image.shape[0], face_y + face_height + padding)

            elif crop_mode == "upper_body":
                # 上半身：使用人物的上半部分
                upper_body_ratio = 0.7  # 上 70%
                upper_height = int(h * upper_body_ratio)
                face_y = y1
                face_x = x1

                # 添加适度边距
                expand_x = int(w * crop_padding_ratio * 0.5)
                expand_y = int(h * crop_padding_ratio * 0.3)

                face_x1 = max(0, face_x - expand_x)
                face_y1 = max(0, face_y - expand_y)
                face_x2 = min(image.shape[1], face_x + w + expand_x)
                face_y2 = min(image.shape[0], face_y + upper_height + expand_y)

            elif crop_mode == "full_body":
                # 全身：使用完整的人物检测区域
                face_x1 = max(0, x1 - padding)
                face_y1 = max(0, y1 - padding)
                face_x2 = min(image.shape[1], x2 + padding)
                face_y2 = min(image.shape[0], y2 + padding)

            else:
                # 默认使用 face 模式
                face_height = int(h * self.face_crop_ratio)
                face_x1 = max(0, x1 - padding)
                face_y1 = max(0, y1 - padding)
                face_x2 = min(image.shape[1], x2 + padding)
                face_y2 = min(image.shape[0], y1 + face_height + padding)

            # 从原始图像裁剪区域
            face_image = image[face_y1:face_y2, face_x1:face_x2].copy()

            # 元数据
            metadata = {
                'face_id': idx,
                'person_bbox': bbox,
                'face_bbox': (face_x1, face_y1, face_x2, face_y2),
                'face_crop_ratio': self.face_crop_ratio,
                'face_center_crop': self.face_center_crop,
                'image_shape': face_image.shape,
                'crop_mode': crop_mode,
                'confdence': 1.0
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
