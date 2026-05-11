"""人物检测器模块"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import logging

from .sam_wrapper import SAMWrapper


logger = logging.getLogger(__name__)


class PersonDetector:
    """人物检测器"""

    def __init__(
        self,
        sam_wrapper: SAMWrapper,
        min_person_size: int = 256,
        max_persons_per_image: int = 10,
        confidence_threshold: float = 0.5,
        prompt_strategy: str = "grid",
        image_enhancer: Optional['ImageEnhancer'] = None
    ):
        """
        初始化人物检测器

        Args:
            sam_wrapper: SAM 模型包装器
            min_person_size: 人物最小尺寸（像素）
            max_persons_per_image: 单张图像最大检测人数
            confidence_threshold: 置信度阈值
            prompt_strategy: 提示策略 (grid, center)
            image_enhancer: 可选的图像增强器（用于提高检测精度）
        """
        self.sam = sam_wrapper
        self.min_person_size = min_person_size
        self.max_persons_per_image = max_persons_per_image
        self.confidence_threshold = confidence_threshold
        self.prompt_strategy = prompt_strategy
        self.image_enhancer = image_enhancer

    def detect_persons(
        self,
        image: np.ndarray
    ) -> List[Tuple[np.ndarray, Dict]]:
        """
        检测图像中的所有人物

        Args:
            image: BGR 格式的图像（OpenCV 默认格式）

        Returns:
            [(掩码, 元数据字典), ...] 列表
        """
        # 应用图像增强（如果启用）
        if self.image_enhancer is not None:
            image = self.image_enhancer.enhance_for_detection(image)
            logger.debug("已应用图像增强预处理")

        # 转换为 RGB（SAM 需要 RGB 格式）
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 使用 SAM 自动分割
        masks = self.sam.segment_auto(
            image_rgb,
            prompt_strategy=self.prompt_strategy
        )

        logger.info(f"检测到 {len(masks)} 个候选对象")

        # 过滤低质量掩码
        valid_detections = []
        for idx, mask in enumerate(masks):
            # 计算掩码质量指标
            quality_score = self._evaluate_mask_quality(mask, image)
            metadata = {
                'mask_id': idx,
                'quality_score': quality_score,
                'area': int(np.sum(mask > 0)),
                'bbox': self._get_mask_bbox(mask)
            }

            # 过滤条件
            if self._is_valid_detection(mask, metadata):
                valid_detections.append((mask, metadata))

        # 按面积排序（大的优先）
        valid_detections.sort(
            key=lambda x: x[1]['area'],
            reverse=True
        )

        # 限制最大数量
        valid_detections = valid_detections[:self.max_persons_per_image]

        logger.info(f"过滤后保留 {len(valid_detections)} 个有效检测")
        return valid_detections

    def _is_valid_detection(
        self,
        mask: np.ndarray,
        metadata: Dict
    ) -> bool:
        """
        判断是否为有效的检测

        Args:
            mask: 掩码
            metadata: 元数据

        Returns:
            是否有效
        """
        # 检查掩码面积
        bbox = metadata['bbox']
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]

        if width < self.min_person_size or height < self.min_person_size:
            logger.debug(f"掩码尺寸过小: {width}x{height}")
            return False

        # 检查质量分数
        if metadata['quality_score'] < self.confidence_threshold:
            logger.debug(f"质量分数过低: {metadata['quality_score']:.2f}")
            return False

        # 检查掩码占比（避免过小的分割）
        mask_area = metadata['area']
        bbox_area = width * height
        coverage = mask_area / bbox_area if bbox_area > 0 else 0

        if coverage < 0.3:  # 掩码应该至少占边框的 30%
            logger.debug(f"掩码占比过低: {coverage:.2f}")
            return False

        return True

    def _evaluate_mask_quality(
        self,
        mask: np.ndarray,
        image: np.ndarray
    ) -> float:
        """
        评估掩码质量

        Args:
            mask: 掩码
            image: 原始图像

        Returns:
            质量分数 (0-1)
        """
        # 计算掩码边界框
        bbox = self._get_mask_bbox(mask)
        x1, y1, x2, y2 = bbox

        # 裁剪掩码区域
        mask_roi = mask[y1:y2, x1:x2]
        image_roi = image[y1:y2, x1:x2]

        # 计算边缘平滑度（简单的梯度检测）
        if mask_roi.size > 0:
            # 掩码边缘
            kernel = np.ones((5, 5), np.uint8)
            mask_dilated = cv2.dilate(mask_roi.astype(np.uint8), kernel, iterations=1)
            mask_eroded = cv2.erode(mask_roi.astype(np.uint8), kernel, iterations=1)
            edge = mask_dilated - mask_eroded

            # 边缘占比（边缘像素 / 总像素）
            edge_ratio = np.sum(edge > 0) / (mask_roi.size + 1e-6)

            # 质量分数：边缘占比适中（0.05-0.2）为佳
            edge_score = 1.0 - min(abs(edge_ratio - 0.1) / 0.1, 1.0)
        else:
            edge_score = 0.0

        # 计算面积分数（较大的掩码更可能是有效人物）
        area = np.sum(mask > 0)
        area_score = min(area / (512 * 512), 1.0)

        # 综合质量分数
        quality_score = 0.7 * area_score + 0.3 * edge_score

        return float(quality_score)

    def _get_mask_bbox(self, mask: np.ndarray) -> Tuple[int, int, int, int]:
        """
        获取掩码的边界框

        Args:
            mask: 掩码

        Returns:
            (x1, y1, x2, y2) 边界框坐标
        """
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        if not np.any(rows) or not np.any(cols):
            return (0, 0, 0, 0)

        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]

        return (int(x1), int(y1), int(x2), int(y2))

    def extract_person_images(
        self,
        image: np.ndarray,
        detections: List[Tuple[np.ndarray, Dict]]
    ) -> List[Tuple[np.ndarray, Dict]]:
        """
        从图像中提取人物图像

        Args:
            image: 原始图像
            detections: 检测结果列表

        Returns:
            [(人物图像, 元数据), ...] 列表
        """
        extracted = []

        for mask, metadata in detections:
            # 获取边界框
            bbox = metadata['bbox']
            x1, y1, x2, y2 = bbox

            # 裁剪图像和掩码
            person_image = image[y1:y2, x1:x2].copy()
            person_mask = mask[y1:y2, x1:x2]

            # 更新元数据
            metadata['bbox'] = bbox
            metadata['image_shape'] = person_image.shape

            extracted.append((person_image, person_mask, metadata))

        return extracted

    def detect_and_extract(
        self,
        image_path: str
    ) -> List[Tuple[np.ndarray, np.ndarray, Dict]]:
        """
        从图像文件中检测并提取人物

        Args:
            image_path: 图像文件路径

        Returns:
            [(人物图像, 掩码, 元数据), ...] 列表
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"无法读取图像: {image_path}")
            return []

        # 检测人物
        detections = self.detect_persons(image)

        if not detections:
            logger.warning(f"未检测到有效人物: {image_path}")
            return []

        # 提取人物图像
        extracted = self.extract_person_images(image, detections)

        logger.info(f"从 {image_path} 提取了 {len(extracted)} 个人物")
        return extracted


def detect_persons_batch(
    image_paths: List[str],
    detector: PersonDetector,
    progress_callback: Optional[callable] = None
) -> Dict[str, List[Tuple[np.ndarray, np.ndarray, Dict]]]:
    """
    批量检测人物

    Args:
        image_paths: 图像路径列表
        detector: 人物检测器
        progress_callback: 进度回调函数

    Returns:
        {图像路径: [(人物图像, 掩码, 元数据), ...]} 字典
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
