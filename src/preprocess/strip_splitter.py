"""长条图分割模块"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple
import logging


logger = logging.getLogger(__name__)


class StripSplitter:
    """长条图分割器"""

    def __init__(
        self,
        min_aspect_ratio: float = 3.0,
        min_gap_size: int = 20,
        blur_kernel: int = 5,
        threshold_value: int = 200
    ):
        """
        初始化分割器

        Args:
            min_aspect_ratio: 判断为长条图的最小高宽比
            min_gap_size: 分割线最小像素宽度
            blur_kernel: 高斯模糊核大小
            threshold_value: 二值化阈值
        """
        self.min_aspect_ratio = min_aspect_ratio
        self.min_gap_size = min_gap_size
        self.blur_kernel = blur_kernel
        self.threshold_value = threshold_value

    def is_strip_image(self, image: np.ndarray) -> bool:
        """
        判断是否为长条图

        Args:
            image: 输入图像

        Returns:
            是否为长条图
        """
        height, width = image.shape[:2]
        aspect_ratio = height / width if width > 0 else 0

        return aspect_ratio >= self.min_aspect_ratio

    def detect_split_lines(self, image: np.ndarray) -> List[int]:
        """
        检测长条图的横向分割线位置

        Args:
            image: 输入图像（灰度或彩色）

        Returns:
            分割线的 y 坐标列表
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 高斯模糊降噪
        blurred = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)

        # 二值化
        _, binary = cv2.threshold(blurred, self.threshold_value, 255, cv2.THRESH_BINARY_INV)

        # 计算水平投影
        horizontal_proj = np.sum(binary, axis=1)

        # 平滑投影曲线
        kernel = np.ones(self.min_gap_size // 2 + 1) / (self.min_gap_size // 2 + 1)
        smoothed_proj = np.convolve(horizontal_proj, kernel, mode='same')

        # 寻找投影的低谷位置（分割线）
        split_lines = []
        threshold = np.max(smoothed_proj) * 0.1  # 投影值的 10% 作为阈值

        for i in range(1, len(smoothed_proj) - 1):
            # 检查是否为局部最小值
            if (smoothed_proj[i] < smoothed_proj[i - 1] and
                smoothed_proj[i] < smoothed_proj[i + 1] and
                smoothed_proj[i] < threshold):

                # 检查是否与已有的分割线距离足够远
                if not split_lines or i - split_lines[-1] > self.min_gap_size:
                    split_lines.append(i)

        # 添加图像顶部和底部
        split_lines = [0] + split_lines + [image.shape[0]]

        return sorted(split_lines)

    def split_strip_image(
        self,
        image_path: str,
        output_dir: str
    ) -> List[str]:
        """
        分割长条图并保存单格图像

        Args:
            image_path: 长条图路径
            output_dir: 输出目录

        Returns:
            保存的单格图像路径列表
        """
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"无法读取图像: {image_path}")
            return []

        # 检查是否为长条图
        if not self.is_strip_image(image):
            logger.info(f"图像不是长条图（高宽比 < {self.min_aspect_ratio}）: {image_path}")
            return []

        # 检测分割线
        split_lines = self.detect_split_lines(image)

        if len(split_lines) < 2:
            logger.warning(f"未检测到分割线: {image_path}")
            return []

        logger.info(f"检测到 {len(split_lines) - 1} 个分割点: {image_path}")

        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 分割并保存
        output_files = []
        stem_name = Path(image_path).stem

        for i in range(len(split_lines) - 1):
            y1, y2 = split_lines[i], split_lines[i + 1]

            # 跳过过小的分割
            if y2 - y1 < self.min_gap_size:
                continue

            # 裁剪图像
            panel = image[y1:y2, :]

            # 保存
            output_file = output_path / f"{stem_name}_panel_{i:03d}.png"
            cv2.imwrite(str(output_file), panel)
            output_files.append(str(output_file))

            logger.debug(f"保存分割图像: {output_file} (尺寸: {panel.shape})")

        logger.info(f"成功分割为 {len(output_files)} 个单格图像")
        return output_files

    def split_and_return_arrays(
        self,
        image_path: str
    ) -> List[np.ndarray]:
        """
        分割长条图并返回图像数组（不保存）

        Args:
            image_path: 长条图路径

        Returns:
            单格图像数组列表
        """
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"无法读取图像: {image_path}")
            return []

        # 检查是否为长条图
        if not self.is_strip_image(image):
            return [image]  # 返回原图

        # 检测分割线
        split_lines = self.detect_split_lines(image)

        if len(split_lines) < 2:
            return [image]

        # 分割图像
        panels = []
        for i in range(len(split_lines) - 1):
            y1, y2 = split_lines[i], split_lines[i + 1]

            # 跳过过小的分割
            if y2 - y1 < self.min_gap_size:
                continue

            # 裁剪图像
            panel = image[y1:y2, :]
            panels.append(panel)

        return panels


def split_manga_pages(
    image_paths: List[str],
    output_dir: str,
    min_aspect_ratio: float = 3.0
) -> Tuple[List[str], List[str]]:
    """
    批量分割漫画页面

    Args:
        image_paths: 图像路径列表
        output_dir: 输出目录
        min_aspect_ratio: 判断为长条图的最小高宽比

    Returns:
        (分割后的图像路径列表, 未分割的图像路径列表)
    """
    splitter = StripSplitter(min_aspect_ratio=min_aspect_ratio)

    split_images = []
    original_images = []

    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            continue

        if splitter.is_strip_image(image):
            # 长条图，进行分割
            output_files = splitter.split_strip_image(image_path, output_dir)
            split_images.extend(output_files)
        else:
            # 普通图像，不分割
            original_images.append(image_path)

    return split_images, original_images
