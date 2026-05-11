"""图像增强预处理模块"""

import cv2
import numpy as np
from typing import Optional, Dict, Any
import logging


logger = logging.getLogger(__name__)


class ImageEnhancer:
    """图像增强器 - 用于提高检测精度和输出质量"""

    def __init__(
        self,
        enable_clahe: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_tile_size: int = 8,
        enable_sharpen: bool = True,
        sharpen_strength: float = 1.0,
        enable_denoise: bool = False,
        denoise_h: float = 10.0,
        enable_super_resolution: bool = False
    ):
        """
        初始化图像增强器

        Args:
            enable_clahe: 是否启用 CLAHE 对比度增强
            clahe_clip_limit: CLAHE 限制对比度阈值 (1.0-3.0)
            clahe_tile_size: CLAHE 网格大小 (4-16)
            enable_sharpen: 是否启用锐化
            sharpen_strength: 锐化强度 (0.5-2.0)
            enable_denoise: 是否启用去噪
            denoise_h: 去噪滤波强度 (5-20)
            enable_super_resolution: 是否启用超分辨率（需要额外模型）
        """
        self.enable_clahe = enable_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_size = clahe_tile_size
        self.enable_sharpen = enable_sharpen
        self.sharpen_strength = sharpen_strength
        self.enable_denoise = enable_denoise
        self.denoise_h = denoise_h
        self.enable_super_resolution = enable_super_resolution

        # 创建 CLAHE 对象
        if self.enable_clahe:
            self.clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip_limit,
                tileGridSize=(self.clahe_tile_size, self.clahe_tile_size)
            )

        logger.info(
            f"图像增强器初始化: CLAHE={enable_clahe}, "
            f"锐化={enable_sharpen}, 去噪={enable_denoise}"
        )

    def enhance_for_detection(self, image: np.ndarray) -> np.ndarray:
        """
        为 AI 检测优化图像（增强对比度、锐化边缘）

        注意：这个增强只在检测时使用，不影响最终输出

        Args:
            image: BGR 输入图像

        Returns:
            增强后的 BGR 图像
        """
        enhanced = image.copy()

        # 1. 转换到 LAB 色彩空间进行对比度增强（只处理 L 通道）
        if self.enable_clahe:
            try:
                lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)

                # 应用 CLAHE 到 L 通道
                l = self.clahe.apply(l)

                # 合并通道
                lab = cv2.merge([l, a, b])
                enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            except Exception as e:
                logger.warning(f"CLAHE 增强失败: {e}")

        # 2. 锐化（提高边缘清晰度，帮助 SAM 检测）
        if self.enable_sharpen:
            try:
                # 创建锐化核
                kernel = np.array([
                    [-1, -1, -1],
                    [-1, 9, -1],
                    [-1, -1, -1]
                ]) * self.sharpen_strength

                # 调整中心值以保持亮度
                kernel[1, 1] = 1 + 8 * self.sharpen_strength

                # 应用锐化
                enhanced = cv2.filter2D(enhanced, -1, kernel)
            except Exception as e:
                logger.warning(f"锐化失败: {e}")

        # 3. 轻微去噪（减少 JPEG 压缩噪声）
        if self.enable_denoise:
            try:
                enhanced = cv2.fastNlMeansDenoisingColored(
                    enhanced,
                    None,
                    self.denoise_h,
                    self.denoise_h,
                    7,
                    21
                )
            except Exception as e:
                logger.warning(f"去噪失败: {e}")

        return enhanced

    def enhance_for_output(self, image: np.ndarray) -> np.ndarray:
        """
        为最终输出优化图像（提升训练数据质量）

        Args:
            image: BGR 输入图像

        Returns:
            增强后的 BGR 图像
        """
        enhanced = image.copy()

        # 输出增强通常更保守，只做轻微的对比度调整
        if self.enable_clahe:
            try:
                # 降低强度以保持自然外观
                lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)

                # 使用更保守的参数
                clahe_output = cv2.createCLAHE(
                    clipLimit=self.clahe_clip_limit * 0.5,
                    tileGridSize=(self.clahe_tile_size * 2, self.clahe_tile_size * 2)
                )
                l = clahe_output.apply(l)

                lab = cv2.merge([l, a, b])
                enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            except Exception as e:
                logger.warning(f"输出 CLAHE 增强失败: {e}")

        return enhanced

    def enhance_with_config(self, image: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """
        使用自定义配置进行增强

        Args:
            image: 输入图像
            config: 配置字典

        Returns:
            增强后的图像
        """
        mode = config.get('mode', 'detection')  # 'detection' or 'output'

        if mode == 'detection':
            return self.enhance_for_detection(image)
        elif mode == 'output':
            return self.enhance_for_output(image)
        else:
            logger.warning(f"未知的增强模式: {mode}")
            return image


def create_enhancer_from_config(config: Dict[str, Any]) -> ImageEnhancer:
    """
    从配置字典创建图像增强器

    Args:
        config: 配置字典

    Returns:
        ImageEnhancer 实例
    """
    return ImageEnhancer(
        enable_clahe=config.get('enable_clahe', True),
        clahe_clip_limit=config.get('clahe_clip_limit', 2.0),
        clahe_tile_size=config.get('clahe_tile_size', 8),
        enable_sharpen=config.get('enable_sharpen', True),
        sharpen_strength=config.get('sharpen_strength', 1.0),
        enable_denoise=config.get('enable_denoise', False),
        denoise_h=config.get('denoise_h', 10.0),
        enable_super_resolution=config.get('enable_super_resolution', False)
    )
