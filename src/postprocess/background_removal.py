"""背景透明化模块"""

import cv2
import numpy as np
from typing import Tuple
import logging


logger = logging.getLogger(__name__)


def apply_mask_with_alpha(
    image: np.ndarray,
    mask: np.ndarray,
    blur_radius: int = 3
) -> np.ndarray:
    """
    应用掩码并生成透明背景的 RGBA 图像

    Args:
        image: BGR 格式的图像
        mask: 二值掩码（0 或 255）
        blur_radius: 边缘羽化半径

    Returns:
        RGBA 格式的图像（带 Alpha 通道）
    """
    # 确保掩码是二值的
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255

    # 边缘羽化处理
    if blur_radius > 0:
        # 使用高斯模糊对掩码边缘进行羽化
        mask_blurred = cv2.GaussianBlur(
            mask,
            (blur_radius * 2 + 1, blur_radius * 2 + 1),
            0
        )
        # 归一化到 0-1 范围
        alpha = mask_blurred.astype(np.float32) / 255.0
    else:
        alpha = mask.astype(np.float32) / 255.0

    # 确保图像是 BGR 格式
    if len(image.shape) == 2:
        # 灰度图，转换为 BGR
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_bgr = image.copy()

    # 分离 BGR 通道
    b, g, r = cv2.split(image_bgr)

    # 合并为 RGBA
    rgba_image = cv2.merge([
        b.astype(np.uint8),
        g.astype(np.uint8),
        r.astype(np.uint8),
        (alpha * 255).astype(np.uint8)
    ])

    return rgba_image


def refine_mask_edge(
    mask: np.ndarray,
    image: np.ndarray,
    iterations: int = 2
) -> np.ndarray:
    """
    基于图像内容优化掩码边缘

    Args:
        mask: 原始掩码
        image: 原始图像
        iterations: 迭代次数

    Returns:
        优化后的掩码
    """
    # 使用 GrabCut 算法优化边缘
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255

    # 确保 image 是 BGR 格式
    if len(image.shape) == 2:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_bgr = image

    # 创建 GrabCut 的初始掩码
    grabcut_mask = np.zeros(mask.shape, np.uint8)
    grabcut_mask[mask > 0] = 3  # 可能的前景
    grabcut_mask[mask == 0] = 0  # 肯定的背景

    # 创建前景和背景模型
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        # 运行 GrabCut
        cv2.grabCut(
            image_bgr,
            grabcut_mask,
            None,
            bgd_model,
            fgd_model,
            iterations,
            cv2.GC_INIT_WITH_MASK
        )

        # 提取优化后的掩码
        refined_mask = np.where(
            (grabcut_mask == 1) | (grabcut_mask == 3),
            255,
            0
        ).astype(np.uint8)

        return refined_mask

    except cv2.error as e:
        logger.warning(f"GrabCut 失败，使用原始掩码: {e}")
        return mask


def mask_to_contour(
    mask: np.ndarray,
    simplify: bool = True
) -> np.ndarray:
    """
    将掩码转换为轮廓

    Args:
        mask: 二值掩码
        simplify: 是否简化轮廓

    Returns:
        轮廓点集
    """
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255

    # 查找轮廓
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE if simplify else cv2.CHAIN_APPROX_NONE
    )

    if not contours:
        return np.array([])

    # 合并所有轮廓
    all_contours = np.vstack(contours)

    return all_contours


def blend_mask_edge(
    image: np.ndarray,
    mask: np.ndarray,
    feather_radius: int = 5
) -> np.ndarray:
    """
    在图像上混合掩码边缘（用于预览）

    Args:
        image: 原始图像
        mask: 掩码
        feather_radius: 羽化半径

    Returns:
        带边缘高亮的图像
    """
    # 确保掩码是二值的
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255

    # 复制图像
    result = image.copy()

    # 查找轮廓
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # 绘制轮廓
    cv2.drawContours(
        result,
        contours,
        -1,
        (0, 255, 0),  # 绿色
        2
    )

    return result


def create_thumbnail(
    image: np.ndarray,
    max_size: int = 256
) -> np.ndarray:
    """
    创建缩略图（用于预览）

    Args:
        image: 输入图像
        max_size: 最大尺寸

    Returns:
        缩略图
    """
    height, width = image.shape[:2]
    scale = min(max_size / width, max_size / height, 1.0)

    if scale < 1.0:
        new_width = int(width * scale)
        new_height = int(height * scale)
        thumbnail = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    else:
        thumbnail = image.copy()

    return thumbnail


def validate_alpha_channel(rgba_image: np.ndarray) -> bool:
    """
    验证 RGBA 图像的 Alpha 通道是否有效

    Args:
        rgba_image: RGBA 图像

    Returns:
        是否有效
    """
    if rgba_image.shape[2] != 4:
        return False

    alpha = rgba_image[:, :, 3]

    # 检查 Alpha 通道是否有变化
    if np.all(alpha == alpha[0, 0]):
        return False

    # 检查是否有透明区域
    if np.all(alpha == 255) or np.all(alpha == 0):
        return False

    return True


def get_image_stats(rgba_image: np.ndarray) -> dict:
    """
    获取 RGBA 图像的统计信息

    Args:
        rgba_image: RGBA 图像

    Returns:
        统计信息字典
    """
    alpha = rgba_image[:, :, 3]

    stats = {
        'total_pixels': alpha.size,
        'opaque_pixels': np.sum(alpha == 255),
        'transparent_pixels': np.sum(alpha == 0),
        'semi_transparent_pixels': np.sum((alpha > 0) & (alpha < 255)),
        'avg_alpha': float(np.mean(alpha) / 255.0),
        'min_alpha': int(np.min(alpha)),
        'max_alpha': int(np.max(alpha))
    }

    # 计算透明占比
    stats['transparency_ratio'] = stats['transparent_pixels'] / stats['total_pixels']
    stats['content_ratio'] = (stats['opaque_pixels'] + stats['semi_transparent_pixels']) / stats['total_pixels']

    return stats
