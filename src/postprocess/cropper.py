"""白边裁剪模块"""

import cv2
import numpy as np
from typing import Tuple, Optional
import logging


logger = logging.getLogger(__name__)


def crop_transparent_borders(
    image_rgba: np.ndarray,
    padding: int = 10,
    min_size: Optional[int] = None
) -> np.ndarray:
    """
    裁剪透明边框

    Args:
        image_rgba: RGBA 格式的图像
        padding: 裁剪后保留的边距（像素）
        min_size: 最小输出尺寸，避免过度裁剪

    Returns:
        裁剪后的 RGBA 图像
    """
    if image_rgba is None or image_rgba.size == 0:
        return image_rgba

    # 提取 Alpha 通道
    if image_rgba.shape[2] != 4:
        logger.warning("图像不是 RGBA 格式，无法裁剪透明边框")
        return image_rgba

    alpha = image_rgba[:, :, 3]

    # 查找非透明边界
    rows = np.any(alpha > 0, axis=1)
    cols = np.any(alpha > 0, axis=0)

    if not np.any(rows) or not np.any(cols):
        logger.warning("图像完全透明，无法裁剪")
        return image_rgba

    # 获取边界
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    # 添加边距
    height, width = image_rgba.shape[:2]
    y_min = max(0, y_min - padding)
    y_max = min(height, y_max + padding + 1)
    x_min = max(0, x_min - padding)
    x_max = min(width, x_max + padding + 1)

    # 应用最小尺寸限制
    if min_size is not None:
        crop_height = y_max - y_min
        crop_width = x_max - x_min

        if crop_height < min_size:
            # 垂直方向扩展
            diff = min_size - crop_height
            y_min = max(0, y_min - diff // 2)
            y_max = min(height, y_max + (diff - diff // 2))

        if crop_width < min_size:
            # 水平方向扩展
            diff = min_size - crop_width
            x_min = max(0, x_min - diff // 2)
            x_max = min(width, x_max + (diff - diff // 2))

    # 裁剪图像
    cropped = image_rgba[y_min:y_max, x_min:x_max].copy()

    return cropped


def crop_white_borders(
    image: np.ndarray,
    threshold: int = 250,
    padding: int = 10
) -> np.ndarray:
    """
    裁剪白色边框（用于非透明图像）

    Args:
        image: BGR 或灰度图像
        threshold: 白色阈值（0-255）
        padding: 裁剪后保留的边距

    Returns:
        裁剪后的图像
    """
    if image is None or image.size == 0:
        return image

    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 创建白色区域的掩码
    white_mask = gray > threshold

    # 反转掩码（查找非白色区域）
    content_mask = ~white_mask

    # 查找内容边界
    rows = np.any(content_mask, axis=1)
    cols = np.any(content_mask, axis=0)

    if not np.any(rows) or not np.any(cols):
        logger.warning("图像全为白色，无法裁剪")
        return image

    # 获取边界
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    # 添加边距
    height, width = image.shape[:2]
    y_min = max(0, y_min - padding)
    y_max = min(height, y_max + padding + 1)
    x_min = max(0, x_min - padding)
    x_max = min(width, x_max + padding + 1)

    # 裁剪图像
    cropped = image[y_min:y_max, x_min:x_max].copy()

    return cropped


def resize_with_aspect_ratio(
    image: np.ndarray,
    max_size: int = 2048,
    min_size: int = 512,
    interpolation: int = cv2.INTER_AREA
) -> np.ndarray:
    """
    保持宽高比调整图像大小

    Args:
        image: 输入图像
        max_size: 最大尺寸
        min_size: 最小尺寸
        interpolation: 插值方法

    Returns:
        调整后的图像
    """
    if image is None or image.size == 0:
        return image

    height, width = image.shape[:2]

    # 检查是否需要调整
    max_dim = max(height, width)
    min_dim = min(height, width)

    if max_dim <= max_size and min_dim >= min_size:
        return image

    # 计算缩放比例
    if max_dim > max_size:
        scale = max_size / max_dim
    else:
        scale = min_size / min_dim

    # 调整大小
    new_width = int(width * scale)
    new_height = int(height * scale)

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=interpolation
    )

    return resized


def center_crop(
    image: np.ndarray,
    target_size: int
) -> np.ndarray:
    """
    中心裁剪图像到指定大小

    Args:
        image: 输入图像
        target_size: 目标尺寸（正方形）

    Returns:
        裁剪后的图像
    """
    if image is None or image.size == 0:
        return image

    height, width = image.shape[:2]

    # 如果图像已经足够小，直接返回
    if height <= target_size and width <= target_size:
        return image

    # 计算裁剪区域
    y_start = (height - target_size) // 2
    x_start = (width - target_size) // 2

    # 确保不超出边界
    y_start = max(0, y_start)
    x_start = max(0, x_start)

    cropped = image[
        y_start:y_start + target_size,
        x_start:x_start + target_size
    ]

    return cropped


def pad_to_square(
    image_rgba: np.ndarray,
    pad_color: Tuple[int, int, int, int] = (255, 255, 255, 0)
) -> np.ndarray:
    """
    填充图像为正方形（透明背景）

    Args:
        image_rgba: RGBA 图像
        pad_color: 填充颜色 (R, G, B, A)

    Returns:
        填充后的正方形图像
    """
    if image_rgba is None or image_rgba.size == 0:
        return image_rgba

    height, width = image_rgba.shape[:2]

    if height == width:
        return image_rgba

    # 计算目标尺寸
    max_size = max(height, width)

    # 创建填充后的图像
    padded = np.full(
        (max_size, max_size, 4),
        pad_color,
        dtype=image_rgba.dtype
    )

    # 计算粘贴位置
    y_start = (max_size - height) // 2
    x_start = (max_size - width) // 2

    # 粘贴原图
    padded[y_start:y_start + height, x_start:x_start + width] = image_rgba

    return padded


def auto_crop(
    image_rgba: np.ndarray,
    padding: int = 10,
    min_size: int = 512,
    max_size: int = 2048,
    to_square: bool = False
) -> np.ndarray:
    """
    自动裁剪和调整图像

    Args:
        image_rgba: RGBA 图像
        padding: 裁剪边距
        min_size: 最小尺寸
        max_size: 最大尺寸
        to_square: 是否填充为正方形

    Returns:
        处理后的图像
    """
    if image_rgba is None or image_rgba.size == 0:
        return image_rgba

    # 裁剪透明边框
    cropped = crop_transparent_borders(image_rgba, padding=padding)

    # 调整大小
    resized = resize_with_aspect_ratio(
        cropped,
        max_size=max_size,
        min_size=min_size
    )

    # 可选：填充为正方形
    if to_square:
        resized = pad_to_square(resized)

    return resized


def get_content_bbox(
    image_rgba: np.ndarray,
    alpha_threshold: int = 10
) -> Tuple[int, int, int, int]:
    """
    获取图像内容（非透明区域）的边界框

    Args:
        image_rgba: RGBA 图像
        alpha_threshold: Alpha 阈值

    Returns:
        (x1, y1, x2, y2) 边界框
    """
    if image_rgba is None or image_rgba.size == 0:
        return (0, 0, 0, 0)

    # 提取 Alpha 通道
    alpha = image_rgba[:, :, 3]

    # 查找非透明区域
    content_mask = alpha > alpha_threshold

    rows = np.any(content_mask, axis=1)
    cols = np.any(content_mask, axis=0)

    if not np.any(rows) or not np.any(cols):
        return (0, 0, 0, 0)

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    return (int(x_min), int(y_min), int(x_max), int(y_max))


def calculate_crop_margin(
    image_rgba: np.ndarray,
    target_margin_ratio: float = 0.1
) -> int:
    """
    计算合适的裁剪边距

    Args:
        image_rgba: RGBA 图像
        target_margin_ratio: 目标边距比例（相对于内容尺寸）

    Returns:
        推荐的边距（像素）
    """
    if image_rgba is None or image_rgba.size == 0:
        return 10

    # 获取内容边界框
    x1, y1, x2, y2 = get_content_bbox(image_rgba)

    content_width = x2 - x1
    content_height = y2 - y1

    # 计算边距
    margin = int(min(content_width, content_height) * target_margin_ratio)

    # 限制边距范围
    margin = max(5, min(margin, 50))

    return margin
