"""SAM (Segment Anything Model) 模型封装"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from segment_anything import sam_model_registry, SamPredictor
import urllib.request
import os
from pathlib import Path


logger = logging.getLogger(__name__)


# SAM 模型下载链接
SAM_CHECKPOINT_URLS = {
    'vit_h': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
    'vit_l': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
    'vit_b': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth'
}


class SAMWrapper:
    """Segment Anything Model 包装类"""

    def __init__(
        self,
        model_type: str = "vit_l",
        device: str = "cuda",
        checkpoint_path: Optional[str] = None
    ):
        """
        初始化 SAM 模型

        Args:
            model_type: 模型类型 (vit_h, vit_l, vit_b)
            device: 运行设备 (cuda, cpu, mps)
            checkpoint_path: 模型权重路径，如果为 None 则自动下载
        """
        self.model_type = model_type
        self.device = device

        # 检查 CUDA 可用性
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA 不可用，使用 CPU")
            self.device = "cpu"

        # 获取或下载模型权重
        if checkpoint_path is None:
            checkpoint_path = self._download_model(model_type)

        # 加载模型
        logger.info(f"加载 SAM 模型: {model_type} (设备: {self.device})")
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=self.device)

        # 创建预测器
        self.predictor = SamPredictor(sam)
        logger.info("SAM 模型加载成功")

    def _download_model(self, model_type: str) -> str:
        """
        下载 SAM 模型权重

        Args:
            model_type: 模型类型

        Returns:
            保存的模型路径
        """
        # 创建模型目录
        models_dir = Path("./models")
        models_dir.mkdir(parents=True, exist_ok=True)

        # 模型文件名
        checkpoint_names = {
            'vit_h': 'sam_vit_h_4b8939.pth',
            'vit_l': 'sam_vit_l_0b3195.pth',
            'vit_b': 'sam_vit_b_01ec64.pth'
        }

        checkpoint_name = checkpoint_names[model_type]
        checkpoint_path = models_dir / checkpoint_name

        # 检查是否已存在
        if checkpoint_path.exists():
            logger.info(f"模型文件已存在: {checkpoint_path}")
            return str(checkpoint_path)

        # 下载模型
        url = SAM_CHECKPOINT_URLS[model_type]
        logger.info(f"下载 SAM 模型: {url}")

        try:
            def _download_with_progress(url, destination):
                """带进度条的下载"""
                def _progress(block_num, block_size, total_size):
                    downloaded = block_num * block_size
                    percent = int(downloaded / total_size * 100) if total_size > 0 else 0
                    print(f"\r下载进度: {percent}%", end='')

                urllib.request.urlretrieve(url, destination, _progress)
                print()  # 换行

            _download_with_progress(url, checkpoint_path)
            logger.info(f"模型下载完成: {checkpoint_path}")

        except Exception as e:
            logger.error(f"模型下载失败: {e}")
            raise

        return str(checkpoint_path)

    def set_image(self, image: np.ndarray):
        """
        设置要处理的图像

        Args:
            image: RGB 格式的 numpy 数组
        """
        self.predictor.set_image(image)

    def generate_grid_prompts(
        self,
        image: np.ndarray,
        grid_size: int = 32
    ) -> List[Dict]:
        """
        生成网格提示点

        Args:
            image: 输入图像
            grid_size: 网格间距

        Returns:
            提示点列表
        """
        height, width = image.shape[:2]

        # 生成网格点
        points = []
        for y in range(grid_size // 2, height, grid_size):
            for x in range(grid_size // 2, width, grid_size):
                points.append([x, y])

        # 转换为 numpy 数组
        points_array = np.array(points)

        # 创建提示（所有点都是前景点）
        prompts = [
            {
                'point_coords': points_array,
                'point_labels': np.ones(len(points_array), dtype=np.int32),
                'box': None,
                'mask_input': None
            }
        ]

        return prompts

    def generate_center_prompt(
        self,
        image: np.ndarray
    ) -> Dict:
        """
        生成图像中心点提示

        Args:
            image: 输入图像

        Returns:
            提示字典
        """
        height, width = image.shape[:2]
        center_point = np.array([[width // 2, height // 2]])
        center_label = np.array([1])  # 前景点

        return {
            'point_coords': center_point,
            'point_labels': center_label,
            'box': None,
            'mask_input': None
        }

    def segment_with_prompt(
        self,
        prompt: Dict,
        multimask_output: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        使用提示进行分割

        Args:
            prompt: 提示字典
            multimask_output: 是否输出多个掩码

        Returns:
            (掩码, 质量分数, 低分辨率掩码)
        """
        masks, scores, logits = self.predictor.predict(
            point_coords=prompt['point_coords'],
            point_labels=prompt['point_labels'],
            box=prompt.get('box'),
            mask_input=prompt.get('mask_input'),
            multimask_output=multimask_output
        )

        return masks, scores, logits

    def segment_auto(
        self,
        image: np.ndarray,
        prompt_strategy: str = "grid",
        grid_size: int = 32
    ) -> List[np.ndarray]:
        """
        自动分割图像中的物体

        Args:
            image: RGB 格式的图像
            prompt_strategy: 提示策略 (grid, center)
            grid_size: 网格间距

        Returns:
            掩码列表
        """
        # 设置图像
        self.set_image(image)

        masks_list = []

        if prompt_strategy == "grid":
            # 网格提示策略
            # 使用重叠网格进行采样
            height, width = image.shape[:2]

            # 生成多个网格提示
            grid_prompts = []
            for offset_y in range(0, grid_size, grid_size // 2):
                for offset_x in range(0, grid_size, grid_size // 2):
                    points = []
                    for y in range(offset_y, height, grid_size):
                        for x in range(offset_x, width, grid_size):
                            points.append([x, y])

                    if points:
                        points_array = np.array(points)
                        grid_prompts.append({
                            'point_coords': points_array,
                            'point_labels': np.ones(len(points_array), dtype=np.int32)
                        })

            # 对每个网格提示进行分割
            for prompt in grid_prompts:
                masks, scores, _ = self.segment_with_prompt(prompt, multimask_output=True)

                # 选择最佳掩码
                if len(masks) > 0:
                    best_idx = np.argmax(scores)
                    masks_list.append(masks[best_idx])

        elif prompt_strategy == "center":
            # 中心点提示策略
            prompt = self.generate_center_prompt(image)
            masks, scores, _ = self.segment_with_prompt(prompt, multimask_output=True)

            # 返回所有掩码
            masks_list = [masks[i] for i in range(len(masks))]

        else:
            raise ValueError(f"不支持的提示策略: {prompt_strategy}")

        # 合并和去重掩码
        unique_masks = self._merge_overlapping_masks(masks_list)

        return unique_masks

    def _merge_overlapping_masks(
        self,
        masks: List[np.ndarray],
        iou_threshold: float = 0.8
    ) -> List[np.ndarray]:
        """
        合并重叠的掩码（NMS）

        Args:
            masks: 掩码列表
            iou_threshold: IoU 阈值

        Returns:
            去重后的掩码列表
        """
        if not masks:
            return []

        # 按面积排序（大的优先）
        mask_areas = [np.sum(mask) for mask in masks]
        sorted_indices = np.argsort(mask_areas)[::-1]
        sorted_masks = [masks[i] for i in sorted_indices]

        # NMS
        keep_masks = []
        for mask in sorted_masks:
            # 检查与已保留掩码的重叠
            is_duplicate = False
            for kept_mask in keep_masks:
                iou = self._compute_mask_iou(mask, kept_mask)
                if iou > iou_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                keep_masks.append(mask)

        return keep_masks

    def _compute_mask_iou(
        self,
        mask1: np.ndarray,
        mask2: np.ndarray
    ) -> float:
        """
        计算两个掩码的 IoU

        Args:
            mask1: 掩码 1
            mask2: 掩码 2

        Returns:
            IoU 值
        """
        intersection = np.sum((mask1 > 0) & (mask2 > 0))
        union = np.sum((mask1 > 0) | (mask2 > 0))

        if union == 0:
            return 0.0

        return intersection / union

    def download_model_static(model_type: str = "vit_l", output_dir: str = "./models"):
        """
        静态方法：下载 SAM 模型

        Args:
            model_type: 模型类型
            output_dir: 输出目录
        """
        models_dir = Path(output_dir)
        models_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_names = {
            'vit_h': 'sam_vit_h_4b8939.pth',
            'vit_l': 'sam_vit_l_0b3195.pth',
            'vit_b': 'sam_vit_b_01ec64.pth'
        }

        checkpoint_name = checkpoint_names[model_type]
        checkpoint_path = models_dir / checkpoint_name
        url = SAM_CHECKPOINT_URLS[model_type]

        if checkpoint_path.exists():
            print(f"模型文件已存在: {checkpoint_path}")
            return str(checkpoint_path)

        print(f"下载 SAM 模型: {url}")

        def _progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = int(downloaded / total_size * 100) if total_size > 0 else 0
            print(f"\r下载进度: {percent}%", end='')

        urllib.request.urlretrieve(url, checkpoint_path, _progress)
        print("\n模型下载完成")

        return str(checkpoint_path)
