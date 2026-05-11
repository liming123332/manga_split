"""批量处理工作流模块"""

import os
import cv2
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Optional, Callable
from datetime import datetime
import logging

from ..utils.config import Config
from ..preprocess.strip_splitter import StripSplitter
from ..preprocess.image_enhancer import ImageEnhancer
from ..detection.face_detector_factory import get_detector
from ..detection.sam_wrapper import SAMWrapper


logger = logging.getLogger(__name__)


class BatchProcessor:
    """批量处理器"""

    def __init__(
        self,
        config: Config,
        progress_callback: Optional[Callable] = None
    ):
        """
        初始化批量处理器

        Args:
            config: 配置对象
            progress_callback: 进度回调函数
        """
        self.config = config

        # 获取检测方法
        detection_method = config.get('detection.method', 'yolo').lower()
        logger.info(f"检测方法: {detection_method}")

        # 初始化 SAM 模型（如果需要）
        self.sam_wrapper = None
        if detection_method == 'sam':
            logger.info("初始化 SAM 模型...")
            self.sam_wrapper = SAMWrapper(
                model_type=config.sam_model_type,
                device=config.sam_device,
                checkpoint_path=config.sam_checkpoint
            )

        # 初始化图像增强器
        if config.get('preprocess.enable_enhancement', False):
            logger.info("初始化图像增强器...")
            self.image_enhancer = ImageEnhancer(
                enable_clahe=config.get('preprocess.clahe_enabled', True),
                clahe_clip_limit=config.get('preprocess.clahe_clip_limit', 2.0),
                clahe_tile_size=config.get('preprocess.clahe_tile_size', 8),
                enable_sharpen=config.get('preprocess.sharpen_enabled', True),
                sharpen_strength=config.get('preprocess.sharpen_strength', 1.0),
                enable_denoise=config.get('preprocess.denoise_enabled', False),
                denoise_h=config.get('preprocess.denoise_strength', 10.0)
            )
        else:
            self.image_enhancer = None

        # 使用工厂创建检测器
        logger.info("初始化人脸检测器...")
        self.detector = get_detector(
            method=detection_method,
            sam_wrapper=self.sam_wrapper,
            image_enhancer=self.image_enhancer,
            # YOLO 参数
            model_path=config.get('detection.yolo_model_path', './models/yolov8n_animeface.pt'),
            confidence_threshold=config.get('detection.confidence_threshold', 0.5),
            iou_threshold=config.get('detection.iou_threshold', 0.45),
            max_detections=config.get('detection.max_persons_per_image', 10),
            device=config.sam_device,  # YOLO 也使用相同设备
            # SAM 参数
            face_crop_ratio=config.get('detection.face_crop_ratio', 0.5),
            face_center_crop=config.get('detection.face_center_crop', 0.6),
            min_person_size=config.get('detection.min_person_size', 256)
        )

        # 初始化长条图分割器
        self.strip_splitter = StripSplitter(
            min_aspect_ratio=config.get('strip_split.min_aspect_ratio', 3.0),
            min_gap_size=config.get('strip_split.min_gap_size', 20)
        )

        self.progress_callback = progress_callback

        # 统计信息
        self.stats = {
            'total_images': 0,
            'processed_images': 0,
            'split_strips': 0,
            'detected_persons': 0,
            'saved_images': 0,
            'failed_images': 0,
            'errors': []
        }

    def _scan_images(self, input_dir: str, recursive: bool = True) -> List[str]:
        """
        扫描目录中的图像文件

        Args:
            input_dir: 输入目录
            recursive: 是否递归扫描

        Returns:
            图像文件路径列表
        """
        input_path = Path(input_dir)
        supported_formats = self.config.supported_formats

        image_files = []

        if recursive:
            # 递归扫描
            for fmt in supported_formats:
                image_files.extend(input_path.rglob(f"*{fmt}"))
        else:
            # 仅扫描当前目录
            for fmt in supported_formats:
                image_files.extend(input_path.glob(f"*{fmt}"))

        # 转换为字符串路径
        image_paths = [str(f) for f in image_files]

        logger.info(f"扫描到 {len(image_paths)} 个图像文件")
        return sorted(image_paths)

    def _process_single_image(
        self,
        image_path: str,
        output_dir: str
    ) -> List[Dict]:
        """
        处理单个图像文件

        Args:
            image_path: 图像路径
            output_dir: 输出目录

        Returns:
            保存的图像元数据列表
        """
        metadata_list = []

        try:
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"无法读取图像: {image_path}")
                self.stats['failed_images'] += 1
                return metadata_list

            # 检查是否为长条图
            is_strip = self.strip_splitter.is_strip_image(image)

            if is_strip and self.config.get('strip_split.enabled', True):
                # 长条图分割
                logger.info(f"处理长条图: {image_path}")
                panels = self.strip_splitter.split_and_return_arrays(image_path)
                self.stats['split_strips'] += 1

                # 处理每个分割后的单格
                for idx, panel in enumerate(panels):
                    panel_metadata = self._process_panel(
                        panel,
                        image_path,
                        output_dir,
                        panel_idx=idx
                    )
                    metadata_list.extend(panel_metadata)
            else:
                # 普通图像
                logger.debug(f"处理普通图像: {image_path}")
                panel_metadata = self._process_panel(
                    image,
                    image_path,
                    output_dir
                )
                metadata_list.extend(panel_metadata)

            self.stats['processed_images'] += 1

        except Exception as e:
            logger.error(f"处理图像失败 {image_path}: {e}")
            self.stats['failed_images'] += 1
            self.stats['errors'].append({
                'image': image_path,
                'error': str(e)
            })

        return metadata_list

    def _process_panel(
        self,
        image: np.ndarray,
        source_path: str,
        output_dir: str,
        panel_idx: int = 0
    ) -> List[Dict]:
        """
        处理单个漫画格子

        Args:
            image: 图像数组
            source_path: 源图像路径
            output_dir: 输出目录
            panel_idx: 格子索引

        Returns:
            保存的图像元数据列表
        """
        metadata_list = []

        try:
            # 检测人脸（传递裁剪模式参数）
            crop_mode = self.config.get('detection.crop_mode', 'upper_body')
            crop_padding_ratio = self.config.get('detection.crop_padding_ratio', 0.5)

            extracted = self.detector.extract_face_images(
                image,
                padding=20,  # 基础边距
                crop_mode=crop_mode,
                crop_padding_ratio=crop_padding_ratio
            )

            if not extracted:
                logger.debug(f"未检测到人脸: {source_path}")
                return metadata_list

            self.stats['detected_persons'] += len(extracted)

            # 处理每个检测到的人脸
            for idx, (face_image, face_metadata) in enumerate(extracted):
                # 使用检测到的人脸图像（已经包含边距）
                output_image = face_image.copy()

                # 可选：应用输出图像增强
                if self.image_enhancer is not None and self.config.get('preprocess.enhance_output', False):
                    output_image = self.image_enhancer.enhance_for_output(output_image)
                    logger.debug("已应用输出图像增强")

                # 可选：调整大小（保持宽高比）
                max_size = self.config.get('postprocess.max_output_size', 2048)
                min_size = self.config.get('postprocess.min_output_size', 512)
                height, width = output_image.shape[:2]

                # 只在图像过大时缩放
                if max(height, width) > max_size:
                    scale = max_size / max(height, width)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    output_image = cv2.resize(output_image, (new_width, new_height), interpolation=cv2.INTER_AREA)

                # 生成文件名（使用 jpg 格式）
                source_name = Path(source_path).stem
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{source_name}_p{panel_idx:02d}_face{idx:02d}_{timestamp}.jpg"

                # 保存图像（BGR 格式，带背景的人脸区域）
                output_path = Path(output_dir) / "images" / filename
                cv2.imwrite(
                    str(output_path),
                    output_image,
                    [int(cv2.IMWRITE_JPEG_QUALITY), self.config.get('output.quality', 95)]
                )

                # 创建元数据
                metadata = {
                    'filename': filename,
                    'source_path': source_path,
                    'panel_idx': panel_idx,
                    'face_idx': idx,
                    'output_path': str(output_path),
                    'image_shape': output_image.shape,
                    'format': 'jpg',
                    'content_type': 'anime_face',
                    'transparent_background': False,
                    'detection_metadata': face_metadata,
                    'timestamp': datetime.now().isoformat()
                }

                metadata_list.append(metadata)
                self.stats['saved_images'] += 1

                logger.debug(f"保存人物图像: {filename}")

        except Exception as e:
            logger.error(f"处理格子失败: {e}")
            self.stats['errors'].append({
                'image': source_path,
                'error': str(e)
            })

        return metadata_list

    def process_directory(
        self,
        input_dir: Optional[str] = None,
        output_dir: Optional[str] = None
    ) -> Dict:
        """
        批量处理目录下的所有漫画图像

        Args:
            input_dir: 输入目录（如果为 None，使用配置中的路径）
            output_dir: 输出目录（如果为 None，使用配置中的路径）

        Returns:
            处理报告字典
        """
        # 使用配置中的路径（如果未指定）
        input_dir = input_dir or self.config.input_dir
        output_dir = output_dir or self.config.output_dir

        # 验证输入目录
        if not input_dir:
            raise ValueError("输入目录未设置")

        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"输入目录不存在: {input_dir}")

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'metadata'), exist_ok=True)

        logger.info(f"开始批量处理")
        logger.info(f"输入目录: {input_dir}")
        logger.info(f"输出目录: {output_dir}")

        # 扫描图像文件
        image_files = self._scan_images(input_dir, self.config.input_recursive)
        self.stats['total_images'] = len(image_files)

        if not image_files:
            logger.warning("未找到图像文件")
            return self._generate_report()

        # 处理每个图像
        all_metadata = []
        for idx, image_path in enumerate(image_files):
            logger.info(f"处理进度: {idx + 1}/{len(image_files)} - {image_path}")

            # 处理图像
            metadata_list = self._process_single_image(image_path, output_dir)
            all_metadata.extend(metadata_list)

            # 进度回调
            if self.progress_callback:
                self.progress_callback(idx + 1, len(image_files), image_path)

        # 保存元数据
        if self.config.get('output.save_metadata', True):
            self._save_metadata(all_metadata, output_dir)

        # 生成报告
        report = self._generate_report()
        self._save_report(report, output_dir)

        logger.info("批量处理完成")
        logger.info(f"处理图像: {self.stats['processed_images']}/{self.stats['total_images']}")
        logger.info(f"检测人物: {self.stats['detected_persons']}")
        logger.info(f"保存图像: {self.stats['saved_images']}")
        logger.info(f"失败图像: {self.stats['failed_images']}")

        return report

    def _save_metadata(self, metadata_list: List[Dict], output_dir: str):
        """
        保存元数据到 JSON 文件

        Args:
            metadata_list: 元数据列表
            output_dir: 输出目录
        """
        metadata_path = Path(output_dir) / "metadata" / "metadata.json"

        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_list, f, indent=2, ensure_ascii=False)

        logger.info(f"保存元数据: {metadata_path}")

    def _generate_report(self) -> Dict:
        """
        生成处理报告

        Returns:
            报告字典
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_images': self.stats['total_images'],
                'processed_images': self.stats['processed_images'],
                'split_strips': self.stats['split_strips'],
                'detected_persons': self.stats['detected_persons'],
                'saved_images': self.stats['saved_images'],
                'failed_images': self.stats['failed_images'],
                'success_rate': self.stats['processed_images'] / max(self.stats['total_images'], 1)
            },
            'errors': self.stats['errors']
        }

        return report

    def _save_report(self, report: Dict, output_dir: str):
        """
        保存报告到 JSON 文件

        Args:
            report: 报告字典
            output_dir: 输出目录
        """
        report_path = Path(output_dir) / "report.json"

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"保存报告: {report_path}")


def create_progress_callback(description: str = "处理进度") -> Callable:
    """
    创建进度条回调函数

    Args:
        description: 进度描述

    Returns:
        回调函数
    """
    def callback(current: int, total: int, message: str = ""):
        percent = current / total * 100
        print(f"\r{description}: {current}/{total} ({percent:.1f}%) - {message}", end='')

        if current == total:
            print()  # 完成后换行

    return callback
