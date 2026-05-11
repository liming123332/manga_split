"""条漫人物自动截取系统 - 主入口"""

import argparse
import logging
import sys
from pathlib import Path

from src.utils.config import Config
from src.pipeline.batch_processor import BatchProcessor, create_progress_callback
from src.detection.sam_wrapper import SAMWrapper


def setup_logging(level: str = "INFO", log_file: str = None):
    """
    设置日志

    Args:
        level: 日志级别
        log_file: 日志文件路径
    """
    # 创建日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 配置根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # 文件处理器
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def download_model(args):
    """
    下载 SAM 模型

    Args:
        args: 命令行参数
    """
    model_type = args.model_type or 'vit_l'
    output_dir = args.output_dir or './models'

    print(f"下载 SAM 模型: {model_type}")
    print(f"保存到: {output_dir}")

    try:
        checkpoint_path = SAMWrapper.download_model_static(model_type, output_dir)
        print(f"✓ 模型下载成功: {checkpoint_path}")
        return 0
    except Exception as e:
        print(f"✗ 模型下载失败: {e}")
        return 1


def process_manga(args):
    """
    处理漫画图像

    Args:
        args: 命令行参数
    """
    # 加载配置
    config_file = args.config
    if config_file:
        config = Config(config_file)
    else:
        # 使用默认配置
        config = Config("config.yaml")

    # 命令行参数覆盖配置文件
    if args.input:
        config.set('input.directory', args.input)

    if args.output:
        config.set('output.directory', args.output)

    if args.model:
        config.set('sam.model_type', args.model)

    if args.device:
        config.set('sam.device', args.device)

    # 设置日志
    log_level = config.get('logging.level', 'INFO')
    log_file = config.get('logging.log_file') if config.get('logging.save_log') else None
    setup_logging(log_level, log_file)

    logger = logging.getLogger(__name__)

    # 创建进度回调
    progress_callback = None
    if not args.quiet:
        progress_callback = create_progress_callback("处理进度")

    try:
        # 初始化批量处理器
        processor = BatchProcessor(
            config=config,
            progress_callback=progress_callback
        )

        # 执行批量处理
        report = processor.process_directory()

        # 输出摘要
        print("\n" + "=" * 50)
        print("处理完成")
        print("=" * 50)
        print(f"总图像数: {report['summary']['total_images']}")
        print(f"成功处理: {report['summary']['processed_images']}")
        print(f"长条图分割: {report['summary']['split_strips']}")
        print(f"检测人物: {report['summary']['detected_persons']}")
        print(f"保存图像: {report['summary']['saved_images']}")
        print(f"失败图像: {report['summary']['failed_images']}")
        print(f"成功率: {report['summary']['success_rate']:.1%}")

        if report['summary']['failed_images'] > 0:
            print(f"\n警告: {report['summary']['failed_images']} 个图像处理失败")
            print(f"详细信息请查看: {Path(config.output_dir) / 'report.json'}")

        print("=" * 50)

        return 0

    except Exception as e:
        logger.error(f"处理失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='条漫人物自动截取系统 - 用于 SDXL LoRA 训练数据准备',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例用法:

  # 1. 下载 SAM 模型
  python main.py --download-sam-model vit_l

  # 2. 使用配置文件处理
  python main.py --process

  # 3. 指定输入输出目录
  python main.py --process --input "D:/漫画/作品名" --output "./output"

  # 4. 使用 CPU 模式
  python main.py --process --device cpu

  # 5. 使用快速模型
  python main.py --process --model vit_b
        '''
    )

    # 子命令
    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # 下载模型命令
    download_parser = subparsers.add_parser(
        'download-model',
        help='下载 SAM 模型'
    )
    download_parser.add_argument(
        '--model-type',
        choices=['vit_h', 'vit_l', 'vit_b'],
        default='vit_l',
        help='模型类型 (默认: vit_l)'
    )
    download_parser.add_argument(
        '--output-dir',
        default='./models',
        help='模型保存目录 (默认: ./models)'
    )

    # 处理漫画命令
    process_parser = subparsers.add_parser(
        'process',
        help='处理漫画图像'
    )
    process_parser.add_argument(
        '--config',
        default='config.yaml',
        help='配置文件路径 (默认: config.yaml)'
    )
    process_parser.add_argument(
        '--input',
        help='输入目录（覆盖配置文件）'
    )
    process_parser.add_argument(
        '--output',
        help='输出目录（覆盖配置文件）'
    )
    process_parser.add_argument(
        '--model',
        choices=['vit_h', 'vit_l', 'vit_b'],
        help='SAM 模型类型 (覆盖配置文件)'
    )
    process_parser.add_argument(
        '--device',
        choices=['cuda', 'cpu', 'mps'],
        help='运行设备 (覆盖配置文件)'
    )
    process_parser.add_argument(
        '--quiet',
        action='store_true',
        help='静默模式（不显示进度）'
    )

    # 为了向后兼容，保留旧的参数格式
    parser.add_argument(
        '--download-sam-model',
        choices=['vit_h', 'vit_l', 'vit_b'],
        help='下载 SAM 模型（旧参数格式，建议使用 download-model 子命令）'
    )
    parser.add_argument(
        '--process',
        action='store_true',
        help='处理漫画图像（旧参数格式，建议使用 process 子命令）'
    )

    args = parser.parse_args()

    # 处理旧参数格式
    if hasattr(args, 'download_sam_model') and args.download_sam_model:
        # 创建下载模型参数
        class DownloadArgs:
            model_type = args.download_sam_model
            output_dir = './models'
        return download_model(DownloadArgs())

    if hasattr(args, 'process') and args.process:
        return process_manga(args)

    # 处理新参数格式（子命令）
    if args.command == 'download-model':
        return download_model(args)
    elif args.command == 'process':
        return process_manga(args)
    else:
        # 没有指定命令，显示帮助
        parser.print_help()
        return 0


if __name__ == '__main__':
    sys.exit(main())
