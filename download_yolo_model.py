"""YOLO 二次元人脸检测模型下载辅助脚本"""

import sys
from pathlib import Path


def print_download_instructions():
    """打印模型下载说明"""
    print("=" * 70)
    print("YOLO 二次元人脸检测模型下载指南")
    print("=" * 70)
    print()

    print("方法 1: YOLOv8-AnimeFace (推荐)")
    print("-" * 70)
    print("1. 访问 GitHub 仓库:")
    print("   https://github.com/Fuyucch1/yolov8_animeface")
    print()
    print("2. 从 HuggingFace 下载模型:")
    print("   https://huggingface.co/Fuyucchi/yolov8_animeface")
    print()
    print("3. 下载 .pt 权重文件到 models/ 目录:")
    print("   例如: models/best.pt 或 models/yolov8x_animeface.pt")
    print()

    print("方法 2: YOLOv5-Anime")
    print("-" * 70)
    print("1. 访问 GitHub 仓库:")
    print("   https://github.com/zymk9/yolov5_anime")
    print()
    print("2. 查看 Releases 部分下载预训练权重")
    print()
    print("3. 将 .pt 文件放到 models/ 目录")
    print()

    print("方法 3: 使用标准 YOLOv8 模型")
    print("-" * 70)
    print("如果下载专用模型有困难，可以使用标准 YOLOv8 模型:")
    print("1. 安装 ultralytics: pip install ultralytics")
    print("2. 修改 config.yaml:")
    print("   detection.yolo_model_path: 'yolov8n.pt'")
    print("3. 首次运行时会自动下载官方预训练模型")
    print()

    print("=" * 70)
    print("下载完成后，请更新 config.yaml 配置:")
    print("=" * 70)
    print()
    print("detection:")
    print("  method: 'yolo'")
    print("  yolo_model_path: './models/你的模型文件.pt'")
    print()

    print("然后运行:")
    print("  python gui.py")
    print("  或")
    print("  python main.py process")
    print()


def create_models_directory():
    """创建 models 目录"""
    models_dir = Path("./models")
    models_dir.mkdir(exist_ok=True)
    print(f"✓ 已创建 models 目录: {models_dir.absolute()}")


def check_ultralytics():
    """检查是否安装了 ultralytics"""
    try:
        import ultralytics
        print(f"✓ ultralytics 已安装 (版本: {ultralytics.__version__})")
        return True
    except ImportError:
        print("✗ ultralytics 未安装")
        print()
        print("请运行以下命令安装:")
        print("  pip install ultralytics")
        return False


def main():
    """主函数"""
    print()
    create_models_directory()
    print()

    if not check_ultralytics():
        print()
        print("安装 ultralytics 后，可以使用标准 YOLOv8 模型进行测试")
        print()

    print_download_instructions()

    # 询问是否要测试标准模型
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        print("测试标准 YOLOv8n 模型...")
        print()

        try:
            from ultralytics import YOLO

            # 加载标准模型（会自动下载）
            model = YOLO('yolov8n.pt')
            print("✓ 成功下载并加载 YOLOv8n 标准模型")
            print()
            print("注意：标准模型未专门针对二次元人脸训练，")
            print("检测效果可能不如专用模型。")
            print()

        except Exception as e:
            print(f"✗ 加载模型失败: {e}")


if __name__ == "__main__":
    main()
