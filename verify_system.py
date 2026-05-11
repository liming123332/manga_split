"""系统验证脚本 - 确认所有组件正常工作"""

import sys
from pathlib import Path


def check_dependencies():
    """检查依赖"""
    print("=" * 60)
    print("1. 检查依赖包")
    print("=" * 60)

    missing = []

    try:
        import ultralytics
        print(f"✓ ultralytics {ultralytics.__version__}")
    except ImportError:
        missing.append("ultralytics")
        print("✗ ultralytics (未安装)")

    try:
        import cv2
        print(f"✓ opencv-python {cv2.__version__}")
    except ImportError:
        missing.append("opencv-python")
        print("✗ opencv-python (未安装)")

    try:
        import torch
        print(f"✓ torch {torch.__version__}")
        print(f"  - CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  - GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        missing.append("torch")
        print("✗ torch (未安装)")

    if missing:
        print(f"\n缺少依赖: {missing}")
        print("请运行: pip install " + " ".join(missing))
        return False

    print("\n所有依赖已安装 ✓")
    return True


def check_model():
    """检查模型文件"""
    print("\n" + "=" * 60)
    print("2. 检查模型文件")
    print("=" * 60)

    model_path = Path("./models/yolov8x6_animeface.pt")
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"✓ YOLO 模型: {model_path}")
        print(f"  大小: {size_mb:.1f} MB")
        return True
    else:
        print(f"✗ 模型文件不存在: {model_path}")
        print("\n请下载 YOLO 模型到 ./models/ 目录")
        print("或使用标准模型（会自动下载）")
        return False


def check_modules():
    """检查模块导入"""
    print("\n" + "=" * 60)
    print("3. 检查模块导入")
    print("=" * 60)

    try:
        from src.detection.yolo_face_detector import YOLOFaceDetector
        print("✓ YOLOFaceDetector")
    except Exception as e:
        print(f"✗ YOLOFaceDetector: {e}")
        return False

    try:
        from src.detection.face_detector_factory import get_detector
        print("✓ FaceDetectorFactory")
    except Exception as e:
        print(f"✗ FaceDetectorFactory: {e}")
        return False

    try:
        from src.pipeline.batch_processor import BatchProcessor
        print("✓ BatchProcessor")
    except Exception as e:
        print(f"✗ BatchProcessor: {e}")
        return False

    try:
        from gui import MangaSplitterGUI
        print("✓ GUI")
    except Exception as e:
        print(f"✗ GUI: {e}")
        return False

    print("\n所有模块导入成功 ✓")
    return True


def test_yolo_detector():
    """测试 YOLO 检测器初始化"""
    print("\n" + "=" * 60)
    print("4. 测试 YOLO 检测器")
    print("=" * 60)

    try:
        from src.detection.yolo_face_detector import YOLOFaceDetector

        print("初始化检测器...")
        detector = YOLOFaceDetector(
            model_path="./models/yolov8x6_animeface.pt",
            confidence_threshold=0.5,
            device="cuda"
        )
        print("✓ 检测器初始化成功")
        print(f"  模型: {detector.model_path}")
        print(f"  置信度阈值: {detector.confidence_threshold}")
        print("\n检测器测试通过 ✓")
        return True
    except Exception as e:
        print(f"✗ 检测器初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("\n条漫人物自动截取系统 - 系统验证")
    print()

    results = []

    # 检查依赖
    results.append(("依赖检查", check_dependencies()))

    # 检查模型
    results.append(("模型检查", check_model()))

    # 检查模块
    results.append(("模块导入", check_modules()))

    # 测试检测器
    results.append(("检测器测试", test_yolo_detector()))

    # 总结
    print("\n" + "=" * 60)
    print("验证总结")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("🎉 所有检查通过！系统已就绪！")
        print()
        print("可以运行以下命令开始使用:")
        print("  python gui.py")
        print()
        return 0
    else:
        print("❌ 部分检查未通过，请修复上述问题")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
