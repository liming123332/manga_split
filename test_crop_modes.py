"""测试不同的裁剪模式效果"""

import cv2
import sys
from pathlib import Path

from src.detection.yolo_face_detector import YOLOFaceDetector


def test_crop_modes(image_path: str):
    """
    测试三种裁剪模式

    Args:
        image_path: 测试图像路径
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误：无法读取图像 {image_path}")
        return

    print(f"测试图像: {image_path}")
    print(f"图像尺寸: {image.shape[1]}x{image.shape[0]}")
    print()

    # 创建输出目录
    output_dir = Path("./crop_mode_test")
    output_dir.mkdir(exist_ok=True)

    # 测试三种模式
    modes = [
        ("face", "仅人脸"),
        ("upper_body", "上半身"),
        ("full_body", "全身")
    ]

    try:
        # 初始化检测器
        print("初始化 YOLO 检测器...")
        detector = YOLOFaceDetector(
            model_path="./models/yolov8x6_animeface.pt",
            confidence_threshold=0.5,
            device="cuda"
        )
        print("✓ 检测器初始化成功")
        print()

        # 测试每种模式
        for mode, mode_name in modes:
            print(f"测试模式: {mode_name} ({mode})...")

            # 提取人脸
            extracted = detector.extract_face_images(
                image,
                padding=20,
                crop_mode=mode,
                crop_padding_ratio=0.5
            )

            if not extracted:
                print(f"  ✗ 未检测到人脸")
                print()
                continue

            print(f"  ✓ 检测到 {len(extracted)} 个人脸")

            # 保存第一个检测结果
            if extracted:
                face_image, metadata = extracted[0]

                # 保存
                filename = f"{Path(image_path).stem}_{mode}.jpg"
                output_path = output_dir / filename
                cv2.imwrite(str(output_path), face_image)

                # 显示信息
                bbox = metadata['bbox_with_padding']
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                print(f"  ✓ 保存: {filename}")
                print(f"    输出尺寸: {face_image.shape[1]}x{face_image.shape[0]}")
                print(f"    裁剪范围: {w}x{h}")
                print()

        print("=" * 60)
        print("测试完成！")
        print(f"所有结果已保存到: {output_dir}")
        print()
        print("对比说明：")
        print("- face: 输出尺寸最小，仅包含人脸")
        print("- upper_body: 输出尺寸中等，包含头部到胸/腰部")
        print("- full_body: 输出尺寸最大，尽可能包含完整人物")

    except Exception as e:
        print(f"✗ 错误: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("使用方法:")
        print(f"  python {sys.argv[0]} <图像路径>")
        print()
        print("示例:")
        print(f"  python {sys.argv[0]} test.jpg")
        return

    image_path = sys.argv[1]
    test_crop_modes(image_path)


if __name__ == "__main__":
    main()
