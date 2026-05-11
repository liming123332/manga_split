"""YOLO 人脸检测快速测试"""

import cv2
import sys
from pathlib import Path

from src.detection.yolo_face_detector import YOLOFaceDetector


def main():
    """快速测试 YOLO 检测效果"""
    print("=" * 60)
    print("YOLO 二次元人脸检测 - 快速测试")
    print("=" * 60)
    print()

    # 检查命令行参数
    if len(sys.argv) < 2:
        print("使用方法:")
        print(f"  python {sys.argv[0]} <图像路径> [置信度阈值]")
        print()
        print("示例:")
        print(f"  python {sys.argv[0]} test.jpg")
        print(f"  python {sys.argv[0]} test.jpg 0.3")
        print()
        return

    image_path = sys.argv[1]
    confidence = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    # 读取图像
    print(f"1. 读取图像: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"   错误: 无法读取图像")
        return

    print(f"   图像尺寸: {image.shape[1]}x{image.shape[0]}")
    print()

    # 创建检测器
    print("2. 初始化检测器...")
    try:
        detector = YOLOFaceDetector(
            model_path="./models/yolov8x6_animeface.pt",
            confidence_threshold=confidence,
            device="cuda"
        )
        print("   检测器初始化成功")
    except Exception as e:
        print(f"   错误: {e}")
        return

    print()

    # 检测人脸
    print("3. 检测人脸...")
    try:
        faces = detector.detect_faces(image)
        print(f"   检测到 {len(faces)} 个人脸")
    except Exception as e:
        print(f"   错误: {e}")
        return

    if not faces:
        print()
        print("未检测到人脸，可能原因:")
        print("- 置信度阈值过高（尝试降低到 0.3）")
        print("- 图像中没有明显的人脸")
        print("- 图像质量问题")
        return

    print()

    # 绘制结果
    print("4. 绘制检测结果...")
    result_image = image.copy()

    for idx, (x, y, w, h) in enumerate(faces):
        # 绘制边界框
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # 添加标签
        cv2.putText(
            result_image,
            f"Face {idx + 1}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )
        print(f"   人脸 {idx + 1}: x={x}, y={y}, w={w}, h={h}")

    print()

    # 保存结果
    output_path = Path(image_path).stem + "_detected.jpg"
    cv2.imwrite(output_path, result_image)
    print(f"5. 保存结果: {output_path}")
    print()

    # 提取人脸
    print("6. 提取人脸图像...")
    extracted = detector.extract_face_images(image, padding=30)

    for idx, (face_image, metadata) in enumerate(extracted):
        face_path = f"{Path(image_path).stem}_face{idx + 1}.jpg"
        cv2.imwrite(face_path, face_image)
        print(f"   保存: {face_path} ({face_image.shape[1]}x{face_image.shape[0]})")

    print()
    print("=" * 60)
    print("测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
