"""YOLO 人脸检测测试脚本"""

import cv2
import argparse
from pathlib import Path

from src.detection.yolo_face_detector import YOLOFaceDetector


def test_yolo_detection(
    image_path: str,
    model_path: str = "yolov8n.pt",
    output_dir: str = "./yolo_test_output",
    confidence: float = 0.5
):
    """
    测试 YOLO 人脸检测

    Args:
        image_path: 输入图像路径
        model_path: YOLO 模型路径
        output_dir: 输出目录
        confidence: 置信度阈值
    """
    # 创建输出目录
    Path(output_dir).mkdir(exist_ok=True)

    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误：无法读取图像 {image_path}")
        return

    print(f"测试图像: {image_path}")
    print(f"图像尺寸: {image.shape}")
    print(f"使用模型: {model_path}")
    print()

    try:
        # 创建检测器
        print("初始化 YOLO 检测器...")
        detector = YOLOFaceDetector(
            model_path=model_path,
            confidence_threshold=confidence,
            device="cuda"  # 如果没有 GPU 会自动降级到 CPU
        )
        print("✓ 检测器初始化成功")
        print()

        # 检测人脸
        print("开始检测人脸...")
        faces = detector.detect_faces(image)
        print(f"✓ 检测到 {len(faces)} 个人脸")
        print()

        if not faces:
            print("未检测到人脸，可能原因：")
            print("1. 图像中没有明显的二次元人脸")
            print("2. 置信度阈值过高（使用 --confidence 降低）")
            print("3. 模型不适合此类型的图像")
            return

        # 绘制检测结果
        result_image = image.copy()
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

        for idx, (x, y, w, h) in enumerate(faces):
            color = colors[idx % len(colors)]
            # 绘制边界框
            cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)
            # 添加标签
            label = f"Face {idx + 1}"
            cv2.putText(
                result_image, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
            print(f"  人脸 {idx + 1}: x={x}, y={y}, w={w}, h={h}")

        print()

        # 保存结果
        output_path = Path(output_dir) / "detection_result.jpg"
        cv2.imwrite(str(output_path), result_image)
        print(f"✓ 检测结果已保存: {output_path}")

        # 提取并保存人脸
        print()
        print("提取人脸图像...")
        extracted = detector.extract_face_images(image, padding=30)

        for idx, (face_image, metadata) in enumerate(extracted):
            face_path = Path(output_dir) / f"face_{idx + 1}.jpg"
            cv2.imwrite(str(face_path), face_image)
            print(f"  ✓ 保存: {face_path}")

        print()
        print(f"完成！所有结果已保存到: {output_dir}")

    except Exception as e:
        print(f"✗ 错误: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="测试 YOLO 人脸检测")
    parser.add_argument("image", help="输入图像路径")
    parser.add_argument("--model", default="yolov8n.pt",
                       help="YOLO 模型路径（默认使用 yolov8n.pt）")
    parser.add_argument("--output", default="./yolo_test_output",
                       help="输出目录")
    parser.add_argument("--confidence", type=float, default=0.5,
                       help="置信度阈值（0-1，默认 0.5）")

    args = parser.parse_args()

    test_yolo_detection(
        args.image,
        args.model,
        args.output,
        args.confidence
    )


if __name__ == "__main__":
    main()
