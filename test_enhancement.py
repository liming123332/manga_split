"""图像增强效果测试脚本"""

import cv2
import numpy as np
import argparse
from pathlib import Path

from src.preprocess.image_enhancer import ImageEnhancer


def test_enhancement(image_path: str, output_dir: str = "./enhancement_test"):
    """
    测试图像增强效果

    Args:
        image_path: 输入图像路径
        output_dir: 输出目录
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

    # 创建增强器
    enhancer = ImageEnhancer(
        enable_clahe=True,
        clahe_clip_limit=2.0,
        clahe_tile_size=8,
        enable_sharpen=True,
        sharpen_strength=1.0,
        enable_denoise=False
    )

    # 1. 保存原始图像
    cv2.imwrite(f"{output_dir}/1_original.jpg", image)
    print("✓ 保存: 1_original.jpg")

    # 2. 应用检测增强
    enhanced_detection = enhancer.enhance_for_detection(image)
    cv2.imwrite(f"{output_dir}/2_enhanced_for_detection.jpg", enhanced_detection)
    print("✓ 保存: 2_enhanced_for_detection.jpg")

    # 3. 应用输出增强（保守模式）
    enhanced_output = enhancer.enhance_for_output(image)
    cv2.imwrite(f"{output_dir}/3_enhanced_for_output.jpg", enhanced_output)
    print("✓ 保存: 3_enhanced_for_output.jpg")

    # 4. 创建对比图
    h, w = image.shape[:2]
    comparison = np.zeros((h, w * 2, 3), dtype=np.uint8)
    comparison[:, :w] = image
    comparison[:, w:] = enhanced_detection
    cv2.imwrite(f"{output_dir}/4_comparison.jpg", comparison)
    print("✓ 保存: 4_comparison.jpg (左：原始 | 右：增强)")

    print(f"\n所有测试图像已保存到: {output_dir}")
    print("\n说明：")
    print("- 1_original.jpg: 原始图像")
    print("- 2_enhanced_for_detection.jpg: 检测增强（对比度+锐化，用于提高 SAM 精度）")
    print("- 3_enhanced_for_output.jpg: 输出增强（保守模式，保持原始画风）")
    print("- 4_comparison.jpg: 左右对比图")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="测试图像增强效果")
    parser.add_argument("image", help="输入图像路径")
    parser.add_argument("--output", default="./enhancement_test", help="输出目录")

    args = parser.parse_args()

    test_enhancement(args.image, args.output)


if __name__ == "__main__":
    main()
