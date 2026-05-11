import cv2
import numpy as np
from src.detection.anime_face_detector import AnimeFaceDetector

# 创建一个测试图像（白色背景上的简单形状）
test_image = np.ones((400, 600, 3), dtype=np.uint8) * 255
cv2.rectangle(test_image, (200, 100), (400, 300), (200, 150, 100), -1)

# 测试检测器
detector = AnimeFaceDetector(min_size=30)
faces = detector.detect_faces(test_image)
print(f"检测到 {len(faces)} 个人脸")
print(f"测试图像形状: {test_image.shape}")
