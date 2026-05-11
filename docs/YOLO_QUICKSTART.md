# YOLO 人脸检测快速开始指南

## 5 分钟快速上手

### 步骤 1：安装依赖

```bash
pip install ultralytics
```

### 步骤 2：配置使用 YOLO

编辑 `config.yaml`：

```yaml
detection:
  method: "yolo"  # 改为 "yolo"
  yolo_model_path: "yolov8n.pt"  # 使用标准模型（会自动下载）
```

### 步骤 3：运行

```bash
python gui.py
```

就这么简单喵～ ฅ'ω'ฅ

---

## 三种使用模式

### 模式 1：标准 YOLOv8（最简单）✨

**优点：**
- 零配置，自动下载模型
- 适合快速测试

**配置：**
```yaml
detection:
  method: "yolo"
  yolo_model_path: "yolov8n.pt"
```

**运行：**
```bash
python gui.py
```

---

### 模式 2：二次元专用模型（推荐效果）⭐

**优点：**
- 专门针对二次元人脸训练
- 检测精度更高

**配置：**
1. 下载模型：
   - 访问 https://huggingface.co/Fuyucchi/yolov8_animeface
   - 下载 `.pt` 文件到 `models/` 目录

2. 更新配置：
```yaml
detection:
  method: "yolo"
  yolo_model_path: "./models/yolov8x_animeface.pt"
```

**运行：**
```bash
python gui.py
```

---

### 模式 3：SAM 高精度（精确分割）🌟

**优点：**
- 检测精度最高
- 适合复杂场景

**配置：**
```yaml
detection:
  method: "sam"  # 切换到 SAM
  face_crop_ratio: 0.5
  face_center_crop: 0.6
```

**运行：**
```bash
python gui.py
```

---

## 测试命令

### 测试 YOLO 检测效果

```bash
python test_yolo_detection.py "path/to/your/image.jpg"
```

### 查看模型下载说明

```bash
python download_yolo_model.py
```

---

## 性能对比

| 方法 | 速度 | 精度 | 适用场景 |
|------|------|------|---------|
| **YOLO** | ⚡⚡⚡ | 👍👍 | 批量处理 |
| **SAM** | ⚡ | 👍👍👍 | 精确需求 |

**浮浮酱推荐：**
- 先用 YOLO 快速批量处理
- 对效果不满意的图像再用 SAM 精细处理

---

## 常见问题 30 秒解决

### Q: 检测不到人脸？

```yaml
# 降低置信度阈值
confidence_threshold: 0.3
```

### Q: 误检太多？

```yaml
# 提高置信度阈值
confidence_threshold: 0.7
```

### Q: 想要更好的效果？

```yaml
# 启用图像增强
preprocess:
  enable_enhancement: true
```

### Q: YOLO 和 SAM 怎么选？

- **大量漫画，追求速度** → YOLO
- **少量重要图像，追求精度** → SAM

---

## 完整配置示例

推荐配置（YOLO + 图像增强）：

```yaml
detection:
  method: "yolo"
  yolo_model_path: "yolov8n.pt"
  confidence_threshold: 0.5
  iou_threshold: 0.45

preprocess:
  enable_enhancement: true
  clahe_enabled: true
  sharpen_enabled: true
  denoise_enabled: false
```

保存后运行：
```bash
python gui.py
```

---

## 进阶：自动切换策略

可以根据图像特点自动选择检测方法（需要自定义代码）：

```python
# 简单图像用 YOLO
# 复杂图像用 SAM

# 批量处理时先用 YOLO 快速筛选
# 对检测失败的图像再用 SAM 补充
```

---

准备好了吗？

选择一个模式，开始处理你的漫画吧喵～ φ(≧ω≦*)♪

```bash
python gui.py
```
