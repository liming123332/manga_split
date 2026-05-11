# 条漫人物自动截取系统

自动化从条漫（纵向漫画）中截取人物图像，用于训练 SDXL 的 LoRA 模型。支持长条图自动分割、双模式 AI 检测（YOLO/SAM）、可调节提取范围等功能。

## ✨ 特性

- 🚀 **双模式检测** - YOLO 快速检测（推荐）或 SAM 高精度分割
- 📐 **长条图自动分割** - 智能识别并分割纵向长条图为单格画面
- 🎯 **可调节提取范围** - 支持仅人脸、上半身、全身三种模式
- 🎨 **图像增强预处理** - CLAHE 对比度增强、锐化、去噪
- 🖥️ **友好的 GUI 界面** - 图形化操作，实时进度显示
- 📦 **批量处理** - 递归扫描整个文件夹，自动化处理
- 🎯 **质量保证** - 白边裁剪、尺寸过滤、JPG 高质量输出

## 📊 性能对比

| 检测方法 | 速度 | 精度 | 显存 | 适用场景 |
|---------|------|------|------|---------|
| **YOLO** (推荐) | ⚡⚡⚡ 10-20ms/图 | 👍👍 良好 | ~2GB | 大批量快速处理 |
| **SAM** | ⚡ 1-3s/图 | 👍👍👍 优秀 | ~3-5GB | 精细高质量处理 |

## 🚀 快速开始

### 1. 环境要求

- Python 3.8+
- CUDA 11.0+ (推荐 GPU，RTX 5080 已支持)
- 8GB+ 显存 (推荐)，或使用 CPU 模式

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

**注意**: 
- PyTorch 2.9.1+cu130 已支持 RTX 5080
- 如遇问题请访问 [PyTorch 官网](https://pytorch.org/get-started/locally/)

### 3. 下载模型

#### YOLO 模型（推荐，快速）

1. 下载二次元人脸检测模型：
   - 访问 [YOLOv8-AnimeFace](https://github.com/Fuyucch1/yolov8_animeface) 下载模型
   - 或从 [HuggingFace](https://huggingface.co/Fuyucchi/yolov8_animeface) 下载
   - 将 `.pt` 文件放到 `models/` 目录

2. 或使用标准模型（会自动下载）：
   ```yaml
   # config.yaml
   detection:
     yolo_model_path: "yolov8n.pt"
   ```

#### SAM 模型（可选，高精度）

```bash
python main.py download-model --model-type vit_l
```

### 4. 开始使用

#### 方式 A：GUI 模式（推荐）⭐

```bash
python gui.py
```

**GUI 功能：**
- 📁 可视化选择输入/输出目录
- 🔀 **检测方法切换** - YOLO（快速）或 SAM（高精度）
- 🎯 **提取范围选择** - 仅人脸 / 上半身（推荐） / 全身
- 🎛️ 图形化调整所有参数
- 📊 实时进度条和日志显示
- 📈 处理完成摘要统计

**使用步骤：**
1. 运行 `python gui.py`
2. 选择输入目录（漫画文件夹）和输出目录
3. **选择检测方法**：
   - ✅ **YOLO (快速推荐)** ← 适合大批量处理
   - ✅ **SAM (高精度)** ← 适合高质量需求
4. **选择提取范围**：
   - ○ 仅人脸
   - ⦿ **上半身 (推荐)** ← LoRA 训练推荐
   - ○ 全身
5. 点击"开始处理"
6. 等待完成，查看输出

#### 方式 B：命令行模式

编辑 `config.yaml`：

```yaml
input:
  directory: "D:/漫画/作品名"  # 你的漫画文件夹路径
  recursive: true

detection:
  method: "yolo"              # 检测方法: "yolo" 或 "sam"
  crop_mode: "upper_body"     # 提取范围: "face" / "upper_body" / "full_body"
```

运行处理：

```bash
python main.py process
```

## 🎯 提取范围说明

系统支持三种提取模式，满足不同需求：

### 模式对比

| 模式 | 提取内容 | 适用场景 | 输出大小 |
|------|---------|---------|---------|
| **仅人脸** (face) | 头部 + 颈部 | 人脸识别训练 | 最小 |
| **上半身** (upper_body) ⭐ | 头部 + 肩膀 + 胸部 | LoRA 训练（推荐） | 中等 |
| **全身** (full_body) | 完整人物 | 全身建模 | 最大 |

### 推荐配置

**LoRA 训练（主人的需求）：**
```yaml
detection:
  method: "yolo"              # 快速检测
  crop_mode: "upper_body"     # 上半身（保留更多特征）
  crop_padding_ratio: 0.5     # 标准边距
```

**优势：**
- ✅ 保留完整的面部特征
- ✅ 包含头发、服装、配饰等细节
- ✅ 适中的背景（不会太多）
- ✅ 适合 SDXL LoRA 训练

## ⚙️ 配置说明

### 检测配置

```yaml
detection:
  # 检测方法选择
  method: "yolo"              # "yolo"(快速) 或 "sam"(高精度)
  
  # YOLO 配置
  yolo_model_path: "./models/yolov8x6_animeface.pt"
  confidence_threshold: 0.5   # 置信度阈值（0-1）
  iou_threshold: 0.45         # NMS IoU 阈值
  
  # 提取范围配置
  crop_mode: "upper_body"     # "face" / "upper_body" / "full_body"
  crop_padding_ratio: 0.5     # 额外边距比例
```

### 图像增强配置

```yaml
preprocess:
  enable_enhancement: true    # 启用检测前图像增强（推荐）
  clahe_enabled: true         # CLAHE 对比度增强
  sharpen_enabled: true       # 锐化处理
  denoise_enabled: false      # 去噪（计算密集，慎用）
```

### 输出配置

```yaml
output:
  directory: "./output"
  format: "jpg"               # 输出格式（jpg/png）
  quality: 95                 # JPEG 质量（1-100）
```

## 📁 输出结构

```
output/
├── images/                  # 提取的人物图像
│   ├── manga_001_p00_face00_20260511_120000.jpg
│   ├── manga_001_p00_face01_20260511_120001.jpg
│   └── ...
├── metadata/                # 元数据 JSON 文件
│   └── metadata.json
└── report.json              # 处理报告
```

## 🔧 高级用法

### 测试不同提取模式

```bash
python test_crop_modes.py "路径/到/测试图像.jpg"
```

会生成三张对比图像：
- `test_face.jpg` - 仅人脸
- `test_upper_body.jpg` - 上半身
- `test_full_body.jpg` - 全身

### 切换检测方法

**使用 YOLO（快速批量）：**
```yaml
detection:
  method: "yolo"
  yolo_model_path: "./models/yolov8x6_animeface.pt"
```

**使用 SAM（高精度）：**
```yaml
detection:
  method: "sam"
  face_crop_ratio: 0.5
  face_center_crop: 0.6
```

### 调整检测精度

如果检测不到人脸：
```yaml
detection:
  confidence_threshold: 0.3  # 降低阈值
```

如果误检太多：
```yaml
detection:
  confidence_threshold: 0.7  # 提高阈值
```

## 📖 常见问题

### Q: YOLO 和 SAM 哪个更好？

**A:** 取决于需求：
- **大批量快速处理** → YOLO（推荐）
- **少量重要图像** → SAM（高精度）
- **混合使用** → YOLO 先筛，SAM 补充

### Q: 提取的人物太小怎么办？

**A:** 调整提取范围：
```yaml
detection:
  crop_mode: "upper_body"     # 或 "full_body"
  crop_padding_ratio: 0.8     # 增大边距
```

### Q: 想要更多人物细节（服装、姿势）？

**A:** 使用上半身或全身模式：
```yaml
detection:
  crop_mode: "full_body"     # 全身模式
```

### Q: 检测速度慢怎么办？

**A:** 使用 YOLO 模式：
```yaml
detection:
  method: "yolo"              # 切换到 YOLO
```

速度可提升 **100-150 倍**！

### Q: 如何提高检测精度？

**A:** 
1. 启用图像增强：
   ```yaml
   preprocess:
     enable_enhancement: true
   ```
2. 降低置信度阈值：
   ```yaml
   detection:
     confidence_threshold: 0.3
   ```
3. 或切换到 SAM 模式

### Q: 显存不足怎么办？

**A:**
1. 使用 YOLO（显存占用 ~2GB）
2. 或使用 CPU 模式：
   ```yaml
   sam:
     device: "cpu"
   ```

## 🛠️ 技术细节

### YOLO 检测流程

1. **检测** - YOLO 模型检测所有人脸
2. **裁剪** - 根据裁剪模式扩展边界框
3. **过滤** - NMS 去除重叠检测
4. **提取** - 裁剪并保存图像

### SAM 检测流程

1. **提示生成** - 网格采样生成自动提示点
2. **SAM 分割** - 调用 SAM 模型生成掩码
3. **几何裁剪** - 根据裁剪模式提取区域
4. **质量过滤** - 基于面积、位置过滤
5. **后处理** - 边缘羽化、白边裁剪

### 长条图分割算法

1. **检测** - 计算高宽比（>3.0 视为长条图）
2. **投影** - 水平投影法检测分割线
3. **分割** - 根据分割点裁剪
4. **保存** - 保存为单独图像

## 📂 项目结构

```
manga_split/
├── src/
│   ├── detection/              # 检测模块
│   │   ├── yolo_face_detector.py      # YOLO 人脸检测
│   │   ├── anime_face_detector.py     # SAM + 几何裁剪
│   │   ├── face_detector_factory.py   # 检测器工厂
│   │   ├── sam_wrapper.py             # SAM 模型封装
│   │   └── person_detector.py         # 人物检测器
│   ├── preprocess/             # 预处理模块
│   │   ├── strip_splitter.py          # 长条图分割
│   │   └── image_enhancer.py          # 图像增强
│   ├── pipeline/               # 工作流模块
│   │   └── batch_processor.py         # 批量处理
│   ├── postprocess/            # 后处理模块
│   │   ├── background_removal.py      # 背景处理
│   │   └── cropper.py                  # 裁剪
│   └── utils/                  # 工具模块
│       └── config.py                    # 配置加载
├── models/                         # 模型存储目录
│   ├── yolov8x6_animeface.pt         # YOLO 模型（需下载）
│   ├── sam_vit_l_0b3195.pth          # SAM 模型（可选）
│   └── lbpcascade_animeface.xml       # Haar 分类器
├── docs/                           # 文档目录
│   ├── YOLO_DETECTION.md            # YOLO 检测指南
│   ├── YOLO_QUICKSTART.md           # 快速开始
│   ├── GUIDE_YOLO_GUI.md            # GUI 使用指南
│   ├── IMAGE_ENHANCEMENT.md         # 图像增强说明
│   └── CROP_MODE_GUIDE.md           # 提取范围说明
├── config.yaml                     # 配置文件
├── gui.py                          # 图形界面
├── main.py                         # 命令行入口
├── requirements.txt                # 依赖列表
└── README.md                       # 本文档
```

## 🎓 教程和文档

详细教程请查看 `docs/` 目录：

- [YOLO 检测使用指南](docs/YOLO_DETECTION.md) - YOLO 检测详细说明
- [快速开始指南](docs/YOLO_QUICKSTART.md) - 5 分钟上手
- [GUI 使用指南](docs/GUIDE_YOLO_GUI.md) - GUI 界面说明
- [图像增强说明](docs/IMAGE_ENHANCEMENT.md) - 增强功能详解
- [提取范围说明](docs/CROP_MODE_GUIDE.md) - 裁剪模式详解

## 📊 性能参考

基于 NVIDIA RTX 5080 的测试数据：

| 模式 | 模型 | 处理速度 | 显存 | 推荐场景 |
|------|------|---------|------|---------|
| YOLO | yolov8x6 | ~10ms/图 | ~2GB | 大批量处理 ⭐ |
| SAM | vit_l | ~2s/图 | ~3GB | 高质量处理 |

## 🙏 致谢

- [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) - Meta 开源的高质量分割模型
- [YOLOv8-AnimeFace](https://github.com/Fuyucch1/yolov8_animeface) - 二次元人脸检测模型
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - YOLOv8 实现
- [OpenCV](https://opencv.org/) - 计算机视觉库
- [PyTorch](https://pytorch.org/) - 深度学习框架

## 📜 许可证

本项目仅供学习和研究使用。

## 💬 联系方式

如有问题或建议，欢迎提 Issue。

---

**快速开始使用：**
```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动 GUI
python gui.py

# 3. 选择检测方法：YOLO（快速）
# 4. 选择提取范围：上半身（推荐）
# 5. 开始处理！
```

**推荐配置（LoRA 训练）：**
```yaml
detection:
  method: "yolo"              # 快速检测
  crop_mode: "upper_body"     # 上半身（保留特征）
  confidence_threshold: 0.5
```

祝使用愉快！ฅ'ω'ฅ
