# 条漫人物自动截取系统

自动化从条漫（纵向漫画）中截取人物图像，用于训练 SDXL 的 LoRA 模型。支持长条图自动分割、AI 人物检测、透明背景生成等功能。

## 特性

- ✨ **智能长条图分割** - 自动识别并分割纵向长条图为单格画面
- 🤖 **AI 人物检测** - 使用 Meta 开源的 Segment Anything Model (SAM)，零样本高精度分割
- 🎨 **透明背景输出** - 自动生成带 Alpha 通道的 PNG 图像
- 📦 **批量处理** - 支持递归扫描整个文件夹，自动化处理
- 🎯 **质量保证** - 白边裁剪、边缘羽化、尺寸过滤

## 快速开始

### 1. 环境要求

- Python 3.8+
- CUDA 11.0+ (推荐 GPU，可选 CPU)
- 8GB+ 显存 (推荐)，或使用 CPU 模式

### 2. 使用方式

本系统提供两种使用方式：

#### 方式 A：图形界面（推荐，简单易用）

```bash
# 直接运行图形界面
python gui.py
```

图形界面功能：
- 📁 可视化选择输入/输出目录
- 🎛️ 图形化调整所有参数
- 📊 实时进度条和日志显示
- ⚙️ 一键下载模型
- 📈 处理完成摘要统计

**使用步骤：**
1. 运行 `python gui.py` 打开图形界面
2. 点击"下载模型"按钮下载 SAM 模型（首次使用）
3. 选择输入目录（漫画文件夹）和输出目录
4. 选择 SAM 模型类型和运行设备
5. 点击"开始处理"按钮
6. 等待处理完成，查看日志输出

#### 方式 B：命令行界面（适合自动化脚本）

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

**注意**: 如果遇到 PyTorch 安装问题，请访问 [PyTorch 官网](https://pytorch.org/get-started/locally/) 获取适合您系统的安装命令。

### 4. 下载 SAM 模型

```bash
# 下载推荐模型（vit_l，平衡精度和速度）
python main.py download-model --model-type vit_l

# 或下载其他模型
python main.py download-model --model-type vit_h  # 最高精度，速度较慢
python main.py download-model --model-type vit_b  # 最快速度，略低精度
```

### 5. 开始使用

**图形界面模式（推荐）：**

```bash
python gui.py
```

**命令行模式：**

首先编辑 `config.yaml` 文件，设置输入路径：

```yaml
input:
  directory: "D:/漫画/作品名"  # 修改为你的漫画文件夹路径
  recursive: true
```

然后运行处理：

```bash
# 使用配置文件处理
python main.py process

# 或直接指定参数
python main.py process --input "D:/漫画/作品名" --output "./output"
```

```bash
# 使用配置文件处理
python main.py process

# 或直接指定参数
python main.py process --input "D:/漫画/作品名" --output "./output"
```

## 高级用法

### 使用不同的 SAM 模型

```bash
# 高精度模式（推荐用于最终数据生成）
python main.py process --model vit_h

# 平衡模式（推荐）
python main.py process --model vit_l

# 快速模式（适合测试）
python main.py process --model vit_b
```

### CPU 模式

如果没有 GPU 或显存不足：

```bash
python main.py process --device cpu
```

### 自定义配置

编辑 `config.yaml` 调整更多参数：

```yaml
# 人物检测配置
detection:
  min_person_size: 256  # 人物最小尺寸（降低可检测更小的人物）
  max_persons_per_image: 10  # 单张图像最大检测人数

# 后处理配置
postprocess:
  edge_blur: 3  # 边缘羽化半径（增加可使边缘更柔和）
  crop_borders: true  # 是否裁剪白边
  padding: 10  # 裁剪后保留的边距
  min_output_size: 512  # 输出图像最小尺寸
  max_output_size: 2048  # 输出图像最大尺寸
```

## 输出结构

```
output/
├── images/          # 提取的人物图像（PNG 格式，透明背景）
├── metadata/        # 元数据 JSON 文件
│   └── metadata.json
└── report.json      # 处理报告
```

## 命令参考

### 下载模型

```bash
python main.py download-model [OPTIONS]

选项:
  --model-type {vit_h,vit_l,vit_b}  模型类型（默认: vit_l）
  --output-dir PATH                 模型保存目录（默认: ./models）
```

### 处理漫画

```bash
python main.py process [OPTIONS]

选项:
  --config PATH          配置文件路径（默认: config.yaml）
  --input PATH           输入目录（覆盖配置文件）
  --output PATH          输出目录（覆盖配置文件）
  --model {vit_h,vit_l,vit_b}  SAM 模型类型
  --device {cuda,cpu,mps}       运行设备
  --quiet               静默模式（不显示进度）
```

## 常见问题

### Q: 处理速度慢怎么办？

A: 可以尝试：
1. 使用更快的模型：`--model vit_b`
2. 使用 GPU（如果有）：`--device cuda`
3. 调整检测参数：降低 `max_persons_per_image`

### Q: 检测不到人物？

A: 可能原因和解决方案：
1. 人物太小：降低 `min_person_size` 参数
2. 检测策略问题：修改 `sam.auto_prompt` 为 `center` 或 `random`
3. 图像质量问题：确保输入图像清晰度足够

### Q: 边缘有锯齿？

A: 增加 `edge_blur` 参数（例如设置为 5 或更高）

### Q: 内存不足？

A:
1. 使用更小的模型：`--model vit_b`
2. 使用 CPU 模式：`--device cpu`
3. 降低批处理大小（修改配置文件）

### Q: 如何提高检测精度？

A:
1. 使用高精度模型：`--model vit_h`
2. 调整 `confidence_threshold` 参数
3. 人工审核输出结果，删除低质量样本

## 性能参考

| 模型类型 | 显存占用 | 处理速度 | 精度 | 推荐场景 |
|---------|---------|---------|------|---------|
| vit_h   | ~5GB    | ~5秒/张 | 最高 | 最终数据生成 |
| vit_l   | ~2.5GB  | ~2-3秒/张 | 高  | 日常使用（推荐） |
| vit_b   | ~300MB  | ~1-2秒/张 | 中等 | 快速测试 |

*以上数据基于 NVIDIA RTX 3080 GPU，实际速度因硬件而异*

## 技术细节

### 长条图分割算法

1. **检测**：计算高宽比，大于阈值（默认 3.0）视为长条图
2. **投影**：使用水平投影法检测横向分割线
3. **分割**：根据分割点裁剪图像
4. **保存**：保存为单独的图像文件

### AI 人物检测流程

1. **提示生成**：使用网格采样生成自动提示点
2. **SAM 分割**：调用 SAM 模型生成掩码
3. **质量过滤**：基于面积、位置等过滤低质量结果
4. **NMS 去重**：合并重叠的掩码
5. **后处理**：边缘羽化、白边裁剪

### 输出质量保证

- **边缘羽化**：使用高斯模糊平滑掩码边缘
- **白边裁剪**：自动检测并裁剪透明边框
- **尺寸过滤**：过滤过小的人物
- **格式保证**：输出 PNG (RGBA) 格式，适合训练

## 项目结构

```
manga_split/
├── src/
│   ├── preprocess/          # 预处理模块
│   │   └── strip_splitter.py    # 长条图分割
│   ├── detection/           # 检测模块
│   │   ├── sam_wrapper.py      # SAM 模型封装
│   │   └── person_detector.py  # 人物检测器
│   ├── postprocess/          # 后处理模块
│   │   ├── background_removal.py  # 背景透明化
│   │   └── cropper.py            # 白边裁剪
│   ├── pipeline/             # 工作流模块
│   │   └── batch_processor.py    # 批量处理
│   └── utils/                # 工具模块
│       └── config.py            # 配置加载
├── models/                   # SAM 模型存储目录
├── config.yaml              # 配置文件
├── main.py                  # 主入口
├── requirements.txt         # 依赖列表
└── README.md                # 本文档
```

## 许可证

本项目仅供学习和研究使用。

## 致谢

- [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) - Meta 开源的高质量分割模型
- [OpenCV](https://opencv.org/) - 计算机视觉库
- [PyTorch](https://pytorch.org/) - 深度学习框架

## 联系方式

如有问题或建议，欢迎提 Issue。
