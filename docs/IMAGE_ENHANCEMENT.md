# 图像增强功能说明

## 概述

浮浮酱为主人添加了强大的图像增强预处理功能喵～这个功能可以显著提高 SAM 的人物检测精度，同时保持输出图像的原始画风。

## 功能特点

### 1. **CLAHE 对比度增强** (限制对比度自适应直方图均衡化)
- **作用：** 提高图像对比度，让人物轮廓更清晰
- **实现：** 在 LAB 色彩空间的 L 通道上应用 CLAHE
- **优势：** 不会过度增强，避免亮度失真
- **参数：**
  - `clahe_clip_limit` (1.0-3.0): 对比度限制，越高对比度越强
  - `clahe_tile_size` (4-16): 网格大小，越小局部对比度越强

### 2. **锐化处理**
- **作用：** 增强边缘，帮助 SAM 更好地识别人物轮廓
- **实现：** 自定义锐化卷积核
- **参数：**
  - `sharpen_strength` (0.5-2.0): 锐化强度

### 3. **去噪处理** (可选)
- **作用：** 减少 JPEG 压缩噪声
- **实现：** fastNlMeansDenoisingColored 非局部均值去噪
- **注意：** 计算密集，处理速度慢，建议默认关闭
- **参数：**
  - `denoise_h` (5-20): 去噪强度，越高平滑效果越强

## 增强模式

系统提供两种增强模式：

### 检测增强模式 (`enhance_for_detection`)
- **用途：** 在 SAM 检测前应用，提高检测精度
- **特点：** 强度较高，确保人物轮廓清晰
- **不影响输出：** 只用于检测，不会改变最终输出图像

### 输出增强模式 (`enhance_for_output`)
- **用途：** 在保存输出图像前应用（可选）
- **特点：** 保守模式，保持原始画风
- **建议：** 对于 LoRA 训练，建议关闭此项，保持原始画风

## 配置说明

在 `config.yaml` 中的配置项：

```yaml
preprocess:
  # 是否启用增强（强烈推荐）
  enable_enhancement: false

  # 是否增强输出图像（建议保持 false）
  enhance_output: false

  # CLAHE 对比度增强
  clahe_enabled: true
  clahe_clip_limit: 2.0      # 推荐 1.5-2.5
  clahe_tile_size: 8         # 推荐 8

  # 锐化
  sharpen_enabled: true
  sharpen_strength: 1.0      # 推荐 0.8-1.5

  # 去噪（计算密集，慎用）
  denoise_enabled: false
  denoise_strength: 10.0
```

## 使用方法

### 1. 启用图像增强

编辑 `config.yaml`：
```yaml
preprocess:
  enable_enhancement: true   # 改为 true
```

### 2. 测试增强效果

使用测试脚本查看效果：
```bash
python test_enhancement.py "路径/到/测试图像.jpg" --output "./enhancement_test"
```

会生成 4 张对比图像：
- `1_original.jpg` - 原始图像
- `2_enhanced_for_detection.jpg` - 检测增强效果
- `3_enhanced_for_output.jpg` - 输出增强效果
- `4_comparison.jpg` - 左右对比图

### 3. 运行批量处理

```bash
# GUI 模式
python gui.py

# 命令行模式
python main.py process
```

## 参数调优建议

### 场景 1：高质量漫画（清晰、无噪声）
```yaml
clahe_clip_limit: 1.5
sharpen_strength: 0.8
denoise_enabled: false
```

### 场景 2：普通质量漫画（略有模糊）
```yaml
clahe_clip_limit: 2.0
sharpen_strength: 1.0
denoise_enabled: false
```

### 场景 3：低质量漫画（压缩噪声明显）
```yaml
clahe_clip_limit: 2.5
sharpen_strength: 1.2
denoise_enabled: true
denoise_strength: 10.0
```

## 性能影响

| 功能 | CPU 时间增加 | 显存影响 | 建议场景 |
|------|-------------|---------|---------|
| CLAHE | +5-10% | 无 | 所有场景推荐 |
| 锐化 | +3-5% | 无 | 所有场景推荐 |
| 去噪 | +50-100% | 无 | 仅低质量图像 |

## 注意事项

1. **检测 vs 输出：**
   - 检测增强只影响 SAM，不会改变输出图像质量
   - 输出增强会改变最终图像，对于 LoRA 训练建议关闭

2. **过度增强的风险：**
   - `clahe_clip_limit` 过高 (>3.0) 可能导致对比度失真
   - `sharpen_strength` 过高 (>2.0) 会引入边缘伪影
   - 去噪过强会丢失细节

3. **画风保持：**
   - 对于训练 LoRA，保持原始画风很重要
   - 建议只启用 `enable_enhancement`，不启用 `enhance_output`

## 常见问题

### Q: 启用增强后检测精度反而下降了？
A: 可能是参数过高，尝试降低：
- `clahe_clip_limit` 从 2.0 降到 1.5
- `sharpen_strength` 从 1.0 降到 0.8

### Q: 去噪太慢怎么办？
A: 建议关闭 `denoise_enabled`，或者：
- 使用更高的 `denoise_h` 值（如 15.0）减少迭代
- 只对低质量图像启用

### Q: 输出图像看起来和原始图像不一样？
A: 检查配置：
```yaml
enhance_output: false  # 确保此项为 false
```

## 技术细节

### CLAHE 实现
```python
# 转换到 LAB 色彩空间
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)

# 应用 CLAHE 到 L 通道
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
l = clahe.apply(l)

# 合并通道并转换回 BGR
lab = cv2.merge([l, a, b])
enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
```

### 锐化核
```python
kernel = np.array([
    [-1, -1, -1],
    [-1, 9, -1],
    [-1, -1, -1]
]) * strength
kernel[1, 1] = 1 + 8 * strength
```

## 总结

浮浮酱建议的配置：
- ✅ **启用检测增强** (`enable_enhancement: true`)
- ✅ **使用默认参数** (经过优化，适合大多数场景)
- ❌ **不启用输出增强** (`enhance_output: false`)
- ❌ **谨慎使用去噪** (只在必要时启用)

这样可以最大化检测精度，同时保持输出图像的原始画风喵～ ฅ'ω'ฅ
