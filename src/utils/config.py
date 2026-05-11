"""配置加载和验证模块"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, List


class Config:
    """配置管理类"""

    def __init__(self, config_path: str = "config.yaml", strict_validation: bool = True):
        """
        初始化配置

        Args:
            config_path: 配置文件路径
            strict_validation: 是否严格验证（GUI模式可设为False）
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.strict_validation = strict_validation
        if strict_validation:
            self._validate_config()
        else:
            self._validate_config_basic()

    def _load_config(self) -> Dict[str, Any]:
        """加载 YAML 配置文件"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        return config

    def _validate_config_basic(self):
        """基本配置验证（不验证目录路径，用于GUI模式）"""
        required_sections = ['input', 'output', 'sam', 'detection', 'postprocess']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"配置文件缺少必需的节: {section}")

        # 验证 SAM 模型配置
        valid_model_types = ['vit_h', 'vit_l', 'vit_b']
        if self.config['sam']['model_type'] not in valid_model_types:
            raise ValueError(f"无效的 SAM 模型类型: {self.config['sam']['model_type']}")

        valid_devices = ['cuda', 'cpu', 'mps']
        if self.config['sam']['device'] not in valid_devices:
            raise ValueError(f"无效的设备类型: {self.config['sam']['device']}")

    def _validate_config(self):
        """验证配置的完整性"""
        # 先进行基本验证
        self._validate_config_basic()

        # 验证输入目录
        if not self.config['input']['directory']:
            raise ValueError("请先在 config.yaml 中设置 input.directory")

        if not os.path.exists(self.config['input']['directory']):
            raise FileNotFoundError(f"输入目录不存在: {self.config['input']['directory']}")

        # 验证输出目录
        output_dir = self.config['output']['directory']
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'metadata'), exist_ok=True)

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        获取配置项（支持嵌套路径）

        Args:
            key_path: 配置项路径，使用点号分隔，例如 'sam.model_type'
            default: 默认值

        Returns:
            配置值
        """
        keys = key_path.split('.')
        value = self.config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def set(self, key_path: str, value: Any):
        """
        设置配置项（支持嵌套路径）

        Args:
            key_path: 配置项路径，使用点号分隔
            value: 新值
        """
        keys = key_path.split('.')
        config = self.config

        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        config[keys[-1]] = value

    def save(self, path: str = None):
        """
        保存配置到文件

        Args:
            path: 保存路径，默认覆盖原配置文件
        """
        save_path = path or self.config_path

        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, allow_unicode=True, default_flow_style=False)

    @property
    def input_dir(self) -> str:
        """输入目录"""
        return self.config['input']['directory']

    @property
    def output_dir(self) -> str:
        """输出目录"""
        return self.config['output']['directory']

    @property
    def input_recursive(self) -> bool:
        """是否递归扫描输入目录"""
        return self.config['input']['recursive']

    @property
    def supported_formats(self) -> List[str]:
        """支持的图像格式"""
        return self.config['input']['supported_formats']

    @property
    def sam_model_type(self) -> str:
        """SAM 模型类型"""
        return self.config['sam']['model_type']

    @property
    def sam_device(self) -> str:
        """SAM 运行设备"""
        return self.config['sam']['device']

    @property
    def sam_checkpoint(self) -> str:
        """SAM 模型权重路径"""
        return self.config['sam']['checkpoint_path']

    @property
    def min_person_size(self) -> int:
        """人物最小尺寸"""
        return self.config['detection']['min_person_size']

    @property
    def edge_blur(self) -> int:
        """边缘羽化半径"""
        return self.config['postprocess']['edge_blur']

    @property
    def crop_borders(self) -> bool:
        """是否裁剪白边"""
        return self.config['postprocess']['crop_borders']

    def __repr__(self) -> str:
        """字符串表示"""
        return f"Config(input_dir='{self.input_dir}', output_dir='{self.output_dir}')"
