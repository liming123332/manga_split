"""条漫人物自动截取系统 - 图形界面"""

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import threading
import queue
import logging
from pathlib import Path
from datetime import datetime

from src.utils.config import Config
from src.pipeline.batch_processor import BatchProcessor


class GUIHandler(logging.Handler):
    """自定义日志处理器，将日志输出到GUI"""

    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        """发送日志到队列"""
        log_entry = self.format(record)
        self.log_queue.put(('log', log_entry))


class MangaSplitterGUI:
    """条漫人物截取系统图形界面"""

    def __init__(self, root):
        self.root = root
        self.root.title("条漫人物自动截取系统 v1.0")
        self.root.geometry("900x700")

        # 设置样式
        self.style = ttk.Style()
        self.style.theme_use('clam')

        # 变量
        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.model_type = tk.StringVar(value='vit_l')
        self.device = tk.StringVar(value='cuda')
        self.detection_method = tk.StringVar(value='yolo')  # 检测方法
        self.yolo_model_path = tk.StringVar(value='./models/yolov8x6_animeface.pt')  # YOLO 模型路径
        self.crop_mode = tk.StringVar(value='upper_body')  # 裁剪模式
        self.processing = False
        self.worker_thread = None

        # 日志队列
        self.log_queue = queue.Queue()

        # 创建界面
        self.create_widgets()

        # 启动日志更新
        self.update_log()

    def create_widgets(self):
        """创建界面组件"""
        # 主容器
        main_container = ttk.Frame(self.root, padding="10")
        main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_container.columnconfigure(1, weight=1)

        # 标题
        title_label = ttk.Label(
            main_container,
            text="条漫人物自动截取系统",
            font=('Arial', 16, 'bold')
        )
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 10))

        # === 配置区域 ===
        config_frame = ttk.LabelFrame(main_container, text="配置选项", padding="10")
        config_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        config_frame.columnconfigure(1, weight=1)

        # 输入目录
        ttk.Label(config_frame, text="输入目录:").grid(row=0, column=0, sticky=tk.W, pady=5)
        input_entry = ttk.Entry(config_frame, textvariable=self.input_dir, width=50)
        input_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        ttk.Button(config_frame, text="浏览...", command=self.browse_input).grid(row=0, column=2, pady=5)

        # 输出目录
        ttk.Label(config_frame, text="输出目录:").grid(row=1, column=0, sticky=tk.W, pady=5)
        output_entry = ttk.Entry(config_frame, textvariable=self.output_dir, width=50)
        output_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        ttk.Button(config_frame, text="浏览...", command=self.browse_output).grid(row=1, column=2, pady=5)

        # 检测方法选择
        ttk.Label(config_frame, text="检测方法:").grid(row=2, column=0, sticky=tk.W, pady=5)
        method_frame = ttk.Frame(config_frame)
        method_frame.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)

        methods = [
            ('YOLO (快速推荐)', 'yolo'),
            ('SAM (高精度)', 'sam')
        ]
        for i, (text, value) in enumerate(methods):
            ttk.Radiobutton(
                method_frame,
                text=text,
                variable=self.detection_method,
                value=value,
                command=self.on_detection_method_change
            ).grid(row=0, column=i, padx=10)

        # YOLO 模型路径
        self.yolo_model_frame = ttk.Frame(config_frame)
        self.yolo_model_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        self.yolo_model_frame.columnconfigure(1, weight=1)

        ttk.Label(self.yolo_model_frame, text="YOLO 模型:").grid(row=0, column=0, sticky=tk.W, pady=5)
        yolo_model_entry = ttk.Entry(self.yolo_model_frame, textvariable=self.yolo_model_path, width=50)
        yolo_model_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        ttk.Button(self.yolo_model_frame, text="浏览...", command=self.browse_yolo_model).grid(row=0, column=2, pady=5)

        # SAM 模型选择（初始隐藏）
        self.sam_model_frame = ttk.Frame(config_frame)
        # 不初始 grid，在切换时显示
        ttk.Label(self.sam_model_frame, text="SAM 模型:").grid(row=0, column=0, sticky=tk.W, pady=5)
        model_frame_inner = ttk.Frame(self.sam_model_frame)
        model_frame_inner.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)

        models = [
            ('vit_h (高精度)', 'vit_h'),
            ('vit_l (平衡)', 'vit_l'),
            ('vit_b (快速)', 'vit_b')
        ]
        for i, (text, value) in enumerate(models):
            ttk.Radiobutton(model_frame_inner, text=text, variable=self.model_type, value=value).grid(row=0, column=i, padx=10)

        # 设备选择
        ttk.Label(config_frame, text="运行设备:").grid(row=4, column=0, sticky=tk.W, pady=5)
        device_frame = ttk.Frame(config_frame)
        device_frame.grid(row=4, column=1, sticky=tk.W, padx=5, pady=5)

        devices = [('CUDA (GPU)', 'cuda'), ('CPU', 'cpu')]
        for i, (text, value) in enumerate(devices):
            ttk.Radiobutton(device_frame, text=text, variable=self.device, value=value).grid(row=0, column=i, padx=10)

        # 裁剪模式选择
        ttk.Label(config_frame, text="提取范围:").grid(row=5, column=0, sticky=tk.W, pady=5)
        crop_mode_frame = ttk.Frame(config_frame)
        crop_mode_frame.grid(row=5, column=1, sticky=tk.W, padx=5, pady=5)

        crop_modes = [
            ('仅人脸', 'face'),
            ('上半身 (推荐)', 'upper_body'),
            ('全身', 'full_body')
        ]
        for i, (text, value) in enumerate(crop_modes):
            ttk.Radiobutton(
                crop_mode_frame,
                text=text,
                variable=self.crop_mode,
                value=value
            ).grid(row=0, column=i, padx=8)

        # 高级设置按钮
        ttk.Button(config_frame, text="高级设置...", command=self.show_advanced_settings).grid(row=6, column=0, columnspan=3, pady=10)

        # === 操作按钮 ===
        button_frame = ttk.Frame(main_container)
        button_frame.grid(row=2, column=0, columnspan=3, pady=10)

        self.start_button = ttk.Button(button_frame, text="开始处理", command=self.start_processing, width=20)
        self.start_button.grid(row=0, column=0, padx=5)

        self.stop_button = ttk.Button(button_frame, text="停止", command=self.stop_processing, state=tk.DISABLED, width=20)
        self.stop_button.grid(row=0, column=1, padx=5)

        # === 进度条 ===
        progress_frame = ttk.Frame(main_container)
        progress_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        progress_frame.columnconfigure(0, weight=1)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=5)

        self.status_label = ttk.Label(progress_frame, text="就绪", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.grid(row=1, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)

        # === 日志输出 ===
        log_frame = ttk.LabelFrame(main_container, text="处理日志", padding="10")
        log_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        main_container.rowconfigure(4, weight=1)

        # 日志文本框
        self.log_text = scrolledtext.ScrolledText(log_frame, height=15, wrap=tk.WORD)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 清空日志按钮
        ttk.Button(log_frame, text="清空日志", command=self.clear_log).grid(row=1, column=0, pady=5)

        # 底部信息
        info_frame = ttk.Frame(main_container)
        info_frame.grid(row=5, column=0, columnspan=3, pady=5)

        ttk.Label(info_frame, text="提示: YOLO 快速推荐，SAM 高精度可选").grid(row=0, column=0, sticky=tk.W)

        # 下载模型按钮（仅用于 SAM）
        ttk.Button(info_frame, text="下载 SAM 模型", command=self.download_model).grid(row=0, column=1, padx=5)

        # 初始化显示状态（所有组件创建完成后）
        self.on_detection_method_change()

    def browse_input(self):
        """浏览输入目录"""
        directory = filedialog.askdirectory(title="选择漫画文件夹")
        if directory:
            self.input_dir.set(directory)

    def browse_output(self):
        """浏览输出目录"""
        directory = filedialog.askdirectory(title="选择输出文件夹")
        if directory:
            self.output_dir.set(directory)

    def browse_yolo_model(self):
        """浏览 YOLO 模型文件"""
        filename = filedialog.askopenfilename(
            title="选择 YOLO 模型文件",
            filetypes=[
                ("PyTorch 模型", "*.pt"),
                ("ONNX 模型", "*.onnx"),
                ("所有文件", "*.*")
            ]
        )
        if filename:
            self.yolo_model_path.set(filename)

    def on_detection_method_change(self):
        """检测方法切换事件"""
        method = self.detection_method.get()
        if method == 'yolo':
            # 显示 YOLO 模型路径，隐藏 SAM 模型选择
            self.yolo_model_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
            self.sam_model_frame.grid_remove()
            # 只在 log_text 已创建时记录日志
            if hasattr(self, 'log_text'):
                self.log(f"切换到 YOLO 检测方法（快速）")
        else:  # sam
            # 显示 SAM 模型选择，隐藏 YOLO 模型路径
            self.yolo_model_frame.grid_remove()
            self.sam_model_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
            # 只在 log_text 已创建时记录日志
            if hasattr(self, 'log_text'):
                self.log(f"切换到 SAM 检测方法（高精度）")

    def show_advanced_settings(self):
        """显示高级设置对话框"""
        dialog = tk.Toplevel(self.root)
        dialog.title("高级设置")
        dialog.geometry("500x400")
        dialog.resizable(False, False)

        # 使对话框模态
        dialog.transient(self.root)
        dialog.grab_set()

        # 设置变量
        min_person_size = tk.IntVar(value=256)
        max_persons = tk.IntVar(value=10)
        edge_blur = tk.IntVar(value=3)
        crop_borders = tk.BooleanVar(value=True)
        padding = tk.IntVar(value=10)
        split_enabled = tk.BooleanVar(value=True)

        # 创建设置表单
        settings_frame = ttk.Frame(dialog, padding="20")
        settings_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 长条图分割
        ttk.Label(settings_frame, text="长条图分割", font=('Arial', 10, 'bold')).grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
        ttk.Checkbutton(settings_frame, text="启用长条图分割", variable=split_enabled).grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=2)

        ttk.Separator(settings_frame, orient=tk.HORIZONTAL).grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)

        # 人物检测
        ttk.Label(settings_frame, text="人物检测", font=('Arial', 10, 'bold')).grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))

        ttk.Label(settings_frame, text="人物最小尺寸:").grid(row=4, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(settings_frame, from_=64, to=1024, textvariable=min_person_size, width=10).grid(row=4, column=1, sticky=tk.W, pady=2)

        ttk.Label(settings_frame, text="单张最大检测人数:").grid(row=5, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(settings_frame, from_=1, to=50, textvariable=max_persons, width=10).grid(row=5, column=1, sticky=tk.W, pady=2)

        ttk.Separator(settings_frame, orient=tk.HORIZONTAL).grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)

        # 后处理
        ttk.Label(settings_frame, text="后处理", font=('Arial', 10, 'bold')).grid(row=7, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))

        ttk.Label(settings_frame, text="边缘羽化半径:").grid(row=8, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(settings_frame, from_=0, to=10, textvariable=edge_blur, width=10).grid(row=8, column=1, sticky=tk.W, pady=2)

        ttk.Checkbutton(settings_frame, text="裁剪白边", variable=crop_borders).grid(row=9, column=0, columnspan=2, sticky=tk.W, pady=2)

        ttk.Label(settings_frame, text="裁剪边距:").grid(row=10, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(settings_frame, from_=0, to=50, textvariable=padding, width=10).grid(row=10, column=1, sticky=tk.W, pady=2)

        # 按钮
        button_frame = ttk.Frame(dialog)
        button_frame.grid(row=1, column=0, pady=10)

        ttk.Button(button_frame, text="确定", command=dialog.destroy, width=10).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="取消", command=dialog.destroy, width=10).grid(row=0, column=1, padx=5)

    def download_model(self):
        """下载模型"""
        model = self.model_type.get()
        self.log(f"开始下载 SAM 模型: {model}")
        self.status_label.config(text=f"正在下载模型: {model}...")

        def download_thread():
            try:
                from src.detection.sam_wrapper import SAMWrapper
                checkpoint_path = SAMWrapper.download_model_static(model, "./models")
                self.log_queue.put(('download_success', f"模型下载成功: {checkpoint_path}"))
                self.log_queue.put(('status', "模型下载完成"))
            except Exception as e:
                self.log_queue.put(('error', f"模型下载失败: {e}"))
                self.log_queue.put(('status', "模型下载失败"))

        thread = threading.Thread(target=download_thread, daemon=True)
        thread.start()

    def start_processing(self):
        """开始处理"""
        # 验证输入
        if not self.input_dir.get():
            self.log("错误: 请选择输入目录")
            return

        if not Path(self.input_dir.get()).exists():
            self.log(f"错误: 输入目录不存在: {self.input_dir.get()}")
            return

        if not self.output_dir.get():
            self.output_dir.set("./output")

        # 更新界面状态
        self.processing = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.progress_var.set(0)
        self.log("=" * 50)
        self.log(f"开始处理: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"输入目录: {self.input_dir.get()}")
        self.log(f"输出目录: {self.output_dir.get()}")
        self.log(f"检测方法: {self.detection_method.get()}")
        self.log(f"提取范围: {self.crop_mode.get()}")
        if self.detection_method.get() == 'yolo':
            self.log(f"YOLO 模型: {self.yolo_model_path.get()}")
        else:
            self.log(f"SAM 模型: {self.model_type.get()}")
        self.log(f"运行设备: {self.device.get()}")

        # 启动处理线程
        self.worker_thread = threading.Thread(target=self.process_worker, daemon=True)
        self.worker_thread.start()

    def stop_processing(self):
        """停止处理"""
        if self.processing:
            self.processing = False
            self.log("正在停止...")
            self.status_label.config(text="正在停止...")

    def process_worker(self):
        """处理工作线程"""
        try:
            # 创建配置对象（GUI模式不需要严格验证）
            config = Config("config.yaml", strict_validation=False)
            config.set('input.directory', self.input_dir.get())
            config.set('output.directory', self.output_dir.get())
            config.set('sam.model_type', self.model_type.get())
            config.set('sam.device', self.device.get())

            # 设置检测方法
            config.set('detection.method', self.detection_method.get())
            if self.detection_method.get() == 'yolo':
                config.set('detection.yolo_model_path', self.yolo_model_path.get())

            # 设置裁剪模式
            config.set('detection.crop_mode', self.crop_mode.get())

            # 创建处理器
            processor = BatchProcessor(config)

            # 进度回调
            def progress_callback(current, total, message):
                if not self.processing:
                    return False
                percent = (current / total) * 100
                self.log_queue.put(('progress', percent))
                self.log_queue.put(('status', f"处理中: {current}/{total} - {Path(message).name}"))
                return True

            processor.progress_callback = progress_callback

            # 执行处理
            report = processor.process_directory()

            # 完成
            self.log_queue.put(('success', report))
            self.log_queue.put(('status', "处理完成"))
            self.log_queue.put(('progress', 100))

        except Exception as e:
            self.log_queue.put(('error', f"处理失败: {e}"))
            self.log_queue.put(('status', "处理失败"))

        finally:
            self.log_queue.put(('finish', None))

    def update_log(self):
        """更新日志显示"""
        try:
            while True:
                event_type, data = self.log_queue.get_nowait()

                if event_type == 'log':
                    self.log(data)

                elif event_type == 'progress':
                    self.progress_var.set(data)

                elif event_type == 'status':
                    self.status_label.config(text=data)

                elif event_type == 'success':
                    report = data
                    self.log("=" * 50)
                    self.log("处理完成摘要:")
                    self.log(f"  总图像数: {report['summary']['total_images']}")
                    self.log(f"  成功处理: {report['summary']['processed_images']}")
                    self.log(f"  长条图分割: {report['summary']['split_strips']}")
                    self.log(f"  检测人物: {report['summary']['detected_persons']}")
                    self.log(f"  保存图像: {report['summary']['saved_images']}")
                    self.log(f"  失败图像: {report['summary']['failed_images']}")
                    self.log(f"  成功率: {report['summary']['success_rate']:.1%}")
                    self.log("=" * 50)

                elif event_type == 'error':
                    self.log(f"错误: {data}")

                elif event_type == 'finish':
                    self.processing = False
                    self.start_button.config(state=tk.NORMAL)
                    self.stop_button.config(state=tk.DISABLED)

                elif event_type == 'download_success':
                    self.log(data)

        except queue.Empty:
            pass

        # 定期更新
        self.root.after(100, self.update_log)

    def log(self, message):
        """添加日志消息"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)

    def clear_log(self):
        """清空日志"""
        self.log_text.delete(1.0, tk.END)


def main():
    """主函数"""
    root = tk.Tk()
    app = MangaSplitterGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
