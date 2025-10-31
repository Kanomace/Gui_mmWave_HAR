import os
import sys
import glob
import json
import time
import threading
import queue
from typing import List, Tuple, Dict, Any
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# 导入自定义模块
from voxelization import process_single_excel_file, monitor_folder
from projection import create_single_projection, monitor_voxel_folder

# ============ 路径配置 ============
# 实时数据源（XLSX文件）
REALTIME_XLSX_DIR = r"D:\Ti\Gui_mmWave_HAR\data_realtime"

# 中间处理路径
REALTIME_VOXEL_DIR = os.path.join(REALTIME_XLSX_DIR, 'pHistBytes_clustered_voxel')
REALTIME_XOZ_DIR = os.path.join(REALTIME_VOXEL_DIR, 'pHistBytes_clustered_voxel_XOZ')
REALTIME_YOZ_DIR = os.path.join(REALTIME_VOXEL_DIR, 'pHistBytes_clustered_voxel_YOZ')

# 模型与代码路径
ROPE_INFORMER_DIR = r"D:\Ti\Gui_mmWave_HAR\rope_informer"
CHECKPOINT_PATH = r"D:\Ti\Gui_mmWave_HAR\model_checkpoint\checkpoint0914.pth"

# 输出
OUTPUT_DIR = r"D:\Ti\Gui_mmWave_HAR\realtime_inference_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============ 参数与类别映射 ============
behavior_mapping = {
    "stationary": 0,
    "run": 1,
    "squat": 2,
    "stand": 3,
    "walk": 4,
}
id2label = {v: k for k, v in behavior_mapping.items()}

# 体素化参数
GRID_SIZE = (25, 15, 25)  # (x, y, z)
BOUNDARIES = {
    'x': (-5, 5),  # X轴范围为 -5 到 5 米
    'y': (0, 6),  # Y轴范围为 0 到 6 米
    'z': (-5, 5)  # Z轴范围为 -5 到 5 米
}

# 视角图像尺寸
image_sizes = {
    "xoz": (25, 25),  # W, H
    "yoz": (25, 15),
}

# 滑窗与序列长度，与 Kaggle 训练保持一致
WINDOW_SIZE = 3
SEQ_LEN = 3000  # 3 * (25*25 + 25*15) = 3 * (625 + 375) = 3000
BATCH_SIZE = 1  # 实时推理使用小批量

# 是否有标签（实时推理通常没有标签）
has_labels = False

# ============ 导入 rope_informer 代码 ============
if ROPE_INFORMER_DIR not in sys.path:
    sys.path.append(ROPE_INFORMER_DIR)

try:
    from rope_informer import Exp_Informer
except Exception as e:
    raise ImportError(f"无法从 {ROPE_INFORMER_DIR} 导入 Exp_Informer，请检查路径与模块。原始错误：{e}")


# ============ 实时数据缓冲区 ============
class RealTimeBuffer:
    """实时数据缓冲区，用于存储最近的图像序列"""

    def __init__(self, window_size=3):
        self.window_size = window_size
        self.xoz_buffer = []  # 存储XOZ图像路径
        self.yoz_buffer = []  # 存储YOZ图像路径
        self.lock = threading.Lock()
        self.last_inference_time = 0  # 记录上次推理时间
        self.processed_images = set()  # 记录已处理的图像

    def add_image(self, image_path, view):
        """添加新图像到缓冲区"""
        with self.lock:
            # 检查图像是否已处理过
            if image_path in self.processed_images:
                return

            if view == "xoz":
                self.xoz_buffer.append(image_path)
                if len(self.xoz_buffer) > self.window_size:
                    self.xoz_buffer.pop(0)
            elif view == "yoz":
                self.yoz_buffer.append(image_path)
                if len(self.yoz_buffer) > self.window_size:
                    self.yoz_buffer.pop(0)

            self.processed_images.add(image_path)
            self.last_inference_time = time.time()  # 更新最后活动时间

    def get_window(self, view):
        """获取当前窗口"""
        with self.lock:
            if view == "xoz" and len(self.xoz_buffer) >= self.window_size:
                return self.xoz_buffer[-self.window_size:]
            elif view == "yoz" and len(self.yoz_buffer) >= self.window_size:
                return self.yoz_buffer[-self.window_size:]
            return None

    def has_new_data(self, timeout=2.0):
        """检查是否有新数据（在超时时间内）"""
        return (time.time() - self.last_inference_time) < timeout


# ============ 数据集定义（实时推理） ============
class RealTimeDataset(Dataset):
    """
    实时推理数据集，从缓冲区获取数据
    """

    def __init__(self, buffer, seq_len=3000):
        self.buffer = buffer
        self.seq_len = seq_len
        self.image_sizes = {
            "xoz": (25, 25),
            "yoz": (25, 15),
        }

    def __len__(self):
        # 实时数据集长度动态变化
        return 1 if (self.buffer.get_window("xoz") is not None or
                     self.buffer.get_window("yoz") is not None) else 0

    def _view_resize(self, img: Image.Image, view: str) -> Image.Image:
        size = self.image_sizes["xoz"] if view == "xoz" else self.image_sizes["yoz"]
        return img.resize(size)

    def _get_xlsx_files_from_image_paths(self, image_paths):
        """从图像路径获取对应的XLSX文件路径"""
        xlsx_files = []
        for img_path in image_paths:
            # 从图像文件名提取基本名称
            base_name = os.path.basename(img_path)
            # 去掉视图后缀和扩展名
            if "_XOZ" in base_name:
                base_name = base_name.replace("_XOZ", "")
            elif "_YOZ" in base_name:
                base_name = base_name.replace("_YOZ", "")
            base_name = os.path.splitext(base_name)[0]
            # 构建XLSX文件路径
            xlsx_path = os.path.join(REALTIME_XLSX_DIR, f"{base_name}.xlsx")
            xlsx_files.append(xlsx_path)
        return xlsx_files

    def __getitem__(self, idx):
        # 获取XOZ窗口
        xoz_paths = self.buffer.get_window("xoz")
        if xoz_paths:
            window_arrays = []
            for p in xoz_paths:
                img = Image.open(p).convert("L")
                img = self._view_resize(img, "xoz")
                arr = np.array(img, dtype=np.float32).flatten()
                window_arrays.append(arr)
            seq = np.concatenate(window_arrays, axis=0)
            # pad/trunc to seq_len
            if len(seq) < self.seq_len:
                pad = self.seq_len - len(seq)
                seq = np.pad(seq, (0, pad), mode="constant")
            elif len(seq) > self.seq_len:
                seq = seq[:self.seq_len]
            x = torch.from_numpy(seq).float()  # shape [seq_len]

            # 从文件名提取样本名称和时间
            sample_names = [os.path.basename(p).split('.')[0].replace("_XOZ", "") for p in xoz_paths]
            sample_name = ", ".join(sample_names)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 获取对应的XLSX文件
            xlsx_files = self._get_xlsx_files_from_image_paths(xoz_paths)

            meta = {
                "sample_name": sample_name,
                "timestamp": timestamp,
                "paths": xoz_paths,
                "xlsx_files": xlsx_files,
                "view": "xoz"
            }
            return x, torch.tensor(-1), meta

        # 获取YOZ窗口
        yoz_paths = self.buffer.get_window("yoz")
        if yoz_paths:
            window_arrays = []
            for p in yoz_paths:
                img = Image.open(p).convert("L")
                img = self._view_resize(img, "yoz")
                arr = np.array(img, dtype=np.float32).flatten()
                window_arrays.append(arr)
            seq = np.concatenate(window_arrays, axis=0)
            # pad/trunc to seq_len
            if len(seq) < self.seq_len:
                pad = self.seq_len - len(seq)
                seq = np.pad(seq, (0, pad), mode="constant")
            elif len(seq) > self.seq_len:
                seq = seq[:self.seq_len]
            x = torch.from_numpy(seq).float()  # shape [seq_len]

            # 从文件名提取样本名称和时间
            sample_names = [os.path.basename(p).split('.')[0].replace("_YOZ", "") for p in yoz_paths]
            sample_name = ", ".join(sample_names)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 获取对应的XLSX文件
            xlsx_files = self._get_xlsx_files_from_image_paths(yoz_paths)

            meta = {
                "sample_name": sample_name,
                "timestamp": timestamp,
                "paths": yoz_paths,
                "xlsx_files": xlsx_files,
                "view": "yoz"
            }
            return x, torch.tensor(-1), meta

        # 如果没有可用数据，返回空
        return torch.zeros(self.seq_len), torch.tensor(-1), {
            "sample_name": "no_data",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "paths": [],
            "xlsx_files": [],
            "view": "none"
        }


# ============ args 定义 ============
class Args:
    def __init__(self):
        self.model = 'informer'
        self.data = 'Classification'
        self.root_path = ''
        self.enc_in = 1
        self.d_model = 64
        self.d_ff = 256
        self.train_epochs = 1
        self.batch_size = BATCH_SIZE
        self.seq_len = SEQ_LEN
        self.output_path = OUTPUT_DIR
        self.checkpoints = OUTPUT_DIR
        self.test_ratio = 0.2
        self.n_heads = 8
        self.has_rope = True
        self.features = 'M'
        self.target = 'OT'
        self.freq = 'h'
        self.label_len = 48
        self.pred_len = 24
        self.dec_in = 1
        self.c_out = 5
        self.e_layers = 2
        self.d_layers = 1
        self.s_layers = '3,2,1'
        self.factor = 5
        self.padding = 0
        self.distil = True
        self.dropout = 0.05
        self.attn = 'prob'
        self.embed = 'timeF'
        self.activation = 'gelu'
        self.output_attention = False
        self.do_predict = True
        self.mix = True
        self.cols = None
        self.num_workers = 0
        self.itr = 1
        self.patience = 2
        self.learning_rate = 1e-4
        self.des = 'realtime_inference'
        self.loss = 'mse'
        self.lradj = 'type1'
        self.use_amp = False
        self.inverse = False
        self.use_gpu = torch.cuda.is_available()
        self.gpu = 0
        self.use_multi_gpu = False
        self.devices = '0'


args = Args()


# ============ 自定义实验类，只做推理 ============
class RealTimeInferenceExp(Exp_Informer):
    def _get_data(self, flag):
        # 实时推理不需要传统的数据加载方式
        return None, None


# ============ 加载模型 ============
def load_model():
    """加载预训练模型"""
    exp = RealTimeInferenceExp(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = getattr(exp, "model", None)
    if model is None:
        try:
            model = exp._build_model()
            exp.model = model
        except Exception as e:
            raise RuntimeError(f"无法构建模型，请检查 Exp_Informer 实现。错误：{e}")

    model = model.to(device)
    model.eval()

    def load_checkpoint_into_model(model: torch.nn.Module, ckpt_path: str):
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"checkpoint 不存在: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state_dict = state["state_dict"]
        elif isinstance(state, dict) and "model_state_dict" in state:
            state_dict = state["model_state_dict"]
        elif isinstance(state, dict):
            state_dict = state
        else:
            raise ValueError("未知的 checkpoint 格式，需包含 state_dict 或 model_state_dict。")
        new_state = {}
        for k, v in state_dict.items():
            nk = k.replace("module.", "") if k.startswith("module.") else k
            new_state[nk] = v
        missing, unexpected = model.load_state_dict(new_state, strict=False)
        if missing:
            print(f"Warning: 缺失权重: {missing}")
        if unexpected:
            print(f"Warning: 多余权重: {unexpected}")

    load_checkpoint_into_model(model, CHECKPOINT_PATH)
    return model, device


# ============ 实时推理函数 ============
def run_realtime_inference(model, device, buffer):
    """运行实时推理"""
    dataset = RealTimeDataset(buffer, SEQ_LEN)

    def collate_with_meta(batch):
        xs, ys, metas = [], [], []
        for x, y, meta in batch:
            xs.append(x)
            ys.append(y)
            metas.append(meta)
        xs = torch.stack(xs, dim=0) if xs else torch.zeros(1, SEQ_LEN)
        ys = torch.stack(ys, dim=0) if ys else torch.tensor([-1])
        return xs, ys, metas

    softmax = torch.nn.Softmax(dim=-1)
    results = []
    last_sample_name = None  # 记录上一次处理的样本名称

    while True:
        # 只在有新数据时才进行推理
        if buffer.has_new_data() and len(dataset) > 0:
            loader = DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_with_meta
            )

            with torch.no_grad():
                for x, y, meta in loader:
                    # 避免重复处理相同的样本
                    if meta[0]["sample_name"] == last_sample_name:
                        continue

                    x = x.to(device)
                    logits = model(x) if callable(model) else model.forward(x)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    probs = softmax(logits)
                    preds = torch.argmax(probs, dim=-1)

                    pred_label = id2label[int(preds.cpu().numpy()[0])]
                    confidence = probs.cpu().numpy()[0].max()

                    result = {
                        "sample_name": meta[0]["sample_name"],
                        "timestamp": meta[0]["timestamp"],
                        "pred_label": pred_label,
                        "confidence": float(confidence),
                        "xlsx_files": meta[0]["xlsx_files"]
                    }

                    # 只添加有效结果
                    if result["sample_name"] != "no_data":
                        results.append(result)

                        # 输出使用的XLSX文件
                        #xlsx_files_str = ", ".join(result["xlsx_files"])
                        #print(f"使用的XLSX文件: {xlsx_files_str}")

                        # 简化输出格式
                        print(
                            f"样本: {result['sample_name']}, 时间: {result['timestamp']}, 行为: {result['pred_label']}, 置信度: {result['confidence']:.4f}")

                        # 更新最后处理的样本名称
                        last_sample_name = result["sample_name"]

                        # 保存最新结果
                        df = pd.DataFrame(results)
                        csv_path = os.path.join(OUTPUT_DIR, "realtime_results.csv")
                        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        time.sleep(0.1)  # 更短的检查间隔


# ============ 文件监视器 ============
class ImageFileHandler(FileSystemEventHandler):
    """监视图像文件变化的处理器"""

    def __init__(self, buffer):
        self.buffer = buffer
        self.processed_images = set()  # 记录已处理的图像

    def on_created(self, event):
        if not event.is_directory and (event.src_path.endswith('.png') or event.src_path.endswith('.jpg')):
            # 检查图像是否已处理过
            if event.src_path in self.processed_images:
                return

            # 确定视图类型
            if "XOZ" in event.src_path:
                view = "xoz"
            elif "YOZ" in event.src_path:
                view = "yoz"
            else:
                return

            # 使用文件大小变化检测文件是否写完
            last_size = -1
            for _ in range(5):  # 最多检查5次
                try:
                    current_size = os.path.getsize(event.src_path)
                    if current_size == last_size and current_size > 0:
                        # 文件大小稳定，添加到缓冲区
                        self.buffer.add_image(event.src_path, view)
                        self.processed_images.add(event.src_path)
                        break
                    last_size = current_size
                    time.sleep(0.05)  # 短暂等待
                except OSError:
                    # 文件可能还在写入，继续等待
                    time.sleep(0.05)


def start_file_monitoring(buffer, image_folders):
    """启动文件监视"""
    observers = []

    for folder in image_folders:
        event_handler = ImageFileHandler(buffer)
        observer = Observer()
        observer.schedule(event_handler, folder, recursive=False)
        observer.start()
        observers.append(observer)
        print(f"开始监视图像文件夹: {folder}")

    return observers


# ============ 主函数 ============
def main():
    """主函数"""
    print("初始化实时推理系统...")

    # 1. 确保输出目录存在
    os.makedirs(REALTIME_VOXEL_DIR, exist_ok=True)
    os.makedirs(REALTIME_XOZ_DIR, exist_ok=True)
    os.makedirs(REALTIME_YOZ_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. 加载模型
    print("加载模型...")
    model, device = load_model()

    # 3. 创建数据缓冲区
    buffer = RealTimeBuffer(window_size=WINDOW_SIZE)

    # 4. 启动文件监视线程
    image_folders = [REALTIME_XOZ_DIR, REALTIME_YOZ_DIR]
    observers = start_file_monitoring(buffer, image_folders)

    # 5. 启动体素化和投影处理的监视器（在后台线程中）
    def start_voxel_monitor():
        monitor_folder(REALTIME_XLSX_DIR, REALTIME_VOXEL_DIR, GRID_SIZE, BOUNDARIES)

    def start_projection_monitor():
        monitor_voxel_folder(REALTIME_VOXEL_DIR, REALTIME_YOZ_DIR,
                             os.path.join(REALTIME_VOXEL_DIR, 'pHistBytes_clustered_voxel_XOY'),
                             REALTIME_XOZ_DIR)

    voxel_thread = threading.Thread(target=start_voxel_monitor, daemon=True)
    projection_thread = threading.Thread(target=start_projection_monitor, daemon=True)

    voxel_thread.start()
    projection_thread.start()

    print("实时推理系统已启动，等待数据输入...")
    print("按 Ctrl+C 停止系统")

    try:
        # 6. 启动实时推理
        run_realtime_inference(model, device, buffer)
    except KeyboardInterrupt:
        print("停止实时推理系统...")
        for observer in observers:
            observer.stop()
        for observer in observers:
            observer.join()


if __name__ == "__main__":
    main()