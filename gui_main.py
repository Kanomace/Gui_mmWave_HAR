# gui_main.py
# ------------------------------------------
# 完整版本：带 DBSCAN 聚类可视化 + 原始功能（串口采集、实时推理、文件监视）
# - 延迟注册 pointcloud_notifier 回调，避免 AttributeError
# - 支持 payload(dict) 与 file path (str)
# ------------------------------------------

import os
import sys
import threading
import time
import queue
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import serial
import serial.tools.list_ports
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import datetime
import numpy as np
import matplotlib

# Use TkAgg backend for embedding in Tkinter
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Import notifier
from pointcloud_notifier import register_callback

# Project-specific imports (assumed present)
from voxelization import monitor_folder as voxel_monitor_folder
from projection import monitor_voxel_folder as projection_monitor_voxel_folder
from projection import create_single_projection
from voxelization import process_single_excel_file
from main import (
    RealTimeBuffer, load_model,
    GRID_SIZE, BOUNDARIES, WINDOW_SIZE, SEQ_LEN, BATCH_SIZE,
)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")


# ---------------- Utility ----------------
def list_serial_ports():
    """返回可用串口列表"""
    return [p.device for p in serial.tools.list_ports.comports()]


def random_color():
    """返回随机 RGB 颜色元组"""
    return tuple(np.random.rand(3))


# ----------------- Radar Dumper -----------------
class RadarToXlsxDumper:
    """
    雷达数据采集器：通过 gui_parser.uartParser 读取并保存 .xlsx/.bin
    （此类保持原有行为，不做可视化职责）
    """

    def __init__(self, xlsx_out_dir,
                 device_type="IWR6843",
                 demo_type="SDK Out of Box Demo",
                 log_fn=print,
                 cfg_file=None):
        self.xlsx_out_dir = xlsx_out_dir
        os.makedirs(self.xlsx_out_dir, exist_ok=True)

        self.device_type = device_type
        self.demo_type = demo_type
        self.parser = None
        self._stop_event = threading.Event()
        self.thread = None
        self.log = log_fn
        self.cfg_file = cfg_file

        self.cli_com = None
        self.data_com = None

    def start(self):
        try:
            # 延迟导入以避免循环依赖
            from gui_parser import uartParser

            self.parser = uartParser(
                type=self.demo_type,
                out_bin_dir=self.xlsx_out_dir.replace("xlsx", "bin"),
                out_xlsx_dir=self.xlsx_out_dir
            )
            self.parser.setSaveBinary(1)

            if self.cli_com and self.data_com:
                self.parser.connectComPorts(self.cli_com, self.data_com)
                self.log(f"[RadarDumper] Connected CLI={self.cli_com}, DATA={self.data_com}")
            elif self.cli_com:
                self.parser.connectComPort(self.cli_com)
                self.log(f"[RadarDumper] Connected single port CLI={self.cli_com}")
            else:
                raise RuntimeError("[RadarDumper] COM ports not specified")

            if self.cfg_file and os.path.isfile(self.cfg_file):
                with open(self.cfg_file, "r", encoding="utf-8", errors="ignore") as fh:
                    lines = [ln.strip() for ln in fh if ln.strip()]
                self.parser.sendCfg(lines)
                self.log(f"[RadarDumper] Sent cfg: {os.path.basename(self.cfg_file)}")

            self._stop_event.clear()
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
            self.log("[RadarDumper] started")

        except Exception as e:
            self.log(f"[RadarDumper] start failed: {e}")
            raise

    def stop(self):
        self._stop_event.set()
        if self.thread:
            self.thread.join(timeout=1.0)
        try:
            if self.parser and hasattr(self.parser, "close"):
                self.parser.close()
        except Exception:
            pass
        self.log("[RadarDumper] stopped")

    def _run(self):
        last_print = 0
        while not self._stop_event.is_set():
            try:
                if self.parser.parserType == "DoubleCOMPort":
                    _ = self.parser.readAndParseUartDoubleCOMPort()
                elif self.parser.parserType == "SingleCOMPort":
                    _ = self.parser.readAndParseUartSingleCOMPort()
                else:
                    self.log("[RadarDumper] unsupported parser type")
                    break

                if time.time() - last_print > 5:
                    self.log(f"[RadarDumper] running... (device={self.device_type}, demo={self.demo_type})")
                    last_print = time.time()

            except Exception as e:
                self.log(f"[RadarDumper] error in loop: {e}")
                time.sleep(0.1)


# ----------------- Realtime Pipeline Controller -----------------
class RealtimePipelineController:
    def __init__(self, log_fn, result_fn):
        self.log = log_fn
        self.push_result = result_fn

        self.model = None
        self.device = None
        self.buffer = None
        self.stop_event = threading.Event()

        self.threads = []
        self.infer_thread = None

        self.data_root = ""
        self.realtime_xlsx_dir = ""
        self.realtime_voxel_dir = ""
        self.realtime_xoz_dir = ""
        self.realtime_yoz_dir = ""
        self.realtime_xoy_dir = ""
        self.output_dir = ""
        self.realtime_bin_dir = ""  # 新增 bin 文件夹

        self.checkpoint_path = ""
        self.rope_informer_dir = ""
        self.cfg_file = ""

        self.results_acc = []

        self.radar_dumper = None
        self.auto_start_radar = False
        self.radar_device_type = "IWR6843"
        self.cli_com = None
        self.data_com = None

    def set_paths(self, data_root, ckpt_path, rope_dir, cfg_file):
        self.data_root = data_root
        # 生成带时间戳的输出主文件夹
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_root = os.path.join(data_root, timestamp)
        os.makedirs(self.output_root, exist_ok=True)

        # 所有实时文件夹放在 output_root 下
        self.realtime_xlsx_dir = os.path.join(self.output_root, "xlsx")
        self.realtime_bin_dir = os.path.join(self.output_root, "bin")
        self.realtime_voxel_dir = os.path.join(self.output_root, "pHistBytes_voxel")
        self.realtime_xoz_dir = os.path.join(self.output_root, "pHistBytes_voxel_XOZ")
        self.realtime_yoz_dir = os.path.join(self.output_root, "pHistBytes_voxel_YOZ")
        self.realtime_xoy_dir = os.path.join(self.output_root, "pHistBytes_voxel_XOY")
        self.output_dir = os.path.join(self.output_root, "results")

        self.checkpoint_path = ckpt_path
        self.rope_informer_dir = rope_dir
        self.cfg_file = cfg_file

    def _ensure_dirs(self):
        for d in [
            self.realtime_xlsx_dir,
            self.realtime_bin_dir,
            self.realtime_voxel_dir,
            self.realtime_xoz_dir,
            self.realtime_yoz_dir,
            self.realtime_xoy_dir,
            self.output_dir,
        ]:
            os.makedirs(d, exist_ok=True)

    def _load_model(self):
        if self.rope_informer_dir and self.rope_informer_dir not in sys.path:
            sys.path.append(self.rope_informer_dir)
        import main as main_mod
        main_mod.CHECKPOINT_PATH = self.checkpoint_path
        self.log("Loading model...")
        self.model, self.device = load_model()
        self.log("Model loaded.")

    def _start_file_watchers(self):
        def start_voxel_monitor():
            try:
                voxel_monitor_folder(
                    self.realtime_xlsx_dir,
                    self.realtime_voxel_dir,
                    GRID_SIZE,
                    BOUNDARIES
                )
            except Exception as e:
                self.log(f"[Error] voxel monitor: {e}")

        def start_projection_monitor():
            try:
                projection_monitor_voxel_folder(
                    self.realtime_voxel_dir,
                    self.realtime_yoz_dir,
                    self.realtime_xoy_dir,
                    self.realtime_xoz_dir
                )
            except Exception as e:
                self.log(f"[Error] projection monitor: {e}")

        t1 = threading.Thread(target=start_voxel_monitor, daemon=True)
        t2 = threading.Thread(target=start_projection_monitor, daemon=True)
        t1.start()
        t2.start()
        self.threads.extend([t1, t2])
        self.log("Started voxel and projection monitors.")

        class ImageFileHandler(FileSystemEventHandler):
            def __init__(self, buffer, logger):
                self.buffer = buffer
                self.logger = logger
                self.processed_images = set()

            def on_created(self, event):
                if not event.is_directory and (event.src_path.endswith('.png') or event.src_path.endswith('.jpg')):
                    if event.src_path in self.processed_images:
                        return
                    if "XOZ" in event.src_path:
                        view = "xoz"
                    elif "YOZ" in event.src_path:
                        view = "yoz"
                    else:
                        return
                    last_size = -1
                    for _ in range(10):
                        try:
                            current_size = os.path.getsize(event.src_path)
                            if current_size == last_size and current_size > 0:
                                self.buffer.add_image(event.src_path, view)
                                self.processed_images.add(event.src_path)
                                self.logger(f"New {view.upper()} image: {os.path.basename(event.src_path)}")
                                break
                            last_size = current_size
                            time.sleep(0.05)
                        except OSError:
                            time.sleep(0.05)

        self.image_observers = []
        for folder in [self.realtime_xoz_dir, self.realtime_yoz_dir]:
            handler = ImageFileHandler(self.buffer, self.log)
            obs = Observer()
            obs.schedule(handler, folder, recursive=False)
            obs.start()
            self.image_observers.append(obs)
            self.log(f"Watching image folder: {folder}")

    def _stop_file_watchers(self):
        if hasattr(self, "image_observers"):
            for obs in self.image_observers:
                obs.stop()
            for obs in self.image_observers:
                obs.join()
        self.log("Stopped image folder observers.")

    def _inference_loop(self):
        from torch.utils.data import DataLoader
        import torch
        from main import RealTimeDataset, id2label, SEQ_LEN, BATCH_SIZE, OUTPUT_DIR
        import main as main_mod
        main_mod.OUTPUT_DIR = self.output_dir

        dataset = RealTimeDataset(self.buffer, SEQ_LEN)

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
        last_sample_name = None

        self.log("Realtime inference loop started.")
        while not self.stop_event.is_set():
            try:
                if self.buffer.has_new_data() and len(dataset) > 0:
                    loader = DataLoader(
                        dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=False,
                        num_workers=0,
                        collate_fn=collate_with_meta
                    )

                    with torch.no_grad():
                        for x, y, meta in loader:
                            if self.stop_event.is_set():
                                break

                            if meta[0]["sample_name"] == last_sample_name:
                                continue

                            x = x.to(self.device)
                            logits = self.model(x) if callable(self.model) else self.model.forward(x)
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

                            if result["sample_name"] != "no_data":
                                results.append(result)
                                last_sample_name = result["sample_name"]

                                self.log(f"样本: {result['sample_name']} | 行为: {result['pred_label']} | 置信度: {result['confidence']:.4f}")
                                self.push_result(result)

                                df = pd.DataFrame(results)
                                csv_path = os.path.join(self.output_dir, "realtime_results.csv")
                                df.to_csv(csv_path, index=False, encoding="utf-8-sig")

                time.sleep(0.1)
            except Exception as e:
                self.log(f"[Error] inference loop: {e}")
                time.sleep(0.5)

        self.log("Realtime inference loop stopped.")


    # ---------------- Radar dumper ----------------
    def start_radar_dumper(self):
        self.radar_dumper = RadarToXlsxDumper(
            self.realtime_xlsx_dir,
            device_type=self.radar_device_type,
            log_fn=self.log,
            cfg_file=self.cfg_file
        )
        if self.cli_com:
            self.radar_dumper.cli_com = self.cli_com
        if self.data_com:
            self.radar_dumper.data_com = self.data_com

        self.radar_dumper.start()
        self.log("[Controller] Radar dumper started.")

    def stop_radar_dumper(self):
        if self.radar_dumper:
            self.radar_dumper.stop()
            self.radar_dumper = None
            self.log("[Controller] Radar dumper stopped.")


    # ---------------- Pipeline Control ----------------
    def start(self):
        self._ensure_dirs()

        if not os.path.isfile(self.checkpoint_path):
            raise ValueError("模型 checkpoint 不存在")

        if self.infer_thread and self.infer_thread.is_alive():
            self.log("推理已在运行中。")
            return

        self.stop_event.clear()
        self.buffer = RealTimeBuffer(window_size=WINDOW_SIZE)

        if self.auto_start_radar:
            self.start_radar_dumper()

        self._load_model()
        self._start_file_watchers()

        self.infer_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.infer_thread.start()

        self.log("Pipeline started.")

    def stop(self):
        self.stop_event.set()
        self._stop_file_watchers()
        if self.infer_thread:
            self.infer_thread.join(timeout=2.0)
        try:
            self.stop_radar_dumper()
        except Exception:
            pass
        self.log("Pipeline stopped.")


# ----------------- GUI Application -----------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("mmWave HAR 实时推理 - 简易UI")
        self.geometry("1200x800")

        # --- state vars ---
        self.data_root = tk.StringVar()
        self.ckpt_path = tk.StringVar()
        self.rope_dir = tk.StringVar()
        self.cfg_file = tk.StringVar()
        self.radar_enable = tk.BooleanVar(value=True)
        self.radar_device = tk.StringVar(value="IWR6843")
        self.cli_com_var = tk.StringVar()
        self.data_com_var = tk.StringVar()
        self.log_queue = queue.Queue()

        # queue 用于跨线程接收 pointcloud_notifier 的通知（线程安全）
        self.pointcloud_queue = queue.Queue()

        # controller 用于启动 pipeline
        self.controller = RealtimePipelineController(
            log_fn=self._log,
            result_fn=self._push_result_row
        )

        # build UI first
        self._build_ui()
        # load config
        self._load_config()

        # 延迟注册回调，确保 Tk 完全初始化后再注册
        self.after(100, lambda: register_callback(self._on_new_pointcloud))

        # 定时取出日志显示
        self.after(100, self._drain_log_queue)
        # 定时处理点云队列（在主线程绘制）
        self.after(100, self._process_pointcloud_queue)

    # ---------------- UI 构建 ----------------
    def _build_ui(self):
        pad = {"padx": 6, "pady": 4}

        # -------- 顶部路径和雷达设置 --------
        frm_top = ttk.LabelFrame(self, text="路径设置")
        frm_top.pack(fill="x", **pad)

        def add_path_row(parent, label, var, select_dir=True, filetypes=None):
            row = ttk.Frame(parent)
            row.pack(fill="x", **pad)
            ttk.Label(row, text=label, width=22).pack(side="left")
            entry = ttk.Entry(row, textvariable=var)
            entry.pack(side="left", fill="x", expand=True, padx=4)

            def choose():
                if select_dir:
                    p = filedialog.askdirectory()
                else:
                    p = filedialog.askopenfilename(filetypes=filetypes or [("All files", "*.*")])
                if p:
                    var.set(p)

            ttk.Button(row, text="选择", command=choose).pack(side="left")
            return row

        add_path_row(frm_top, "数据根目录", self.data_root, select_dir=True)
        add_path_row(frm_top, "模型 checkpoint", self.ckpt_path, select_dir=False, filetypes=[("pt 文件", "*.pt")])
        add_path_row(frm_top, "ROPE 模型目录", self.rope_dir, select_dir=True)
        add_path_row(frm_top, "雷达配置文件", self.cfg_file, select_dir=False, filetypes=[("cfg 文件", "*.cfg")])

        frm_radar = ttk.LabelFrame(self, text="雷达设置")
        frm_radar.pack(fill="x", **pad)
        ttk.Checkbutton(frm_radar, text="启用雷达", variable=self.radar_enable).pack(side="left", **pad)
        ttk.Label(frm_radar, text="设备型号").pack(side="left")
        ttk.Entry(frm_radar, textvariable=self.radar_device, width=15).pack(side="left", padx=4)
        ttk.Label(frm_radar, text="CLI COM").pack(side="left")
        ttk.Combobox(frm_radar, textvariable=self.cli_com_var, values=list_serial_ports(), width=10).pack(side="left")
        ttk.Label(frm_radar, text="DATA COM").pack(side="left")
        ttk.Combobox(frm_radar, textvariable=self.data_com_var, values=list_serial_ports(), width=10).pack(side="left")

        frm_btn = ttk.Frame(self)
        frm_btn.pack(fill="x", **pad)
        ttk.Button(frm_btn, text="启动", command=self._on_start).pack(side="left", padx=4)
        ttk.Button(frm_btn, text="停止", command=self._on_stop).pack(side="left", padx=4)
        ttk.Button(frm_btn, text="保存配置", command=self._save_config).pack(side="left", padx=4)

        # -------- 主体区域：左日志 + 右点云 --------
        frm_main = ttk.Frame(self)
        frm_main.pack(fill="both", expand=True, **pad)

        # 左侧：日志和结果
        frm_left = ttk.Frame(frm_main)
        frm_left.pack(side="left", fill="both", expand=True, padx=4, pady=4)

        frm_log = ttk.LabelFrame(frm_left, text="日志输出")
        frm_log.pack(fill="both", expand=True, **pad)
        self.txt_log = tk.Text(frm_log, height=12)
        self.txt_log.pack(fill="both", expand=True)

        frm_res = ttk.LabelFrame(frm_left, text="实时推理结果")
        frm_res.pack(fill="both", expand=True, **pad)
        self.tree = ttk.Treeview(frm_res, columns=("sample", "label", "confidence"), show="headings")
        self.tree.heading("sample", text="样本")
        self.tree.heading("label", text="行为")
        self.tree.heading("confidence", text="置信度")
        self.tree.pack(fill="both", expand=True)

        # 右侧：点云显示
        frm_right = ttk.LabelFrame(frm_main, text="点云显示 (X-Y-Z)")
        frm_right.pack(side="right", fill="both", expand=True, padx=4, pady=4)

        fig = Figure(figsize=(5, 5))
        self.ax = fig.add_subplot(111, projection="3d")
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_zlabel("Z (m)")
        self.canvas = FigureCanvasTkAgg(fig, master=frm_right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    # ---------------- Config ----------------
    def _save_config(self):
        cfg = {
            "data_root": self.data_root.get(),
            "ckpt_path": self.ckpt_path.get(),
            "rope_dir": self.rope_dir.get(),
            "cfg_file": self.cfg_file.get(),
            "radar_enable": self.radar_enable.get(),
            "radar_device": self.radar_device.get(),
            "cli_com": self.cli_com_var.get(),
            "data_com": self.data_com_var.get()
        }
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
            self._log("[Config] 保存完成")
        except Exception as e:
            self._log(f"[Config] 保存失败: {e}")

    def _load_config(self):
        if os.path.isfile(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                self.data_root.set(cfg.get("data_root", ""))
                self.ckpt_path.set(cfg.get("ckpt_path", ""))
                self.rope_dir.set(cfg.get("rope_dir", ""))
                self.cfg_file.set(cfg.get("cfg_file", ""))
                self.radar_enable.set(cfg.get("radar_enable", True))
                self.radar_device.set(cfg.get("radar_device", "IWR6843"))
                self.cli_com_var.set(cfg.get("cli_com", ""))
                self.data_com_var.set(cfg.get("data_com", ""))
                self._log("[Config] 加载完成")
            except Exception as e:
                self._log(f"[Config] 加载失败: {e}")

    # ---------------- Events ----------------
    def _on_start(self):
        try:
            self.controller.set_paths(
                self.data_root.get(),
                self.ckpt_path.get(),
                self.rope_dir.get(),
                self.cfg_file.get()
            )
            self.controller.auto_start_radar = self.radar_enable.get()
            self.controller.radar_device_type = self.radar_device.get()
            self.controller.cli_com = self.cli_com_var.get()
            self.controller.data_com = self.data_com_var.get()
            self.controller.start()
            self.status_var = "状态: 已启动"
            self._log("[GUI] Pipeline started")
        except Exception as e:
            self._log(f"[Error] 启动失败: {e}")

    def _on_stop(self):
        try:
            self.controller.stop()
            self._log("[GUI] Pipeline stopped")
        except Exception as e:
            self._log(f"[Error] 停止失败: {e}")

    # ---------------- Logging ----------------
    def _log(self, msg):
        # put to queue to avoid threading issues
        self.log_queue.put(msg)

    def _drain_log_queue(self):
        while not self.log_queue.empty():
            msg = self.log_queue.get_nowait()
            try:
                self.txt_log.insert("end", msg + "\n")
                self.txt_log.see("end")
            except Exception:
                # If widget not ready, ignore
                pass
        self.after(100, self._drain_log_queue)

    # ---------------- Result ----------------
    def _push_result_row(self, result):
        try:
            self.tree.insert("", "end", values=(result["sample_name"], result["pred_label"], f"{result['confidence']:.4f}"))
        except Exception:
            pass

    # ---------------- Pointcloud queue handling ----------------
    def _on_new_pointcloud(self, data):
        """
        回调入口（由 pointcloud_notifier 调用）
        data: str (file path) 或 dict (聚类 payload)
        我们只做入队处理，实际绘制在主线程的 _process_pointcloud_queue 中执行
        """
        try:
            self.pointcloud_queue.put(data)
        except Exception as e:
            self._log(f"[PointCloud] enqueue error: {e}")

    def _process_pointcloud_queue(self):
        """
        Called periodically on the main thread to process queued pointcloud file paths or payloads.
        """
        try:
            while not self.pointcloud_queue.empty():
                data = self.pointcloud_queue.get_nowait()

                if isinstance(data, str):
                    # 原始点云文件路径
                    self._update_pointcloud_view(data)
                elif isinstance(data, dict) and data.get("type") == "clustered":
                    # 聚类 payload
                    self._draw_clustered_pointcloud(data)
                else:
                    # 兼容旧版：如果回调传入的是路径字符串封装的其他类型
                    if isinstance(data, (bytes, bytearray)):
                        # 忽略
                        continue
        except Exception as e:
            self._log(f"[PointCloud] processing error: {e}")
        finally:
            self.after(100, self._process_pointcloud_queue)

    def _update_pointcloud_view(self, file_path):
        """
        Load the saved xlsx file and draw 3D scatter with SNR color mapping.
        This runs on the Tk main thread.
        """
        try:
            if not os.path.isfile(file_path):
                self._log(f"[PointCloud] 文件不存在: {file_path}")
                return

            df = pd.read_excel(file_path)
            # support different case column names
            cols = {c.lower(): c for c in df.columns}
            if not all(k in cols for k in ("x", "y", "z")):
                self._log(f"[PointCloud] 文件缺少 X/Y/Z 列: {file_path}")
                return

            xcol = cols["x"]
            ycol = cols["y"]
            zcol = cols["z"]
            snr_col = cols.get("snr", None)

            xs = df[xcol].to_numpy(dtype=float)
            ys = df[ycol].to_numpy(dtype=float)
            zs = df[zcol].to_numpy(dtype=float)

            # build colors from SNR if present, else use height-based color
            if snr_col:
                snr = df[snr_col].to_numpy(dtype=float)
                if np.nanmax(snr) - np.nanmin(snr) > 1e-6:
                    norm = (snr - np.nanmin(snr)) / (np.nanmax(snr) - np.nanmin(snr))
                else:
                    norm = np.clip(snr, 0, 1)
                cmap = matplotlib.cm.get_cmap("jet")
                colors = cmap(norm)
            else:
                if np.nanmax(zs) - np.nanmin(zs) > 1e-6:
                    normz = (zs - np.nanmin(zs)) / (np.nanmax(zs) - np.nanmin(zs))
                else:
                    normz = np.zeros_like(zs)
                cmap = matplotlib.cm.get_cmap("viridis")
                colors = cmap(normz)

            self.ax.cla()
            self.ax.scatter(xs, ys, zs, c=colors, s=6)
            self.ax.set_xlabel('X (m)')
            self.ax.set_ylabel('Y (m)')
            self.ax.set_zlabel('Z (m)')
            self.ax.set_title(os.path.basename(file_path))
            try:
                margin = 0.5
                self.ax.set_xlim(np.nanmin(xs)-margin, np.nanmax(xs)+margin)
                self.ax.set_ylim(np.nanmin(ys)-margin, np.nanmax(ys)+margin)
                self.ax.set_zlim(np.nanmin(zs)-margin, np.nanmax(zs)+margin)
            except Exception:
                pass
            self.canvas.draw()
            self._log(f"[PointCloud] 显示: {os.path.basename(file_path)}")
        except Exception as e:
            self._log(f"[PointCloud] 更新失败: {e}")

    def _draw_clustered_pointcloud(self, payload):
        """
        显示聚类后的彩色点云
        payload: {
            "type": "clustered",
            "path": "...",
            "clusters": [ ndarray-like lists ],
            "noise": [ ndarray-like lists ]
        }
        """
        try:
            clusters = payload.get("clusters", [])
            noise = payload.get("noise", [])
            cluster_path = payload.get("path", "clustered")

            self.ax.cla()

            for i, cluster in enumerate(clusters):
                pts = np.array(cluster)
                if pts.ndim != 2 or pts.shape[1] < 3:
                    continue
                color = random_color()
                self.ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=12, color=color, label=f"Cluster {i+1}")

            if len(noise) > 0:
                noise_pts = np.array(noise)
                if noise_pts.ndim == 2 and noise_pts.shape[1] >= 3:
                    self.ax.scatter(noise_pts[:, 0], noise_pts[:, 1], noise_pts[:, 2], s=6, color="gray", label="Noise")

            # legend might be empty if too many clusters; safe guard
            try:
                self.ax.legend(loc="upper right", fontsize=8)
            except Exception:
                pass

            self.ax.set_xlabel("X (m)")
            self.ax.set_ylabel("Y (m)")
            self.ax.set_zlabel("Z (m)")
            self.ax.set_title(f"DBSCAN Clustered - {os.path.basename(cluster_path)}")
            self.ax.set_box_aspect([1, 1, 1])
            self.canvas.draw()

            self._log(f"[GUI] 显示聚类结果: {os.path.basename(cluster_path)} ({len(clusters)} clusters)")
        except Exception as e:
            self._log(f"[GUI] 聚类显示失败: {e}")


# ----------------- Main Entrypoint -----------------
if __name__ == "__main__":
    app = App()

    # 正确注册回调（把 app 的方法注入到 notifier）
    try:
        register_callback(app._on_new_pointcloud)
        app._log("[Notifier] GUI callback registered.")
    except Exception as e:
        app._log(f"[Notifier] 注册失败: {e}")

    app.mainloop()
