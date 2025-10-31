# gui_main.py

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

from pointcloud_notifier import register_callback
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# Matplotlib for 3D point cloud display
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

# Your existing modules (unchanged)
from voxelization import monitor_folder as voxel_monitor_folder
from projection import monitor_voxel_folder as projection_monitor_voxel_folder
from projection import create_single_projection
from voxelization import process_single_excel_file
from main import (
    RealTimeBuffer, load_model,
    GRID_SIZE, BOUNDARIES, WINDOW_SIZE, SEQ_LEN, BATCH_SIZE,
)

# Notifier for cross-module callbacks (must exist as file pointcloud_notifier.py)
from pointcloud_notifier import register_callback  # used in main to register callback

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")


def list_serial_ports():
    """返回可用串口列表"""
    return [p.device for p in serial.tools.list_ports.comports()]


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
            from gui_parser import uartParser

            self.parser = uartParser(
                type=self.demo_type,
                out_bin_dir=self.xlsx_out_dir.replace("xlsx", "bin"),  # 自动对应 bin 文件夹
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

        self.controller = RealtimePipelineController(
            log_fn=self._log,
            result_fn=self._push_result_row
        )

        # build UI first
        self._build_ui()
        # load config
        self._load_config()
        # 定时取出日志显示
        self.after(100, self._drain_log_queue)
        register_callback(self._on_new_pointcloud)
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

        # matplotlib 图
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        fig = plt.Figure(figsize=(5, 5))
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
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        self._log("[Config] 保存完成")

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
            self.status_var.set("状态: 已启动")
        except Exception as e:
            self._log(f"[Error] 启动失败: {e}")
            self.status_var.set("状态: 启动失败")

    def _on_stop(self):
        try:
            self.controller.stop()
            self.status_var.set("状态: 已停止")
        except Exception as e:
            self._log(f"[Error] 停止失败: {e}")

    # ---------------- Logging ----------------
    def _log(self, msg):
        # put to queue to avoid threading issues
        self.log_queue.put(msg)

    def _drain_log_queue(self):
        while not self.log_queue.empty():
            msg = self.log_queue.get_nowait()
            self.txt_log.insert("end", msg + "\n")
            self.txt_log.see("end")
        self.after(100, self._drain_log_queue)

    # ---------------- Result ----------------
    def _push_result_row(self, result):
        self.tree.insert("", "end", values=(result["sample_name"], result["pred_label"], f"{result['confidence']:.4f}"))

    # ---------------- Pointcloud queue handling ----------------
    def _enqueue_pointcloud(self, file_path):
        """
        This function is safe to be called from other threads (it's registered as callback).
        It only enqueues the file path for the main thread to process.
        """
        try:
            self.pointcloud_queue.put(file_path)
        except Exception as e:
            # put best-effort log
            self._log(f"[PointCloud] enqueue error: {e}")

    def _process_pointcloud_queue(self):
        """
        Called periodically on the main thread to process queued pointcloud file paths.
        """
        try:
            while not self.pointcloud_queue.empty():
                file_path = self.pointcloud_queue.get_nowait()
                # process and draw
                self._update_pointcloud_view(file_path)
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
            # look for common column name variants
            cols = {c.lower(): c for c in df.columns}
            if not all(k in cols for k in ("x", "y", "z")):
                self._log(f"[PointCloud] 文件缺少 X/Y/Z 列: {file_path}")
                return

            xcol = cols["x"]
            ycol = cols["y"]
            zcol = cols["z"]
            # snr optional
            snr_col = cols.get("snr", None)

            xs = df[xcol].to_numpy(dtype=float)
            ys = df[ycol].to_numpy(dtype=float)
            zs = df[zcol].to_numpy(dtype=float)

            # build colors from SNR if present, else use height-based color
            if snr_col:
                snr = df[snr_col].to_numpy(dtype=float)
                # normalize 0..1
                if np.nanmax(snr) - np.nanmin(snr) > 1e-6:
                    norm = (snr - np.nanmin(snr)) / (np.nanmax(snr) - np.nanmin(snr))
                else:
                    norm = np.clip(snr, 0, 1)
                cmap = matplotlib.cm.get_cmap("jet")
                colors = cmap(norm)
            else:
                # fallback: color by z
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
            # optionally adjust limits
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

    def _on_new_pointcloud(self, payload):
        """当有新的点云文件保存时（带聚类信息）"""
        try:
            if isinstance(payload, tuple):
                file_path, cluster_path = payload
            else:
                file_path, cluster_path = payload, None

            if not os.path.isfile(file_path):
                self._log(f"[PointCloud] 文件不存在: {file_path}")
                return

            df = pd.read_excel(file_path)
            if not {'X', 'Y', 'Z'}.issubset(df.columns):
                self._log(f"[PointCloud] 缺少坐标列: {file_path}")
                return

            self.after(0, lambda: self._update_pointcloud(df, file_path, cluster_path))

        except Exception as e:
            self._log(f"[PointCloud] 加载失败: {e}")

    def _update_pointcloud(self, df, file_path, cluster_path=None):
        try:
            self.ax.clear()

            # 计算 SNR 颜色归一化
            if 'SNR' in df.columns:
                norm_snr = (df['SNR'] - df['SNR'].min()) / (df['SNR'].max() - df['SNR'].min() + 1e-6)
            else:
                norm_snr = np.zeros(len(df))

            # 绘制点云
            self.ax.scatter(df['X'], df['Y'], df['Z'], c=norm_snr, cmap='jet', s=8)

            # 固定坐标范围为 ±2m
            self.ax.set_xlim(-4, 4)
            self.ax.set_ylim(0, 8)
            self.ax.set_zlim(-4, 4)
            self.ax.set_box_aspect([1, 1, 1])

            self.ax.set_xlabel('X (m)')
            self.ax.set_ylabel('Y (m)')
            self.ax.set_zlabel('Z (m)')
            self.ax.set_title(os.path.basename(file_path))

            # ========== 绘制 DBSCAN 聚类框 ==========
            if cluster_path and os.path.isfile(cluster_path):
                try:
                    xl = pd.ExcelFile(cluster_path)
                    for sheet_name in xl.sheet_names:
                        if "Cluster" not in sheet_name:
                            continue
                        cdf = xl.parse(sheet_name)
                        if not {'X', 'Y', 'Z'}.issubset(cdf.columns):
                            continue

                        xmin, xmax = cdf['X'].min(), cdf['X'].max()
                        ymin, ymax = cdf['Y'].min(), cdf['Y'].max()
                        zmin, zmax = cdf['Z'].min(), cdf['Z'].max()

                        # 绘制立方框
                        x = [xmin, xmax, xmax, xmin, xmin, xmax, xmax, xmin]
                        y = [ymin, ymin, ymax, ymax, ymin, ymin, ymax, ymax]
                        z = [zmin, zmin, zmin, zmin, zmax, zmax, zmax, zmax]
                        edges = [
                            (0, 1), (1, 2), (2, 3), (3, 0),
                            (4, 5), (5, 6), (6, 7), (7, 4),
                            (0, 4), (1, 5), (2, 6), (3, 7)
                        ]
                        for e0, e1 in edges:
                            self.ax.plot(
                                [x[e0], x[e1]],
                                [y[e0], y[e1]],
                                [z[e0], z[e1]],
                                color='red', linewidth=1.2
                            )
                except Exception as e:
                    self._log(f"[ClusterDraw] 绘制聚类框失败: {e}")

            self.canvas.draw()

        except Exception as e:
            print(f"[PointCloudView] Failed to update: {e}")


# ----------------- Main Entrypoint -----------------
if __name__ == "__main__":
    # create app
    app = App()

    # register the notifier callback to enqueue pointcloud file paths (thread-safe)
    try:
        # import here to avoid circular import issues in some environments
        from pointcloud_notifier import register_callback

        register_callback(self._enqueue_pointcloud)
        print("[DEBUG] GUI callback registered.")
        self.after(500, self._process_pointcloud_queue)

    except Exception as e:
        app._log(f"[Notifier] 注册失败: {e}")

    app.mainloop()
