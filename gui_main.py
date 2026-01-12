# gui_main_multitrack.py
# ================================================================
# Multi-person Tracking Enhanced Version
# Features:
#   1. Unique ID assignment for each tracked person
#   2. Frame-to-frame association using Hungarian algorithm
#   3. Independent trajectory history for each person
#   4. Different colors for different people's trajectories
# ================================================================

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
import datetime
import warnings
from scipy.optimize import linear_sum_assignment
import numpy as np
import cv2

from pointcloud_notifier import register_callback
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk

from camera_with_pose import CameraManager


warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")


# ============================================================
# Multi-Target Tracker Class
# ============================================================
class MultiTargetTracker:
    """多目标追踪器 - 为每个人分配唯一ID并维护轨迹"""

    def __init__(self, max_distance=2.0, max_age=30):
        """
        Args:
            max_distance: 最大匹配距离（米），超过此距离认为是新目标
            max_age: 最大丢失帧数，超过此数认为目标消失
        """
        self.tracks = {}  # {track_id: Track object}
        self.next_id = 0
        self.max_distance = max_distance
        self.max_age = max_age

        # 预定义颜色列表（最多10个人）
        self.colors = [
            'red', 'blue', 'green', 'orange', 'purple',
            'cyan', 'magenta', 'yellow', 'brown', 'pink'
        ]

    def update(self, detections):
        """
        更新追踪器
        Args:
            detections: List of detection dicts, each with keys: 'cx', 'cy', 'cz', 'box'
        Returns:
            updated_tracks: Dict of {track_id: track_info}
        """
        # Step 1: 预测所有现有轨迹的位置（简单：使用上一帧位置）
        predicted_positions = []
        track_ids = []

        for tid, track in self.tracks.items():
            if not track.is_dead(self.max_age):
                predicted_positions.append([track.positions[-1][0], track.positions[-1][1]])
                track_ids.append(tid)

        # Step 2: 如果没有检测，增加所有轨迹的age
        if len(detections) == 0:
            for track in self.tracks.values():
                track.age += 1
            return self._get_active_tracks()

        # Step 3: 计算检测与预测位置的距离矩阵
        detection_positions = np.array([[d['cx'], d['cy']] for d in detections])

        if len(predicted_positions) == 0:
            # 没有现有轨迹，所有检测都是新目标
            for det in detections:
                self._create_track(det)
        else:
            predicted_positions = np.array(predicted_positions)

            # 计算距离矩阵
            cost_matrix = self._compute_distance_matrix(
                detection_positions, predicted_positions
            )

            # Step 4: 使用匈牙利算法进行匹配
            row_indices, col_indices = linear_sum_assignment(cost_matrix)

            # Step 5: 处理匹配结果
            matched_detections = set()
            matched_tracks = set()

            for det_idx, track_idx in zip(row_indices, col_indices):
                distance = cost_matrix[det_idx, track_idx]

                if distance < self.max_distance:
                    # 匹配成功，更新轨迹
                    tid = track_ids[track_idx]
                    self.tracks[tid].update(detections[det_idx])
                    matched_detections.add(det_idx)
                    matched_tracks.add(tid)

            # Step 6: 创建新轨迹（未匹配的检测）
            for det_idx, det in enumerate(detections):
                if det_idx not in matched_detections:
                    self._create_track(det)

            # Step 7: 增加未匹配轨迹的age
            for tid in track_ids:
                if tid not in matched_tracks:
                    self.tracks[tid].age += 1

        # Step 8: 删除过期轨迹
        dead_tracks = [tid for tid, track in self.tracks.items()
                       if track.is_dead(self.max_age)]
        for tid in dead_tracks:
            del self.tracks[tid]

        return self._get_active_tracks()

    def _compute_distance_matrix(self, detections, predictions):
        """计算检测与预测之间的欧氏距离矩阵"""
        n_det = len(detections)
        n_pred = len(predictions)
        cost_matrix = np.zeros((n_det, n_pred))

        for i in range(n_det):
            for j in range(n_pred):
                diff = detections[i] - predictions[j]
                cost_matrix[i, j] = np.sqrt(diff[0] ** 2 + diff[1] ** 2)

        return cost_matrix

    def _create_track(self, detection):
        """创建新轨迹"""
        track_id = self.next_id
        self.next_id += 1

        color = self.colors[track_id % len(self.colors)]
        self.tracks[track_id] = Track(track_id, detection, color)

    def _get_active_tracks(self):
        """获取所有活跃轨迹"""
        active = {}
        for tid, track in self.tracks.items():
            if not track.is_dead(self.max_age):
                active[tid] = {
                    'id': tid,
                    'position': track.positions[-1] if track.positions else None,
                    'trajectory': track.get_trajectory(),
                    'color': track.color,
                    'age': track.age,
                    'box': track.box
                }
        return active


class Track:
    """单个目标的轨迹"""

    def __init__(self, track_id, detection, color):
        self.id = track_id
        self.color = color
        self.positions = [(detection['cx'], detection['cy'], time.time())]
        self.box = detection['box']
        self.age = 0  # 未匹配的帧数
        self.max_history = 100  # 最大轨迹历史点数

    def update(self, detection):
        """更新轨迹"""
        self.positions.append((detection['cx'], detection['cy'], time.time()))
        self.box = detection['box']
        self.age = 0  # 重置age

        # 限制历史长度
        if len(self.positions) > self.max_history:
            self.positions.pop(0)

    def is_dead(self, max_age):
        """判断轨迹是否过期"""
        return self.age > max_age

    def get_trajectory(self):
        """获取轨迹点列表"""
        return self.positions


# ============================================================
# Helper Functions (keep same as original)
# ============================================================
def list_serial_ports():
    return [p.device for p in serial.tools.list_ports.comports()]


class RadarToXlsxDumper:
    """Same as original - no changes needed"""

    def __init__(self, xlsx_out_dir, device_type="IWR6843",
                 demo_type="SDK Out of Box Demo", log_fn=print, cfg_file=None):
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
                self.log(f"[RadarDumper] Config sent: {os.path.basename(self.cfg_file)}")

            self._stop_event.clear()
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
            self.log("[RadarDumper] Started")
        except Exception as e:
            self.log(f"[RadarDumper] Start failed: {e}")
            raise

    def stop(self):
        self._stop_event.set()
        if self.thread:
            self.thread.join(timeout=1.0)
        try:
            if self.parser and hasattr(self.parser, 'cliCom'):
                if hasattr(self.parser.cliCom, 'close'):
                    self.parser.cliCom.close()
            if self.parser and hasattr(self.parser, 'dataCom'):
                if hasattr(self.parser.dataCom, 'close'):
                    self.parser.dataCom.close()
        except Exception:
            pass
        self.log("[RadarDumper] Stopped")

    def _run(self):
        last_print = 0
        while not self._stop_event.is_set():
            try:
                if self.parser.parserType == "DoubleCOMPort":
                    _ = self.parser.readAndParseUartDoubleCOMPort()
                elif self.parser.parserType == "SingleCOMPort":
                    _ = self.parser.readAndParseUartSingleCOMPort()
                else:
                    self.log("[RadarDumper] Unsupported parser type")
                    break

                if time.time() - last_print > 5:
                    self.log(f"[RadarDumper] Running... (device={self.device_type})")
                    last_print = time.time()
            except Exception as e:
                self.log(f"[RadarDumper] Error in loop: {e}")
                time.sleep(0.1)


class LiteController:
    """Same as original - no changes needed"""

    def __init__(self, log_fn):
        self.log = log_fn
        self.stop_event = threading.Event()
        self.data_root = ""
        self.output_root = ""
        self.realtime_xlsx_dir = ""
        self.realtime_bin_dir = ""
        self.cfg_file = ""
        self.radar_dumper = None
        self.radar_device_type = "IWR6843"
        self.cli_com = None
        self.data_com = None

    def set_paths(self, data_root, cfg_file):
        self.data_root = data_root
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_root = os.path.join(data_root, timestamp)
        os.makedirs(self.output_root, exist_ok=True)
        self.realtime_xlsx_dir = os.path.join(self.output_root, "xlsx")
        self.realtime_bin_dir = os.path.join(self.output_root, "bin")
        self.cfg_file = cfg_file

    def _ensure_dirs(self):
        for d in [self.realtime_xlsx_dir, self.realtime_bin_dir]:
            os.makedirs(d, exist_ok=True)

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
        self.log("[Controller] Radar dumper started")

    def stop_radar_dumper(self):
        if self.radar_dumper:
            self.radar_dumper.stop()
            self.radar_dumper = None
            self.log("[Controller] Radar dumper stopped")

    def start(self):
        self._ensure_dirs()
        self.stop_event.clear()
        self.start_radar_dumper()
        self.log("[System] Lite pipeline started (Collection + Visualization + Tracking)")

    def stop(self):
        self.stop_event.set()
        try:
            self.stop_radar_dumper()
        except Exception:
            pass
        self.log("[System] Lite pipeline stopped")


# ============================================================
# Enhanced GUI Application with Multi-Target Tracking
# ============================================================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("mmWave Multi-Person Tracking System")
        self.geometry("1600x900")

        self.data_root = tk.StringVar()
        self.cfg_file = tk.StringVar()
        self.radar_enable = tk.BooleanVar(value=True)
        self.radar_device = tk.StringVar(value="IWR6843")
        self.cli_com_var = tk.StringVar()
        self.data_com_var = tk.StringVar()
        self.camera_enable = tk.BooleanVar(value=True)

        self.pointcloud_queue = queue.Queue(maxsize=0)

        # ✅ 使用多目标追踪器替代简单的轨迹历史
        self.tracker = MultiTargetTracker(max_distance=2.0, max_age=30)

        self.cluster_cache = {}
        self.cache_max_size = 5

        self.frame_count = 0
        self.fps_start_time = time.time()
        self.current_fps = 0

        self.camera_manager = CameraManager(width=640, height=480, fps=30)
        self.camera_rgb_image = None
        self.camera_depth_image = None

        self.controller = LiteController(log_fn=print)

        self._build_ui()
        self._load_config()

        self.after(50, self._process_pointcloud_queue)
        self.after(30, self._update_camera_display)
        self.after(1000, self._update_fps_display)

        try:
            register_callback(self._on_new_pointcloud)
            print("[Notifier] GUI callback registered")
        except Exception as e:
            print(f"[Notifier] Registration failed: {e}")

    def _build_ui(self):
        """Same as original but with updated title and button"""
        pad = {"padx": 6, "pady": 4}

        # Top: Path and Settings
        frm_top = ttk.LabelFrame(self, text="Path Settings")
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

            ttk.Button(row, text="Browse", command=choose).pack(side="left")
            return row

        add_path_row(frm_top, "Data Root Directory", self.data_root, select_dir=True)
        add_path_row(frm_top, "Radar Config File", self.cfg_file, select_dir=False,
                     filetypes=[("cfg files", "*.cfg")])

        # Device Settings
        frm_devices = ttk.LabelFrame(self, text="Device Settings")
        frm_devices.pack(fill="x", **pad)

        ttk.Checkbutton(frm_devices, text="Enable Radar", variable=self.radar_enable).pack(side="left", **pad)
        ttk.Label(frm_devices, text="Device Model").pack(side="left")
        ttk.Entry(frm_devices, textvariable=self.radar_device, width=15).pack(side="left", padx=4)
        ttk.Label(frm_devices, text="CLI COM").pack(side="left")
        ttk.Combobox(frm_devices, textvariable=self.cli_com_var,
                     values=list_serial_ports(), width=10).pack(side="left")
        ttk.Label(frm_devices, text="DATA COM").pack(side="left")
        ttk.Combobox(frm_devices, textvariable=self.data_com_var,
                     values=list_serial_ports(), width=10).pack(side="left")

        ttk.Separator(frm_devices, orient='vertical').pack(side="left", fill='y', padx=10)

        ttk.Checkbutton(frm_devices, text="Enable Camera", variable=self.camera_enable).pack(side="left", **pad)

        # Control buttons
        frm_btn = ttk.Frame(self)
        frm_btn.pack(fill="x", **pad)
        ttk.Button(frm_btn, text="Start", command=self._on_start).pack(side="left", padx=4)
        ttk.Button(frm_btn, text="Stop", command=self._on_stop).pack(side="left", padx=4)
        ttk.Button(frm_btn, text="Save Config", command=self._save_config).pack(side="left", padx=4)
        ttk.Button(frm_btn, text="Clear Tracks", command=self._clear_tracks).pack(side="left", padx=4)

        ttk.Separator(frm_btn, orient='vertical').pack(side="left", fill='y', padx=10)
        self.fps_label = ttk.Label(frm_btn, text="FPS: 0.0", font=("Arial", 10, "bold"))
        self.fps_label.pack(side="left", padx=10)

        # ✅ 显示追踪目标数量
        self.track_count_label = ttk.Label(frm_btn, text="Tracks: 0",
                                           font=("Arial", 10, "bold"), foreground="blue")
        self.track_count_label.pack(side="left", padx=10)

        # Main area
        frm_main = ttk.Frame(self)
        frm_main.pack(fill="both", expand=True, **pad)

        # Left: Camera Feed
        frm_left = ttk.LabelFrame(frm_main, text="Camera Feed")
        frm_left.pack(side="left", fill="both", expand=False, padx=4, pady=4)
        frm_left.configure(width=400)
        frm_left.pack_propagate(False)

        camera_container = ttk.Frame(frm_left)
        camera_container.pack(fill="both", expand=True)

        rgb_frame = ttk.LabelFrame(camera_container, text="RGB Image", padding=2)
        rgb_frame.pack(fill="both", expand=True, pady=2)
        self.rgb_label = tk.Label(rgb_frame, bg='black', text="Waiting...",
                                  fg='gray', font=('Arial', 10))
        self.rgb_label.pack(fill="both", expand=True)

        depth_frame = ttk.LabelFrame(camera_container, text="Depth Image", padding=2)
        depth_frame.pack(fill="both", expand=True, pady=2)
        self.depth_label = tk.Label(depth_frame, bg='black', text="Waiting...",
                                    fg='gray', font=('Arial', 10))
        self.depth_label.pack(fill="both", expand=True)

        # Center: 3D Point Cloud
        frm_center = ttk.LabelFrame(frm_main, text="3D Point Cloud (X-Y-Z)")
        frm_center.pack(side="left", fill="both", expand=True, padx=4, pady=4)

        fig_3d = Figure(figsize=(6, 6), dpi=80)
        self.ax_3d = fig_3d.add_subplot(111, projection="3d")
        self.ax_3d.set_xlabel("X (m)")
        self.ax_3d.set_ylabel("Y (m)")
        self.ax_3d.set_zlabel("Z (m)")
        self.ax_3d.set_xlim(-4, 4)
        self.ax_3d.set_ylim(0, 8)
        self.ax_3d.set_zlim(-4, 4)
        self.canvas_3d = FigureCanvasTkAgg(fig_3d, master=frm_center)
        self.canvas_3d.get_tk_widget().pack(fill="both", expand=True)

        # Right: Multi-Person XY Trajectories
        frm_right = ttk.LabelFrame(frm_main, text="Multi-Person XY Trajectories")
        frm_right.pack(side="right", fill="both", expand=True, padx=4, pady=4)

        fig_xy = Figure(figsize=(5, 6), dpi=80)
        self.ax_xy = fig_xy.add_subplot(111)
        self.ax_xy.set_xlabel("X (m)")
        self.ax_xy.set_ylabel("Y (m)")
        self.ax_xy.set_xlim(-4, 4)
        self.ax_xy.set_ylim(0, 8)
        self.ax_xy.grid(True, alpha=0.3)
        self.ax_xy.set_aspect('equal')
        self.canvas_xy = FigureCanvasTkAgg(fig_xy, master=frm_right)
        self.canvas_xy.get_tk_widget().pack(fill="both", expand=True)

    def _update_camera_display(self):
        """Same as original"""
        try:
            if self.camera_manager.is_active():
                rgb, depth = self.camera_manager.get_latest_frames()

                if rgb is not None and depth is not None:
                    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                    depth = cv2.cvtColor(depth, cv2.COLOR_BGR2RGB)
                    rgb_width = self.rgb_label.winfo_width()
                    rgb_height = self.rgb_label.winfo_height()
                    depth_width = self.depth_label.winfo_width()
                    depth_height = self.depth_label.winfo_height()

                    if rgb_width <= 1:
                        rgb_width = 380
                    if rgb_height <= 1:
                        rgb_height = 300
                    if depth_width <= 1:
                        depth_width = 380
                    if depth_height <= 1:
                        depth_height = 300

                    h, w = rgb.shape[:2]
                    scale_rgb = min(rgb_width / w, rgb_height / h)
                    new_w_rgb = int(w * scale_rgb)
                    new_h_rgb = int(h * scale_rgb)
                    rgb_resized = Image.fromarray(rgb).resize((new_w_rgb, new_h_rgb), Image.LANCZOS)
                    rgb_photo = ImageTk.PhotoImage(rgb_resized)

                    h, w = depth.shape[:2]
                    scale_depth = min(depth_width / w, depth_height / h)
                    new_w_depth = int(w * scale_depth)
                    new_h_depth = int(h * scale_depth)
                    depth_resized = Image.fromarray(depth).resize((new_w_depth, new_h_depth), Image.LANCZOS)
                    depth_photo = ImageTk.PhotoImage(depth_resized)

                    self.rgb_label.configure(image=rgb_photo, text="")
                    self.rgb_label.image = rgb_photo

                    self.depth_label.configure(image=depth_photo, text="")
                    self.depth_label.image = depth_photo

        except Exception as e:
            print(f"[Camera Display] Error: {e}")
        finally:
            self.after(30, self._update_camera_display)

    def _save_config(self):
        """Same as original"""
        cfg = {
            "data_root": self.data_root.get(),
            "cfg_file": self.cfg_file.get(),
            "radar_enable": self.radar_enable.get(),
            "radar_device": self.radar_device.get(),
            "cli_com": self.cli_com_var.get(),
            "data_com": self.data_com_var.get(),
            "camera_enable": self.camera_enable.get()
        }
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        print("[Config] Configuration saved")

    def _load_config(self):
        """Same as original"""
        if os.path.isfile(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                self.data_root.set(cfg.get("data_root", ""))
                self.cfg_file.set(cfg.get("cfg_file", ""))
                self.radar_enable.set(cfg.get("radar_enable", True))
                self.radar_device.set(cfg.get("radar_device", "IWR6843"))
                self.cli_com_var.set(cfg.get("cli_com", ""))
                self.data_com_var.set(cfg.get("data_com", ""))
                self.camera_enable.set(cfg.get("camera_enable", True))
                print("[Config] Configuration loaded")
            except Exception as e:
                print(f"[Config] Load failed: {e}")

    def _on_start(self):
        """Same as original"""
        try:
            self.controller.set_paths(self.data_root.get(), self.cfg_file.get())
            self.controller.radar_device_type = self.radar_device.get()
            self.controller.cli_com = self.cli_com_var.get()
            self.controller.data_com = self.data_com_var.get()
            self.controller.start()

            if self.camera_enable.get():
                try:
                    self.camera_manager.start()
                    print("[Camera] Camera started successfully")
                except Exception as e:
                    print(f"[Camera] Failed to start: {e}")

            self.frame_count = 0
            self.fps_start_time = time.time()
            print("[System] System started (Radar + Camera)")
        except Exception as e:
            print(f"[Error] Start failed: {e}")

    def _on_stop(self):
        """Same as original"""
        try:
            self.controller.stop()
            self.camera_manager.stop()
            print("[System] System stopped")
        except Exception as e:
            print(f"[Error] Stop failed: {e}")

    def _clear_tracks(self):
        """✅ 清除所有追踪轨迹"""
        self.tracker = MultiTargetTracker(max_distance=2.0, max_age=30)
        self.cluster_cache.clear()

        self.ax_xy.clear()
        self.ax_xy.set_xlabel("X (m)")
        self.ax_xy.set_ylabel("Y (m)")
        self.ax_xy.set_xlim(-4, 4)
        self.ax_xy.set_ylim(0, 8)
        self.ax_xy.grid(True, alpha=0.3)
        self.ax_xy.set_aspect('equal')
        self.canvas_xy.draw_idle()

        self.track_count_label.config(text="Tracks: 0")
        print("[Tracking] All tracks cleared")

    def _update_fps_display(self):
        """Same as original"""
        elapsed = time.time() - self.fps_start_time
        if elapsed > 0:
            self.current_fps = self.frame_count / elapsed
            self.fps_label.config(text=f"FPS: {self.current_fps:.1f}")
        self.after(1000, self._update_fps_display)

    def _on_new_pointcloud(self, payload):
        """Same as original"""
        try:
            if isinstance(payload, tuple) and len(payload) >= 3:
                file_path = payload[0]
                cluster_path = payload[1] if len(payload) > 1 else None
                filtered_data = payload[2] if len(payload) > 2 else None
                track_positions = payload[3] if len(payload) > 3 else []

                self.pointcloud_queue.put({
                    'file_path': file_path,
                    'cluster_path': cluster_path,
                    'filtered_data': filtered_data,
                    'track_positions': track_positions,
                    'timestamp': time.time()
                })
            else:
                print(f"[PointCloud] Unknown payload format: {type(payload)}")
        except Exception as e:
            print(f"[PointCloud] Callback error: {e}")

    def _process_pointcloud_queue(self):
        """Same as original"""
        try:
            if not self.pointcloud_queue.empty():
                data = self.pointcloud_queue.get_nowait()
                threading.Thread(
                    target=self._process_pointcloud_async,
                    args=(data,),
                    daemon=True
                ).start()
        except queue.Empty:
            pass
        except Exception as e:
            print(f"[PointCloud] Queue processing error: {e}")
        finally:
            self.after(50, self._process_pointcloud_queue)

    def _process_pointcloud_async(self, data):
        """✅ 改进：提取检测信息用于追踪"""
        try:
            file_path = data['file_path']
            cluster_path = data['cluster_path']
            filtered_data = data['filtered_data']

            # Prepare cluster data
            cluster_boxes = []
            detections = []  # ✅ 新增：用于追踪的检测列表

            if cluster_path and os.path.isfile(cluster_path):
                if cluster_path in self.cluster_cache:
                    cluster_boxes = self.cluster_cache[cluster_path]
                else:
                    try:
                        xl = pd.ExcelFile(cluster_path)
                        for sheet_name in xl.sheet_names:
                            if "Cluster" not in sheet_name or sheet_name == "Empty":
                                continue

                            cdf = xl.parse(sheet_name)
                            if not {'X', 'Y', 'Z'}.issubset(cdf.columns) or len(cdf) == 0:
                                continue

                            xmin, xmax = cdf['X'].min(), cdf['X'].max()
                            ymin, ymax = cdf['Y'].min(), cdf['Y'].max()
                            zmin, zmax = cdf['Z'].min(), cdf['Z'].max()

                            box = {
                                'xmin': xmin, 'xmax': xmax,
                                'ymin': ymin, 'ymax': ymax,
                                'zmin': zmin, 'zmax': zmax,
                                'label': sheet_name.split("_")[-1]
                            }
                            cluster_boxes.append(box)

                            # ✅ 创建检测对象
                            detections.append({
                                'cx': (xmin + xmax) / 2,
                                'cy': (ymin + ymax) / 2,
                                'cz': (zmin + zmax) / 2,
                                'box': box
                            })

                        self.cluster_cache[cluster_path] = cluster_boxes
                        if len(self.cluster_cache) > self.cache_max_size:
                            oldest = list(self.cluster_cache.keys())[0]
                            del self.cluster_cache[oldest]
                    except Exception as e:
                        print(f"[Cluster] Read failed: {e}")

            # Switch back to main thread
            self.after(0, lambda: self._update_visualization(
                file_path, filtered_data, detections
            ))

        except Exception as e:
            print(f"[AsyncProcess] Error: {e}")

    def _update_visualization(self, file_path, filtered_data, detections):
        """✅ 改进：使用多目标追踪器更新可视化"""
        try:
            # ✅ 更新追踪器
            active_tracks = self.tracker.update(detections)

            # ✅ 更新追踪数量显示
            self.track_count_label.config(text=f"Tracks: {len(active_tracks)}")

            # ==================== 3D Point Cloud ====================
            self.ax_3d.clear()

            # Draw point cloud
            if filtered_data is not None and len(filtered_data) > 0:
                xs = filtered_data[:, 0]
                ys = filtered_data[:, 1]
                zs = filtered_data[:, 2]
                snr = filtered_data[:, 4] if filtered_data.shape[1] >= 5 else np.zeros_like(xs)

                if np.ptp(snr) > 1e-6:
                    norm_snr = (snr - np.min(snr)) / (np.ptp(snr) + 1e-6)
                else:
                    norm_snr = np.zeros_like(snr)

                if len(xs) > 300:
                    step = len(xs) // 300
                    xs, ys, zs, norm_snr = xs[::step], ys[::step], zs[::step], norm_snr[::step]

                self.ax_3d.scatter(xs, ys, zs, c=norm_snr, cmap='jet', s=6, alpha=0.7)

            # ✅ 绘制每个追踪目标的边界框（用追踪颜色）
            for track_id, track_info in active_tracks.items():
                box = track_info['box']
                color = track_info['color']

                xmin, xmax = box['xmin'], box['xmax']
                ymin, ymax = box['ymin'], box['ymax']
                zmin, zmax = box['zmin'], box['zmax']

                cx = (xmin + xmax) / 2
                cy = (ymin + ymax) / 2
                cz = (zmin + zmax) / 2

                # Draw bounding box with track color
                x = [xmin, xmax, xmax, xmin, xmin, xmax, xmax, xmin]
                y = [ymin, ymin, ymax, ymax, ymin, ymin, ymax, ymax]
                z = [zmin, zmin, zmin, zmin, zmax, zmax, zmax, zmax]

                edges = [
                    (0, 1), (1, 2), (2, 3), (3, 0),
                    (4, 5), (5, 6), (6, 7), (7, 4),
                    (0, 4), (1, 5), (2, 6), (3, 7)
                ]

                for e0, e1 in edges:
                    self.ax_3d.plot(
                        [x[e0], x[e1]],
                        [y[e0], y[e1]],
                        [z[e0], z[e1]],
                        color=color, linewidth=1.5
                    )

                # ✅ 显示追踪ID
                self.ax_3d.text(cx, cy, cz, f'ID{track_id}',
                                color=color, fontsize=10, weight='bold')

            self.ax_3d.set_xlabel('X (m)', fontsize=9)
            self.ax_3d.set_ylabel('Y (m)', fontsize=9)
            self.ax_3d.set_zlabel('Z (m)', fontsize=9)
            self.ax_3d.set_xlim(-4, 4)
            self.ax_3d.set_ylim(0, 8)
            self.ax_3d.set_zlim(-4, 4)
            self.ax_3d.set_box_aspect([1, 1, 1])
            self.ax_3d.set_title(os.path.basename(file_path) if file_path else "Realtime", fontsize=10)

            self.canvas_3d.draw_idle()

            # ==================== Multi-Person XY Trajectories ====================
            self._update_multi_person_trajectories(active_tracks)

            self.frame_count += 1

        except Exception as e:
            print(f"[Visualization] Update failed: {e}")

    def _update_multi_person_trajectories(self, active_tracks):
        """✅ 新增：绘制多人轨迹（每个人独立颜色）"""
        try:
            self.ax_xy.clear()

            self.ax_xy.set_xlabel("X (m)", fontsize=9)
            self.ax_xy.set_ylabel("Y (m)", fontsize=9)
            self.ax_xy.set_xlim(-4, 4)
            self.ax_xy.set_ylim(0, 8)
            self.ax_xy.grid(True, alpha=0.3)
            self.ax_xy.set_aspect('equal')

            if len(active_tracks) == 0:
                self.ax_xy.set_title("Multi-Person Trajectories (No Tracks)", fontsize=10)
                self.canvas_xy.draw_idle()
                return

            current_time = time.time()

            # ✅ 为每个追踪目标绘制独立轨迹
            for track_id, track_info in active_tracks.items():
                trajectory = track_info['trajectory']
                color = track_info['color']

                if len(trajectory) == 0:
                    continue

                xs = [p[0] for p in trajectory]
                ys = [p[1] for p in trajectory]
                timestamps = [p[2] for p in trajectory]

                # Calculate time-based alpha
                alphas = [max(0.2, 1.0 - (current_time - ts) / 30.0) for ts in timestamps]

                # Downsample if too many points
                if len(xs) > 50:
                    step = max(1, len(xs) // 50)
                    xs_plot = xs[::step]
                    ys_plot = ys[::step]
                    alphas_plot = alphas[::step]
                else:
                    xs_plot, ys_plot, alphas_plot = xs, ys, alphas

                # ✅ 绘制轨迹线（使用追踪颜色）
                for i in range(len(xs_plot) - 1):
                    self.ax_xy.plot(
                        xs_plot[i:i + 2], ys_plot[i:i + 2],
                        color=color,
                        alpha=alphas_plot[i] * 0.8,
                        linewidth=2.0
                    )

                # ✅ 绘制轨迹点
                self.ax_xy.scatter(xs_plot, ys_plot,
                                   color=color, s=30, alpha=0.6,
                                   edgecolors='black', linewidths=0.5)

                # ✅ 标注当前位置和ID
                if len(xs) > 0:
                    self.ax_xy.scatter(xs[-1], ys[-1],
                                       color=color, s=150, marker='*',
                                       edgecolors='black', linewidths=2, zorder=10)
                    self.ax_xy.text(xs[-1] + 0.2, ys[-1] + 0.2,
                                    f'ID{track_id}',
                                    fontsize=10, color=color, weight='bold',
                                    bbox=dict(boxstyle='round,pad=0.3',
                                              facecolor='white', alpha=0.7))

            self.ax_xy.set_title(f"Multi-Person Trajectories ({len(active_tracks)} tracks)",
                                 fontsize=10)
            self.canvas_xy.draw_idle()

        except Exception as e:
            print(f"[Multi-Track] Update failed: {e}")

    def on_closing(self):
        """Same as original"""
        print("[System] Shutting down...")
        try:
            self.controller.stop()
            self.camera_manager.stop()
        except Exception as e:
            print(f"[Shutdown] Error: {e}")
        finally:
            self.destroy()


if __name__ == "__main__":
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()