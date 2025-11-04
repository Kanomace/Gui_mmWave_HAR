# gui_main_lite_en.py
# ================================================================
# Lite Version: Data Collection + Visualization + Tracking Only
# Removed: Voxelization, Projection, Behavior Inference
# Language: Full English
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

from pointcloud_notifier import register_callback
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

# ✅ Suppress all warnings
warnings.filterwarnings('ignore')

# ✅ Configure Matplotlib with English fonts only
import matplotlib.font_manager as fm

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")


def list_serial_ports():
    """Return available serial ports"""
    return [p.device for p in serial.tools.list_ports.comports()]


# ----------------- Radar Dumper -----------------
class RadarToXlsxDumper:
    """Radar data collector via gui_parser.uartParser"""

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


# ----------------- Lite Controller (No Inference) -----------------
class LiteController:
    """Lightweight controller: Manages radar collection only"""

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


# ----------------- GUI Application -----------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("mmWave Real-time Visualization - Lite Mode")
        self.geometry("1600x900")

        self.data_root = tk.StringVar()
        self.cfg_file = tk.StringVar()
        self.radar_enable = tk.BooleanVar(value=True)
        self.radar_device = tk.StringVar(value="IWR6843")
        self.cli_com_var = tk.StringVar()
        self.data_com_var = tk.StringVar()
        self.log_queue = queue.Queue()

        # Point cloud data queue (thread-safe)
        self.pointcloud_queue = queue.Queue(maxsize=0)

        # Trajectory history
        self.trajectory_history = []
        self.max_trajectory_points = 100

        # Cache Excel data
        self.cluster_cache = {}
        self.cache_max_size = 5

        # Frame rate statistics
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.current_fps = 0

        # ✅ Use lightweight controller
        self.controller = LiteController(log_fn=self._log)

        self._build_ui()
        self._load_config()

        # Scheduled tasks
        self.after(50, self._drain_log_queue)
        self.after(50, self._process_pointcloud_queue)
        self.after(1000, self._update_fps_display)

        # Register point cloud callback
        try:
            register_callback(self._on_new_pointcloud)
            self._log("[Notifier] GUI callback registered")
        except Exception as e:
            self._log(f"[Notifier] Registration failed: {e}")

    # ---------------- UI Construction ----------------
    def _build_ui(self):
        pad = {"padx": 6, "pady": 4}

        # -------- Top: Path and Radar Settings --------
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
        add_path_row(frm_top, "Radar Config File", self.cfg_file, select_dir=False, filetypes=[("cfg files", "*.cfg")])

        frm_radar = ttk.LabelFrame(self, text="Radar Settings")
        frm_radar.pack(fill="x", **pad)
        ttk.Checkbutton(frm_radar, text="Enable Radar", variable=self.radar_enable).pack(side="left", **pad)
        ttk.Label(frm_radar, text="Device Model").pack(side="left")
        ttk.Entry(frm_radar, textvariable=self.radar_device, width=15).pack(side="left", padx=4)
        ttk.Label(frm_radar, text="CLI COM").pack(side="left")
        ttk.Combobox(frm_radar, textvariable=self.cli_com_var, values=list_serial_ports(), width=10).pack(side="left")
        ttk.Label(frm_radar, text="DATA COM").pack(side="left")
        ttk.Combobox(frm_radar, textvariable=self.data_com_var, values=list_serial_ports(), width=10).pack(side="left")

        # ✅ Control buttons + FPS display (All English)
        frm_btn = ttk.Frame(self)
        frm_btn.pack(fill="x", **pad)
        ttk.Button(frm_btn, text="Start", command=self._on_start).pack(side="left", padx=4)
        ttk.Button(frm_btn, text="Stop", command=self._on_stop).pack(side="left", padx=4)
        ttk.Button(frm_btn, text="Save Config", command=self._save_config).pack(side="left", padx=4)
        ttk.Button(frm_btn, text="Clear Trajectory", command=self._clear_trajectory).pack(side="left", padx=4)

        ttk.Separator(frm_btn, orient='vertical').pack(side="left", fill='y', padx=10)
        self.fps_label = ttk.Label(frm_btn, text="FPS: 0.0", font=("Arial", 10, "bold"))
        self.fps_label.pack(side="left", padx=10)

        # -------- Main area: Left log + Center 3D + Right XY trajectory --------
        frm_main = ttk.Frame(self)
        frm_main.pack(fill="both", expand=True, **pad)

        # Left: Log
        frm_left = ttk.Frame(frm_main, width=400)
        frm_left.pack(side="left", fill="both", expand=False, padx=4, pady=4)
        frm_left.pack_propagate(False)

        frm_log = ttk.LabelFrame(frm_left, text="Log Output")
        frm_log.pack(fill="both", expand=True, **pad)
        self.txt_log = tk.Text(frm_log, height=30)
        self.txt_log.pack(fill="both", expand=True)

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

        # Right: XY Plane Trajectory
        frm_right = ttk.LabelFrame(frm_main, text="XY Trajectory View")
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

    # ---------------- Config ----------------
    def _save_config(self):
        cfg = {
            "data_root": self.data_root.get(),
            "cfg_file": self.cfg_file.get(),
            "radar_enable": self.radar_enable.get(),
            "radar_device": self.radar_device.get(),
            "cli_com": self.cli_com_var.get(),
            "data_com": self.data_com_var.get()
        }
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        self._log("[Config] Configuration saved")

    def _load_config(self):
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
                self._log("[Config] Configuration loaded")
            except Exception as e:
                self._log(f"[Config] Load failed: {e}")

    # ---------------- Events ----------------
    def _on_start(self):
        try:
            self.controller.set_paths(
                self.data_root.get(),
                self.cfg_file.get()
            )
            self.controller.radar_device_type = self.radar_device.get()
            self.controller.cli_com = self.cli_com_var.get()
            self.controller.data_com = self.data_com_var.get()
            self.controller.start()

            # Reset FPS statistics
            self.frame_count = 0
            self.fps_start_time = time.time()

            self._log("[System] System started (Lite Mode)")
        except Exception as e:
            self._log(f"[Error] Start failed: {e}")

    def _on_stop(self):
        try:
            self.controller.stop()
            self._log("[System] System stopped")
        except Exception as e:
            self._log(f"[Error] Stop failed: {e}")

    def _clear_trajectory(self):
        """Clear trajectory history"""
        self.trajectory_history.clear()
        self.cluster_cache.clear()
        self.ax_xy.clear()
        self.ax_xy.set_xlabel("X (m)")
        self.ax_xy.set_ylabel("Y (m)")
        self.ax_xy.set_xlim(-4, 4)
        self.ax_xy.set_ylim(0, 8)
        self.ax_xy.grid(True, alpha=0.3)
        self.ax_xy.set_aspect('equal')
        self.canvas_xy.draw_idle()
        self._log("[Trajectory] Trajectory cleared")

    # ---------------- FPS Display ----------------
    def _update_fps_display(self):
        """Update FPS display"""
        elapsed = time.time() - self.fps_start_time
        if elapsed > 0:
            self.current_fps = self.frame_count / elapsed
            self.fps_label.config(text=f"FPS: {self.current_fps:.1f}")
        self.after(1000, self._update_fps_display)

    # ---------------- Logging ----------------
    def _log(self, msg):
        self.log_queue.put(msg)

    def _drain_log_queue(self):
        """Periodically drain log queue"""
        count = 0
        while not self.log_queue.empty() and count < 10:
            try:
                msg = self.log_queue.get_nowait()
                self.txt_log.insert("end", msg + "\n")
                count += 1
            except queue.Empty:
                break
        self.txt_log.see("end")
        self.after(50, self._drain_log_queue)

    # ---------------- Pointcloud Callback ----------------
    def _on_new_pointcloud(self, payload):
        """Callback function (called in worker thread)"""
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
        """Main thread processes point cloud queue"""
        try:
            if not self.pointcloud_queue.empty():
                data = self.pointcloud_queue.get_nowait()

                # Process in background thread
                threading.Thread(
                    target=self._process_pointcloud_async,
                    args=(data,),
                    daemon=True
                ).start()

        except queue.Empty:
            pass
        except Exception as e:
            self._log(f"[PointCloud] Queue processing error: {e}")
        finally:
            self.after(50, self._process_pointcloud_queue)

    def _process_pointcloud_async(self, data):
        """Background thread: Read Excel"""
        try:
            file_path = data['file_path']
            cluster_path = data['cluster_path']
            filtered_data = data['filtered_data']

            # Prepare cluster data
            cluster_boxes = []
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

                            box = {
                                'xmin': cdf['X'].min(),
                                'xmax': cdf['X'].max(),
                                'ymin': cdf['Y'].min(),
                                'ymax': cdf['Y'].max(),
                                'zmin': cdf['Z'].min(),
                                'zmax': cdf['Z'].max(),
                                'label': sheet_name.split("_")[-1]
                            }
                            cluster_boxes.append(box)

                        self.cluster_cache[cluster_path] = cluster_boxes
                        if len(self.cluster_cache) > self.cache_max_size:
                            oldest = list(self.cluster_cache.keys())[0]
                            del self.cluster_cache[oldest]
                    except Exception as e:
                        print(f"[Cluster] Read failed: {e}")

            # Switch back to main thread for drawing
            self.after(0, lambda: self._update_visualization(
                file_path, filtered_data, cluster_boxes
            ))

        except Exception as e:
            print(f"[AsyncProcess] Error: {e}")

    def _update_visualization(self, file_path, filtered_data, cluster_boxes):
        """Main thread: Update visualization"""
        try:
            # ==================== 3D Point Cloud ====================
            self.ax_3d.clear()

            # Draw point cloud
            if filtered_data is not None and len(filtered_data) > 0:
                xs = filtered_data[:, 0]
                ys = filtered_data[:, 1]
                zs = filtered_data[:, 2]
                snr = filtered_data[:, 4] if filtered_data.shape[1] >= 5 else np.zeros_like(xs)

                # Color normalization
                if np.ptp(snr) > 1e-6:
                    norm_snr = (snr - np.min(snr)) / (np.ptp(snr) + 1e-6)
                else:
                    norm_snr = np.zeros_like(snr)

                # ✅ Downsample if too many points
                if len(xs) > 300:
                    step = len(xs) // 300
                    xs, ys, zs, norm_snr = xs[::step], ys[::step], zs[::step], norm_snr[::step]

                self.ax_3d.scatter(xs, ys, zs, c=norm_snr, cmap='jet', s=6, alpha=0.7)

            # Draw cluster boxes
            for box in cluster_boxes:
                xmin, xmax = box['xmin'], box['xmax']
                ymin, ymax = box['ymin'], box['ymax']
                zmin, zmax = box['zmin'], box['zmax']

                cx = (xmin + xmax) / 2
                cy = (ymin + ymax) / 2
                cz = (zmin + zmax) / 2

                # Add to trajectory
                self.trajectory_history.append((cx, cy, time.time()))
                if len(self.trajectory_history) > self.max_trajectory_points:
                    self.trajectory_history.pop(0)

                # Draw bounding box
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
                        color='red', linewidth=1.2
                    )

                self.ax_3d.text(cx, cy, cz, f'C{box["label"]}',
                                color='yellow', fontsize=9, weight='bold')

            # Set axes
            self.ax_3d.set_xlabel('X (m)', fontsize=9)
            self.ax_3d.set_ylabel('Y (m)', fontsize=9)
            self.ax_3d.set_zlabel('Z (m)', fontsize=9)
            self.ax_3d.set_xlim(-4, 4)
            self.ax_3d.set_ylim(0, 8)
            self.ax_3d.set_zlim(-4, 4)
            self.ax_3d.set_box_aspect([1, 1, 1])
            self.ax_3d.set_title(os.path.basename(file_path) if file_path else "Realtime", fontsize=10)

            self.canvas_3d.draw_idle()

            # ==================== XY Trajectory ====================
            self._update_xy_trajectory()

            # ✅ Update FPS statistics
            self.frame_count += 1

        except Exception as e:
            self._log(f"[Visualization] Update failed: {e}")

    def _update_xy_trajectory(self):
        """Update XY trajectory"""
        try:
            self.ax_xy.clear()

            self.ax_xy.set_xlabel("X (m)", fontsize=9)
            self.ax_xy.set_ylabel("Y (m)", fontsize=9)
            self.ax_xy.set_xlim(-4, 4)
            self.ax_xy.set_ylim(0, 8)
            self.ax_xy.grid(True, alpha=0.3)
            self.ax_xy.set_aspect('equal')

            if len(self.trajectory_history) == 0:
                self.ax_xy.set_title("XY Trajectory (No Data)", fontsize=10)
                self.canvas_xy.draw_idle()
                return

            xs = [p[0] for p in self.trajectory_history]
            ys = [p[1] for p in self.trajectory_history]
            timestamps = [p[2] for p in self.trajectory_history]

            current_time = time.time()
            alphas = [max(0.1, 1.0 - (current_time - ts) / 30.0) for ts in timestamps]

            # ✅ Downsample trajectory
            if len(xs) > 50:
                step = max(1, len(xs) // 50)
                xs_plot = xs[::step]
                ys_plot = ys[::step]
                alphas_plot = alphas[::step]
            else:
                xs_plot, ys_plot, alphas_plot = xs, ys, alphas

            # Draw trajectory lines
            for i in range(len(xs_plot) - 1):
                self.ax_xy.plot(
                    xs_plot[i:i + 2], ys_plot[i:i + 2],
                    color='blue',
                    alpha=alphas_plot[i],
                    linewidth=1.5
                )

            # Draw trajectory points
            self.ax_xy.scatter(xs_plot, ys_plot, c=alphas_plot, cmap='Blues',
                               s=20, alpha=0.7, edgecolors='navy', linewidths=0.5)

            # Annotate latest position
            if len(xs) > 0:
                self.ax_xy.scatter(xs[-1], ys[-1], c='red', s=80, marker='*',
                                   edgecolors='darkred', linewidths=1.5, zorder=10)
                self.ax_xy.text(xs[-1] + 0.2, ys[-1] + 0.2, 'Current',
                                fontsize=9, color='red', weight='bold')

            self.ax_xy.set_title(f"XY Trajectory ({len(self.trajectory_history)} pts)", fontsize=10)

            self.canvas_xy.draw_idle()

        except Exception as e:
            self._log(f"[Trajectory] Update failed: {e}")


# ----------------- Main Entrypoint -----------------
if __name__ == "__main__":
    app = App()
    app.mainloop()