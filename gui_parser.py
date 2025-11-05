# gui_parser_en.py
# ================================================================
# Full English Version with Warmup Mechanism
# Features:
#   1. Real-time point cloud frame reception and saving as xlsx
#   2. Background noise filtering using BGNoiseFilter
#   3. DBSCAN clustering with empty file protection
#   4. Human tracking using HumanTracking module
#   5. Notify GUI via pointcloud_notifier
# ================================================================

import os
import sys
import time
import datetime
import pandas as pd
import numpy as np
import serial
import warnings

# ✅ Suppress all warnings
warnings.filterwarnings('ignore')

from pointcloud_notifier import notify_new_pointcloud
from library.parseFrame import *
from library.DBSCAN_generator import DBSCANGenerator
from library.bgnoise_filter import BGNoiseFilter
from library.human_tracking import HumanTracking


class uartParser:
    def __init__(self, type='SDK Out of Box Demo', out_bin_dir=None, out_xlsx_dir=None):
        """UART parser initialization"""
        self.saveBinary = 0
        self.replay = 0
        self.binData = bytearray()
        self.uartCounter = 0
        self.framesPerFile = 5
        self.first_file = True
        self.filepath = datetime.datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        self.pointCloudCache = []

        self.out_bin_dir = out_bin_dir
        self.out_xlsx_dir = out_xlsx_dir

        # ------------------- Type Detection -------------------
        if type in [DEMO_NAME_OOB, DEMO_NAME_LRPD, DEMO_NAME_3DPC,
                    DEMO_NAME_SOD, DEMO_NAME_VITALS, DEMO_NAME_MT, DEMO_NAME_GESTURE]:
            self.parserType = "DoubleCOMPort"
        elif type in [DEMO_NAME_x432_OOB, DEMO_NAME_x432_GESTURE]:
            self.parserType = "SingleCOMPort"
        elif type == "Replay":
            self.replay = 1
        else:
            print("[ERROR] Unsupported demo type selected!")

        # ------------------- Module Configuration -------------------
        cfg = {
            'DBSCAN_GENERATOR_CFG': {
                'Default': {
                    'DBS_eps': 0.5,
                    'DBS_min_samples': 10,
                    'DBS_cp_pos_xlim': None,
                    'DBS_cp_pos_ylim': None,
                    'DBS_cp_pos_zlim': (-1.8, 1.8),
                    'DBS_size_xlim': (0.2, 1),
                    'DBS_size_ylim': (0.2, 1),
                    'DBS_size_zlim': (0.0, 2),
                    'DBS_sort': 3,
                }
            },
            'BGNOISE_FILTER_CFG': {
                'BGN_enable': True,
                'BGN_deque_length': 50,
                'BGN_accept_SNR_threshold': (None, 200),
                'BGN_filter_SNR_threshold': (None, 200),
                'BGN_DBS_window_step': 50,
                'BGN_DBS_eps': 0.06,
                'BGN_DBS_min_samples': 5,
                'BGN_cluster_tf': 0.05,
                'BGN_cluster_xextension': 0.05,
                'BGN_cluster_yextension': 0.05,
                'BGN_cluster_zextension': 0.05,
            },
            'HUMAN_TRACKING_CFG': {
                'TRK_enable': True,
                'TRK_obj_bin_number': 2,
                'TRK_poss_clus_deque_length': 3,
                'TRK_redundant_clus_remove_cp_dis': 1,
            },
            'HUMAN_OBJECT_CFG': {
                'obj_deque_length': 60,
                'dis_diff_threshold': {
                    'threshold': 0.8,
                    'dynamic_ratio': 0.2,
                },
                'size_diff_threshold': 1,
                'expect_pos': {
                    'default': (None, None, 1.1),
                    'standing': (None, None, 1.1),
                    'sitting': (None, None, 0.7),
                    'lying': (None, None, 0.5),
                },
                'expect_shape': {
                    'default': (0.8, 0.8, 1.8),
                    'standing': (0.7, 0.7, 1.5),
                    'sitting': (0.3, 0.3, 0.6),
                    'lying': (0.8, 0.8, 0.4),
                },
                'sub_possibility_proportion': (1, 1, 1, 1),
                'inactive_timeout': 5,
                'obj_delete_timeout': 60,
                'fuzzy_boundary_enter': False,
                'fuzzy_boundary_threshold': 0.5,
                'scene_xlim': (-4, 4),
                'scene_ylim': (0, 8),
                'scene_zlim': (-4, 4),
                'standing_sitting_threshold': 0.9,
                'sitting_lying_threshold': 0.4,
                'get_fuzzy_pos_No': 20,
                'get_fuzzy_status_No': 40,
            },
        }

        # ------------------- Module Initialization -------------------
        self.dbscan = DBSCANGenerator(**cfg)
        self.bgn = BGNoiseFilter(**cfg)
        self.tracker = HumanTracking(
            HUMAN_TRACKING_CFG=cfg['HUMAN_TRACKING_CFG'],
            HUMAN_OBJECT_CFG=cfg['HUMAN_OBJECT_CFG']
        )

        # ✅ Background filter warmup counter
        self.bgn_warmup_counter = 0
        self.bgn_warmup_frames = 50  # 🔧 Fixed: Match BGN_deque_length (50 frames)

        # 🔧 Added: Failure counter and diagnostics
        self.bgn_fail_counter = 0
        self.bgn_fail_threshold = 3  # Consecutive failures before fallback

        # Output paths
        if self.out_xlsx_dir:
            self.cluster_dir = os.path.join(self.out_xlsx_dir, "cluster_xlsx")
            os.makedirs(self.cluster_dir, exist_ok=True)
        else:
            self.cluster_dir = None

    # ------------------- Public Methods -------------------
    def setSaveBinary(self, saveBinary: int):
        self.saveBinary = saveBinary
        print(f"[Parser] SaveBinary set to: {self.saveBinary}")

    # ================================================================
    # Serial Port Mode
    # ================================================================
    def readAndParseUartDoubleCOMPort(self):
        """Double COM port read + background filtering + DBSCAN clustering + human tracking"""
        self.fail = 0
        if self.replay:
            return self.replayHist()

        index = 0
        magicByte = self.dataCom.read(1)
        frameData = bytearray()

        # --- Frame header recognition ---
        while True:
            if len(magicByte) < 1:
                magicByte = self.dataCom.read(1)
            elif magicByte[0] == UART_MAGIC_WORD[index]:
                index += 1
                frameData.append(magicByte[0])
                if index == 8:
                    break
                magicByte = self.dataCom.read(1)
            else:
                if index == 0:
                    magicByte = self.dataCom.read(1)
                index = 0
                frameData = bytearray()

        # --- Read frame body ---
        frameData += self.dataCom.read(4)  # version
        lengthBytes = self.dataCom.read(4)
        frameData += lengthBytes
        frameLength = int.from_bytes(lengthBytes, 'little') - 16
        frameData += self.dataCom.read(frameLength)

        # ================================================================
        # ✅ Step1: Parse frame
        # ================================================================
        if self.saveBinary == 1:
            self.binData += frameData
            self.uartCounter += 1
            outputDict = parseStandardFrame(frameData)
            frame_df = pd.DataFrame(
                outputDict["pointCloud"],
                columns=['X', 'Y', 'Z', 'Doppler', 'SNR', 'Noise', 'Track index']
            )
            self.pointCloudCache.append(frame_df)

            # Cache control
            if len(self.pointCloudCache) > self.framesPerFile:
                self.pointCloudCache.pop(0)

            # ================================================================
            # ✅ Step2: Process after reaching cache limit
            # ================================================================
            if len(self.pointCloudCache) == self.framesPerFile:
                if self.first_file:
                    os.makedirs(self.out_bin_dir, exist_ok=True)
                    os.makedirs(self.out_xlsx_dir, exist_ok=True)
                    self.first_file = False

                # Save bin file
                file_bin = os.path.join(self.out_bin_dir, f"pHistBytes_{self.uartCounter}.bin")
                with open(file_bin, 'wb') as bfile:
                    bfile.write(bytes(self.binData))
                self.binData = bytearray()

                # Merge frame data
                all_points = pd.concat(self.pointCloudCache, ignore_index=True)
                data_np = all_points[['X', 'Y', 'Z', 'Doppler', 'SNR']].to_numpy(dtype=np.float32)

                # ================================================================
                # ✅ Step3: Background filtering (with warmup mechanism)
                # ================================================================
                self.bgn.BGN_update(data_np)
                self.bgn_warmup_counter += 1

                # ✅ First N frames: no filtering, use raw data
                if self.bgn_warmup_counter <= self.bgn_warmup_frames:
                    filtered_data = data_np
                    print(f"[BGN] Warmup ({self.bgn_warmup_counter}/{self.bgn_warmup_frames}), using raw point cloud")
                else:
                    filtered_data = self.bgn.BGN_filter(data_np)
                    filter_rate = filtered_data.shape[0] / data_np.shape[0] if data_np.shape[0] > 0 else 0
                    print(f"[BGN] Raw: {data_np.shape[0]}, Filtered: {filtered_data.shape[0]} ({filter_rate:.1%})")

                    # 🔧 Improved: Use percentage-based threshold with fault tolerance
                    min_points = max(5, int(data_np.shape[0] * 0.05))  # At least 5 points or 5% of raw
                    if filtered_data.shape[0] < min_points:
                        self.bgn_fail_counter += 1
                        print(
                            f"[BGN] ⚠️  Filter failure {self.bgn_fail_counter}/{self.bgn_fail_threshold}: {filtered_data.shape[0]} < {min_points}")

                        if self.bgn_fail_counter >= self.bgn_fail_threshold:
                            print(f"[BGN] 🔄 Consecutive failures threshold reached, using raw point cloud")
                            filtered_data = data_np
                            self.bgn_fail_counter = 0  # Reset counter
                        else:
                            # Still use filtered data even if low, unless consecutive failures
                            pass
                    else:
                        # Success, reset failure counter
                        if self.bgn_fail_counter > 0:
                            print(f"[BGN] ✓ Filter recovered, resetting failure counter")
                        self.bgn_fail_counter = 0

                filtered_df = pd.DataFrame(filtered_data, columns=['X', 'Y', 'Z', 'Doppler', 'SNR'])
                file_xlsx = os.path.join(self.out_xlsx_dir, f"pHistBytes_{self.uartCounter}.xlsx")
                filtered_df.to_excel(file_xlsx, index=False)
                print(f"[Save] Point cloud saved -> {file_xlsx}")

                # ================================================================
                # ✅ Step4: DBSCAN clustering + empty file protection
                # ================================================================
                cluster_path = None
                valid_points_list = []
                try:
                    _, valid_points_list, _, noise = self.dbscan.DBS(filtered_data)
                    cluster_path = os.path.join(self.cluster_dir, f"cluster_{self.uartCounter}.xlsx")

                    with pd.ExcelWriter(cluster_path, engine="openpyxl") as writer:
                        if len(valid_points_list) == 0 and len(noise) == 0:
                            pd.DataFrame(columns=['X', 'Y', 'Z', 'Doppler', 'SNR']).to_excel(
                                writer, sheet_name='Empty', index=False)
                            print(f"[DBSCAN] No clusters found -> {cluster_path}")
                        else:
                            for i, cluster in enumerate(valid_points_list):
                                if len(cluster) == 0:
                                    continue
                                df_cluster = pd.DataFrame(cluster, columns=['X', 'Y', 'Z', 'Doppler', 'SNR'])
                                df_cluster.to_excel(writer, sheet_name=f'Cluster_{i + 1}', index=False)
                            if len(noise) > 0:
                                df_noise = pd.DataFrame(noise, columns=['X', 'Y', 'Z', 'Doppler', 'SNR'])
                                df_noise.to_excel(writer, sheet_name='Noise', index=False)
                            print(f"[DBSCAN] Cluster results saved -> {cluster_path}")

                except Exception as e:
                    print(f"[DBSCAN ERROR] {e}")

                # ================================================================
                # ✅ Step5: Human tracking
                # ================================================================
                track_positions = []
                try:
                    if len(valid_points_list) > 0:
                        self.tracker.TRK_update_poss_matrix(valid_points_list)
                        for person in self.tracker.TRK_people_list:
                            obj_cp, _, _ = person.get_info()
                            if obj_cp.size > 0:
                                track_positions.append(obj_cp.flatten().tolist())
                except Exception as e:
                    print(f"[Tracking ERROR] {e}")

                # ================================================================
                # ✅ Step6: Notify GUI
                # ================================================================
                notify_new_pointcloud((file_xlsx, cluster_path, filtered_data, track_positions))

        return parseStandardFrame(frameData)

    # ------------------- Single COM Port Mode -------------------
    def readAndParseUartSingleCOMPort(self):
        """Single COM port mode (similar logic)"""
        if not self.cliCom.isOpen():
            self.cliCom.open()

        self.fail = 0
        if self.replay:
            return self.replayHist()

        index = 0
        magicByte = self.cliCom.read(1)
        frameData = bytearray()

        # Frame header recognition
        while True:
            if len(magicByte) < 1:
                magicByte = self.cliCom.read(1)
            elif magicByte[0] == UART_MAGIC_WORD[index]:
                index += 1
                frameData.append(magicByte[0])
                if index == 8:
                    break
                magicByte = self.cliCom.read(1)
            else:
                if index == 0:
                    magicByte = self.cliCom.read(1)
                index = 0
                frameData = bytearray()

        # Read frame
        frameData += self.cliCom.read(4)
        lengthBytes = self.cliCom.read(4)
        frameData += lengthBytes
        frameLength = int.from_bytes(lengthBytes, 'little') - 16
        frameData += self.cliCom.read(frameLength)

        return parseStandardFrame(frameData)

    # ------------------- Serial Port Connection -------------------
    def connectComPorts(self, cliCom, dataCom):
        """Connect double COM ports"""
        self.cliCom = serial.Serial(cliCom, 115200, parity=serial.PARITY_NONE,
                                    stopbits=serial.STOPBITS_ONE, timeout=0.6)
        self.dataCom = serial.Serial(dataCom, 921600, parity=serial.PARITY_NONE,
                                     stopbits=serial.STOPBITS_ONE, timeout=0.6)
        self.dataCom.reset_output_buffer()
        print('[Serial] Connected (Double COM port)')

    def connectComPort(self, cliCom, cliBaud=115200):
        """Connect single COM port"""
        self.cliCom = serial.Serial(cliCom, cliBaud, parity=serial.PARITY_NONE,
                                    stopbits=serial.STOPBITS_ONE, timeout=4)
        self.cliCom.reset_output_buffer()
        print('[Serial] Connected (Single COM port)')

    # ------------------- Configuration Sending -------------------
    def sendCfg(self, cfg):
        """Send configuration to radar"""
        for i, line in enumerate(cfg):
            if line == '\n':
                cfg.remove(line)
            elif line[-1] != '\n':
                cfg[i] = line + '\n'

        for line in cfg:
            time.sleep(.03)
            if self.cliCom.baudrate == 1250000:
                for char in [*line]:
                    time.sleep(.001)
                    self.cliCom.write(char.encode())
            else:
                self.cliCom.write(line.encode())

            ack = self.cliCom.readline()
            print(f"[Config] {ack.decode('utf-8', errors='ignore').strip()}")
            ack = self.cliCom.readline()
            print(f"[Config] {ack.decode('utf-8', errors='ignore').strip()}")

            splitLine = line.split()
            if splitLine[0] == "baudRate":
                try:
                    self.cliCom.baudrate = int(splitLine[1])
                    print(f"[Config] Baud rate changed to {splitLine[1]}")
                except:
                    print("[Config ERROR] Invalid baud rate")
                    sys.exit(1)

        time.sleep(0.03)
        self.cliCom.reset_input_buffer()
        print("[Config] Configuration completed")

    def sendLine(self, line):
        """Send single command line"""
        self.cliCom.write(line.encode())
        ack = self.cliCom.readline()
        print(f"[Command] {ack.decode('utf-8', errors='ignore').strip()}")
        ack = self.cliCom.readline()
        print(f"[Command] {ack.decode('utf-8', errors='ignore').strip()}")


def getBit(byte, bitNum):
    """Get bit value from byte"""
    mask = 1 << bitNum
    return 1 if byte & mask else 0