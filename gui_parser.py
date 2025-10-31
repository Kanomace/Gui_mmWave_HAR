# gui_parser_with_bgn.py
# ---------------------------------------------------------------
# 功能：
#   1️⃣ 实时接收点云帧并保存为 xlsx
#   2️⃣ 在 DBSCAN 聚类前使用 BGNoiseFilter 背景噪声过滤
#   3️⃣ GUI 与保存文件均使用过滤后的点云
# ---------------------------------------------------------------

import os
import sys
import time
import datetime
import pandas as pd
import numpy as np
import serial

from pointcloud_notifier import notify_new_pointcloud
from library.parseFrame import *
from library.DBSCAN_generator import DBSCANGenerator
from library.bgnoise_filter import BGNoiseFilter   # ✅ 新增：背景噪声滤波模块


class uartParser:
    def __init__(self, type='SDK Out of Box Demo', out_bin_dir=None, out_xlsx_dir=None):
        """初始化UART解析器"""
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

        # ------------------- 类型判断 -------------------
        if type in [DEMO_NAME_OOB, DEMO_NAME_LRPD, DEMO_NAME_3DPC,
                    DEMO_NAME_SOD, DEMO_NAME_VITALS, DEMO_NAME_MT, DEMO_NAME_GESTURE]:
            self.parserType = "DoubleCOMPort"
        elif type in [DEMO_NAME_x432_OOB, DEMO_NAME_x432_GESTURE]:
            self.parserType = "SingleCOMPort"
        elif type == "Replay":
            self.replay = 1
        else:
            print("ERROR, unsupported demo type selected!")

        # ------------------- 初始化 DBSCAN + BGNoiseFilter -------------------
        cfg = {
            'DBSCAN_GENERATOR_CFG': {
                'Default': {
                    'DBS_eps': 0.5,
                    # maximum distance, larger means the further points can be clustered, smaller means the points need to be closer
                    'DBS_min_samples': 10,
                    # minimum samples, larger means more points are needed to form a cluster, 1-each point can be treated as a cluster, no noise

                    # DBSCAN filter para
                    'DBS_cp_pos_xlim': None,  # the position limit in x-direction for central points of clusters
                    'DBS_cp_pos_ylim': None,
                    'DBS_cp_pos_zlim': (-1.8, 1.8),
                    'DBS_size_xlim': (0.2, 1),  # the cluster size limit in x-direction
                    'DBS_size_ylim': (0.2, 1),
                    'DBS_size_zlim': (0.0, 2),
                    'DBS_sort': 3,
                    # if sort is required, set it to a number for acquiring this number of the largest cluster
                }
            },
            'BGNOISE_FILTER_CFG': {
                'BGN_enable': True,
                'BGN_deque_length': 50,  # ⚠️ 降低帧数要求，加快测试
                'BGN_accept_SNR_threshold': (None, 200),
                'BGN_filter_SNR_threshold': (None, 200),
                'BGN_DBS_window_step': 50,  # 更频繁更新
                'BGN_DBS_eps': 0.06,  # ⚠️ 扩大聚类半径
                'BGN_DBS_min_samples': 5,  # ⚠️ 降低簇形成要求
                'BGN_cluster_tf': 0.05,  # ⚠️ 降低背景点比例阈值
                'BGN_cluster_xextension': 0.05,
                'BGN_cluster_yextension': 0.05,
                'BGN_cluster_zextension': 0.05,  # ⚠️ 提高z方向容忍度
            }

        }
        self.dbscan = DBSCANGenerator(**cfg)
        self.bgn = BGNoiseFilter(**cfg)  # ✅ 初始化背景噪声滤波器

        # 输出文件夹
        if self.out_xlsx_dir:
            self.cluster_dir = os.path.join(self.out_xlsx_dir, "cluster_xlsx")
            os.makedirs(self.cluster_dir, exist_ok=True)
        else:
            self.cluster_dir = None

    # ------------------- 公共方法 -------------------
    def setSaveBinary(self, saveBinary: int):
        self.saveBinary = saveBinary
        print(f"saveBinary set to: {self.saveBinary}")

    # ------------------- 双串口模式 -------------------
    def readAndParseUartDoubleCOMPort(self):
        """双串口读取 + 背景滤波 + DBSCAN 聚类"""
        self.fail = 0
        if self.replay:
            return self.replayHist()

        index = 0
        magicByte = self.dataCom.read(1)
        frameData = bytearray()

        # --- 识别帧头 ---
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

        # --- 读取帧体 ---
        frameData += self.dataCom.read(4)  # version
        lengthBytes = self.dataCom.read(4)
        frameData += lengthBytes
        frameLength = int.from_bytes(lengthBytes, 'little') - 16
        frameData += self.dataCom.read(frameLength)

        # -------------------- 解析点云 --------------------
        if self.saveBinary == 1:
            self.binData += frameData
            self.uartCounter += 1

            outputDict = parseStandardFrame(frameData)
            frame_df = pd.DataFrame(
                outputDict["pointCloud"],
                columns=['X', 'Y', 'Z', 'Doppler', 'SNR', 'Noise', 'Track index']
            )
            self.pointCloudCache.append(frame_df)

            # 保持缓存帧数
            if len(self.pointCloudCache) > self.framesPerFile:
                self.pointCloudCache.pop(0)

            # 达到缓存数量后保存
            if len(self.pointCloudCache) == self.framesPerFile:
                if self.first_file:
                    os.makedirs(self.out_bin_dir, exist_ok=True)
                    os.makedirs(self.out_xlsx_dir, exist_ok=True)
                    self.first_file = False

                # 保存 bin
                file_bin = os.path.join(self.out_bin_dir, f"pHistBytes_{self.uartCounter}.bin")
                with open(file_bin, 'wb') as bfile:
                    bfile.write(bytes(self.binData))
                self.binData = bytearray()

                # 合并帧
                all_points = pd.concat(self.pointCloudCache, ignore_index=True)
                data_np = all_points[['X', 'Y', 'Z', 'Doppler', 'SNR']].to_numpy(dtype=np.float32)

                # ✅ Step 1: 更新背景模型（学习阶段）
                self.bgn.BGN_update(data_np)
                print(f"[BGN] 当前背景簇数量: {len(self.bgn.BGN_cluster_boundary)}")
                print(f"[BGN] 队列缓存帧数: {len(self.bgn.BGN_deque)}/{self.bgn.BGN_deque.maxlen}")

                # ✅ Step 2: 滤波（当模型已建立时才会生效）
                filtered_data = self.bgn.BGN_filter(data_np)

                print(f"[BGN] 输入点数: {data_np.shape[0]}, 滤波后点数: {filtered_data.shape[0]}")

                filtered_df = pd.DataFrame(filtered_data, columns=['X', 'Y', 'Z', 'Doppler', 'SNR'])

                # ✅ Step2: 保存滤波后的点云
                file_xlsx = os.path.join(self.out_xlsx_dir, f"pHistBytes_{self.uartCounter}.xlsx")
                filtered_df.to_excel(file_xlsx, index=False)
                print(f"[Save] 滤波点云已保存 {file_xlsx}")

                # ✅ Step3: DBSCAN 聚类
                cluster_path = None
                if self.cluster_dir:
                    try:
                        _, valid_points_list, _, noise = self.dbscan.DBS(filtered_data)
                        cluster_path = os.path.join(self.cluster_dir, f"cluster_{self.uartCounter}.xlsx")

                        with pd.ExcelWriter(cluster_path) as writer:
                            for i, cluster in enumerate(valid_points_list):
                                df_cluster = pd.DataFrame(cluster, columns=['X', 'Y', 'Z', 'Doppler', 'SNR'])
                                df_cluster.to_excel(writer, sheet_name=f'Cluster_{i + 1}', index=False)
                            if len(noise) > 0:
                                df_noise = pd.DataFrame(noise, columns=['X', 'Y', 'Z', 'Doppler', 'SNR'])
                                df_noise.to_excel(writer, sheet_name='Noise', index=False)
                        print(f"✅ Cluster saved: {cluster_path}")
                    except Exception as e:
                        print(f"[DBSCAN ERROR] {e}")

                # ✅ Step4: 通知 GUI（使用滤波后点云）
                notify_new_pointcloud((file_xlsx, cluster_path))

        return parseStandardFrame(frameData)

    # ------------------- 单串口模式 -------------------
    def readAndParseUartSingleCOMPort(self):
        """单串口读取 + 背景滤波 + DBSCAN 聚类"""
        if not self.cliCom.isOpen():
            self.cliCom.open()

        self.fail = 0
        if self.replay:
            return self.replayHist()

        index = 0
        magicByte = self.cliCom.read(1)
        frameData = bytearray()

        # --- 帧头识别 ---
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

        # --- 读取帧 ---
        frameData += self.cliCom.read(4)
        lengthBytes = self.cliCom.read(4)
        frameData += lengthBytes
        frameLength = int.from_bytes(lengthBytes, 'little') - 16
        frameData += self.cliCom.read(frameLength)

        # --- 解析点云 ---
        if self.saveBinary == 1:
            self.binData += frameData
            self.uartCounter += 1

            outputDict = parseStandardFrame(frameData)
            frame_df = pd.DataFrame(
                outputDict["pointCloud"],
                columns=['X', 'Y', 'Z', 'Doppler', 'SNR', 'Noise', 'Track index']
            )
            self.pointCloudCache.append(frame_df)

            if len(self.pointCloudCache) > self.framesPerFile:
                self.pointCloudCache.pop(0)

            if len(self.pointCloudCache) == self.framesPerFile:
                if self.first_file:
                    os.makedirs(self.out_bin_dir, exist_ok=True)
                    os.makedirs(self.out_xlsx_dir, exist_ok=True)
                    self.first_file = False

                # 保存 bin 文件
                file_bin = os.path.join(self.out_bin_dir, f"pHistBytes_{self.uartCounter}.bin")
                with open(file_bin, 'wb') as bfile:
                    bfile.write(bytes(self.binData))
                self.binData = bytearray()

                # 合并帧
                all_points = pd.concat(self.pointCloudCache, ignore_index=True)
                data_np = all_points[['X', 'Y', 'Z', 'Doppler', 'SNR']].to_numpy(dtype=np.float32)

                # ✅ 背景滤波
                filtered_data = self.bgn.BGN_filter(data_np)
                filtered_df = pd.DataFrame(filtered_data, columns=['X', 'Y', 'Z', 'Doppler', 'SNR'])

                # 保存滤波点云
                file_xlsx = os.path.join(self.out_xlsx_dir, f"pHistBytes_{self.uartCounter}.xlsx")
                filtered_df.to_excel(file_xlsx, index=False)

                # DBSCAN 聚类
                cluster_path = None
                if self.cluster_dir:
                    try:
                        _, valid_points_list, _, noise = self.dbscan.DBS(filtered_data)
                        cluster_path = os.path.join(self.cluster_dir, f"cluster_{self.uartCounter}.xlsx")
                        with pd.ExcelWriter(cluster_path) as writer:
                            for i, cluster in enumerate(valid_points_list):
                                df_cluster = pd.DataFrame(cluster, columns=['X', 'Y', 'Z', 'Doppler', 'SNR'])
                                df_cluster.to_excel(writer, sheet_name=f'Cluster_{i + 1}', index=False)
                            if len(noise) > 0:
                                df_noise = pd.DataFrame(noise, columns=['X', 'Y', 'Z', 'Doppler', 'SNR'])
                                df_noise.to_excel(writer, sheet_name='Noise', index=False)
                        print(f"✅ Cluster saved: {cluster_path}")
                    except Exception as e:
                        print(f"[DBSCAN ERROR] {e}")

                notify_new_pointcloud((file_xlsx, cluster_path))

        return parseStandardFrame(frameData)

    # ------------------- 串口连接 -------------------
    def connectComPorts(self, cliCom, dataCom):
        self.cliCom = serial.Serial(cliCom, 115200, parity=serial.PARITY_NONE,
                                    stopbits=serial.STOPBITS_ONE, timeout=0.6)
        self.dataCom = serial.Serial(dataCom, 921600, parity=serial.PARITY_NONE,
                                     stopbits=serial.STOPBITS_ONE, timeout=0.6)
        self.dataCom.reset_output_buffer()
        print('Connected')

    def connectComPort(self, cliCom, cliBaud=115200):
        self.cliCom = serial.Serial(cliCom, cliBaud, parity=serial.PARITY_NONE,
                                    stopbits=serial.STOPBITS_ONE, timeout=4)
        self.cliCom.reset_output_buffer()
        print('Connected (one port)')

    # ------------------- 配置下发 -------------------
    def sendCfg(self, cfg):
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

            ack = self.cliCom.readline(); print(ack)
            ack = self.cliCom.readline(); print(ack)

            splitLine = line.split()
            if splitLine[0] == "baudRate":
                try:
                    self.cliCom.baudrate = int(splitLine[1])
                except:
                    print("Error - Invalid baud rate")
                    sys.exit(1)

        time.sleep(0.03)
        self.cliCom.reset_input_buffer()

    def sendLine(self, line):
        self.cliCom.write(line.encode())
        ack = self.cliCom.readline(); print(ack)
        ack = self.cliCom.readline(); print(ack)


def getBit(byte, bitNum):
    mask = 1 << bitNum
    return 1 if byte & mask else 0
