# gui_parser_with_dbscan.py
# ---------------------------------------------------------------
# 功能：
#   1️⃣ 实时接收点云帧并保存为 xlsx
#   2️⃣ 自动调用 DBSCAN_generator 聚类
#   3️⃣ 把聚类结果保存到 cluster_xlsx 文件夹
# ---------------------------------------------------------------

import struct
import serial
import time
import numpy as np
import math
import datetime
import pandas as pd
import os
import sys
from pointcloud_notifier import notify_new_pointcloud  # 🔔 通知 GUI
from parseFrame import *                               # 原有解析函数
from DBSCAN_generator import DBSCANGenerator            # ✅ 引入DBSCAN模块


def write_output_data(file_path, parsed_data):
    """写入输出文件"""
    with open(file_path, 'w') as file:
        file.write(str(parsed_data))


class uartParser():
    def __init__(self, type='SDK Out of Box Demo', out_bin_dir=None, out_xlsx_dir=None):
        self.saveBinary = 0
        self.replay = 0
        self.binData = bytearray()
        self.uartCounter = 0
        self.framesPerFile = 3
        self.first_file = True
        self.filepath = datetime.datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        self.pointCloudCache = []

        self.out_bin_dir = out_bin_dir
        self.out_xlsx_dir = out_xlsx_dir

        # 选择解析类型
        if type in [DEMO_NAME_OOB, DEMO_NAME_LRPD, DEMO_NAME_3DPC,
                    DEMO_NAME_SOD, DEMO_NAME_VITALS, DEMO_NAME_MT, DEMO_NAME_GESTURE]:
            self.parserType = "DoubleCOMPort"
        elif type in [DEMO_NAME_x432_OOB, DEMO_NAME_x432_GESTURE]:
            self.parserType = "SingleCOMPort"
        elif type == "Replay":
            self.replay = 1
        else:
            print("ERROR, unsupported demo type selected!")

        self.now_time = datetime.datetime.now().strftime('%Y%m%d-%H%M')

        # ✅ 初始化 DBSCAN 模块
        cfg = {
            'DBSCAN_GENERATOR_CFG': {
                'Default': {
                    'DBS_eps': 0.25,  # 聚类半径
                    'DBS_min_samples': 5,  # 最小点数
                    'DBS_cp_pos_xlim': (-2, 2),
                    'DBS_cp_pos_ylim': (0, 5),
                    'DBS_cp_pos_zlim': (-1, 2),
                    'DBS_size_xlim': (0, 2),
                    'DBS_size_ylim': (0, 2),
                    'DBS_size_zlim': (0, 2),
                    'DBS_sort': 3
                }
            }
        }
        self.dbscan = DBSCANGenerator(**cfg)

        # ✅ 结果保存路径
        if self.out_xlsx_dir:
            self.cluster_dir = os.path.join(self.out_xlsx_dir, "cluster_xlsx")
            os.makedirs(self.cluster_dir, exist_ok=True)
        else:
            self.cluster_dir = None

    # ------------------- 公共方法 --------------------
    def setSaveBinary(self, saveBinary: int):
        self.saveBinary = saveBinary
        print(f"saveBinary set to: {self.saveBinary}")

    # ------------------- DoubleCOMPort --------------------
    def readAndParseUartDoubleCOMPort(self):
        """双串口读取 + DBSCAN 聚类保存"""
        self.fail = 0
        if self.replay:
            return self.replayHist()

        index = 0
        magicByte = self.dataCom.read(1)
        frameData = bytearray()

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

        frameData += self.dataCom.read(4)  # version
        lengthBytes = self.dataCom.read(4)
        frameData += lengthBytes
        frameLength = int.from_bytes(lengthBytes, 'little') - 16
        frameData += self.dataCom.read(frameLength)

        if self.saveBinary == 1:
            self.binData += frameData
            self.uartCounter += 1

            # --- 解析点云 ---
            outputDict = parseStandardFrame(frameData)
            frame_df = pd.DataFrame(
                outputDict["pointCloud"],
                columns=['X', 'Y', 'Z', 'Doppler', 'SNR', 'Noise', 'Track index']
            )
            self.pointCloudCache.append(frame_df)

            # ✅ 保持缓存最多 N 帧
            if len(self.pointCloudCache) > self.framesPerFile:
                self.pointCloudCache.pop(0)

            if len(self.pointCloudCache) == self.framesPerFile:
                if self.first_file:
                    if self.out_bin_dir is None or self.out_xlsx_dir is None:
                        raise ValueError("uartParser: out_bin_dir or out_xlsx_dir not set")
                    os.makedirs(self.out_bin_dir, exist_ok=True)
                    os.makedirs(self.out_xlsx_dir, exist_ok=True)
                    self.first_file = False

                # 保存 bin 文件
                file_bin = os.path.join(self.out_bin_dir, f"pHistBytes_{self.uartCounter}.bin")
                with open(file_bin, 'wb') as bfile:
                    bfile.write(bytes(self.binData))
                self.binData = bytearray()

                # 保存点云 xlsx
                all_points = pd.concat(self.pointCloudCache, ignore_index=True)
                file_xlsx = os.path.join(self.out_xlsx_dir, f"pHistBytes_{self.uartCounter}.xlsx")
                all_points.to_excel(file_xlsx, index=False)
                notify_new_pointcloud(file_xlsx)

                # ====== 🧩 DBSCAN 聚类并保存 ======
                if self.cluster_dir:
                    try:
                        data = all_points[['X', 'Y', 'Z', 'Doppler', 'SNR']].to_numpy(dtype=np.float32)
                        _, valid_points_list, _, noise = self.dbscan.DBS(data)
                        cluster_path = os.path.join(self.cluster_dir, f"cluster_{self.uartCounter}.xlsx")

                        with pd.ExcelWriter(cluster_path) as writer:
                            for i, cluster in enumerate(valid_points_list):
                                df_cluster = pd.DataFrame(cluster, columns=['X', 'Y', 'Z', 'Doppler', 'SNR'])
                                df_cluster.to_excel(writer, sheet_name=f'Cluster_{i+1}', index=False)
                            if len(noise) > 0:
                                df_noise = pd.DataFrame(noise, columns=['X', 'Y', 'Z', 'Doppler', 'SNR'])
                                df_noise.to_excel(writer, sheet_name='Noise', index=False)
                        print(f"✅ Cluster saved: {cluster_path}")
                    except Exception as e:
                        print(f"[DBSCAN ERROR] {e}")

        return parseStandardFrame(frameData)

    # ------------------- SingleCOMPort --------------------
    def readAndParseUartSingleCOMPort(self):
        """单串口读取 + DBSCAN 聚类保存"""
        if not self.cliCom.isOpen():
            self.cliCom.open()

        self.fail = 0
        if self.replay:
            return self.replayHist()

        index = 0
        magicByte = self.cliCom.read(1)
        frameData = bytearray()

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

        frameData += self.cliCom.read(4)
        lengthBytes = self.cliCom.read(4)
        frameData += lengthBytes
        frameLength = int.from_bytes(lengthBytes, 'little') - 16
        frameData += self.cliCom.read(frameLength)

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

                file_bin = os.path.join(self.out_bin_dir, f"pHistBytes_{self.uartCounter}.bin")
                with open(file_bin, 'wb') as bfile:
                    bfile.write(bytes(self.binData))
                self.binData = bytearray()

                all_points = pd.concat(self.pointCloudCache, ignore_index=True)
                file_xlsx = os.path.join(self.out_xlsx_dir, f"pHistBytes_{self.uartCounter}.xlsx")
                all_points.to_excel(file_xlsx, index=False)
                notify_new_pointcloud(file_xlsx)

                # ====== 🧩 DBSCAN 聚类并保存 ======
                if self.cluster_dir:
                    try:
                        data = all_points[['X', 'Y', 'Z', 'Doppler', 'SNR']].to_numpy(dtype=np.float32)
                        _, valid_points_list, _, noise = self.dbscan.DBS(data)
                        cluster_path = os.path.join(self.cluster_dir, f"cluster_{self.uartCounter}.xlsx")

                        with pd.ExcelWriter(cluster_path) as writer:
                            for i, cluster in enumerate(valid_points_list):
                                df_cluster = pd.DataFrame(cluster, columns=['X', 'Y', 'Z', 'Doppler', 'SNR'])
                                df_cluster.to_excel(writer, sheet_name=f'Cluster_{i+1}', index=False)
                            if len(noise) > 0:
                                df_noise = pd.DataFrame(noise, columns=['X', 'Y', 'Z', 'Doppler', 'SNR'])
                                df_noise.to_excel(writer, sheet_name='Noise', index=False)
                        print(f"✅ Cluster saved: {cluster_path}")
                    except Exception as e:
                        print(f"[DBSCAN ERROR] {e}")

        return parseStandardFrame(frameData)

    # ------------------- COM Port 连接 --------------------
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

    # ------------------- 配置下发 --------------------
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
