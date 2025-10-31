import numpy as np
import os
import matplotlib.pyplot as plt
import re
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading


def extract_number(filename):
    """
    从文件名中提取数字部分
    """
    match = re.search(r'pHistBytes_(\d+)\.npy', filename)
    if match:
        return int(match.group(1))
    return 0


def create_single_projection(voxel_path, save_path_YOZ, save_path_XOY, save_path_XOZ):
    """
    创建单个体素文件的三个投影视图
    """
    try:
        # 读取体素文件
        voxel_data = np.load(voxel_path)

        # 提取XOY平面
        xoyslice = np.max(voxel_data, axis=2)
        xoyslice = np.rot90(xoyslice)
        xoyslice = np.where(xoyslice > 0, 1, 0)  # 将体素区域设为白色，空白区域设为黑色

        # 提取XOZ平面
        xozslice = np.max(voxel_data, axis=1)
        xozslice = np.rot90(xozslice)
        xozslice = np.where(xozslice > 0, 1, 0)  # 将体素区域设为白色，空白区域设为黑色

        # 提取YOZ平面
        yozslice = np.max(voxel_data, axis=0)
        yozslice = np.rot90(yozslice)
        yozslice = np.where(yozslice > 0, 1, 0)  # 将体素区域设为白色，空白区域设为黑色

        # 保存为图像文件
        base_name = os.path.splitext(os.path.basename(voxel_path))[0]
        save_file_name_YOZ = os.path.join(save_path_YOZ, base_name + '_YOZ.png')
        save_file_name_XOY = os.path.join(save_path_XOY, base_name + '_XOY.png')
        save_file_name_XOZ = os.path.join(save_path_XOZ, base_name + '_XOZ.png')

        plt.imsave(save_file_name_YOZ, yozslice, cmap='gray')
        plt.imsave(save_file_name_XOY, xoyslice, cmap='gray')
        plt.imsave(save_file_name_XOZ, xozslice, cmap='gray')

        # 显示处理进度
        #frame_num = extract_number(os.path.basename(voxel_path))
        #print(f'Processed frame {frame_num}: {os.path.basename(voxel_path)}')

        return True

    except Exception as e:
        print(f"Error occurred for {voxel_path}: {e}")
        return False


def create_projections(voxel_folder, save_path_YOZ, save_path_XOY, save_path_XOZ):
    """
    创建体素数据的三个投影视图
    """
    # 创建保存路径文件夹
    os.makedirs(save_path_YOZ, exist_ok=True)
    os.makedirs(save_path_XOY, exist_ok=True)
    os.makedirs(save_path_XOZ, exist_ok=True)

    # 获取所有.npy文件并按数字顺序排序
    npy_files = [f for f in os.listdir(voxel_folder) if f.endswith('.npy')]
    npy_files.sort(key=extract_number)

    print(f"Found {len(npy_files)} .npy files, processing in order...")

    processed_count = 0
    error_count = 0

    # 按顺序处理每个文件
    for file_name in npy_files:
        try:
            # 处理单个文件
            if create_single_projection(
                    os.path.join(voxel_folder, file_name),
                    save_path_YOZ,
                    save_path_XOY,
                    save_path_XOZ
            ):
                processed_count += 1
            else:
                error_count += 1

        except Exception as e:
            error_count += 1
            print(f"Error occurred for {file_name}: {e}")

    print("All frames processed successfully!")
    return processed_count, error_count


class VoxelFileHandler(FileSystemEventHandler):
    """监视体素文件变化的处理器"""

    def __init__(self, save_path_YOZ, save_path_XOY, save_path_XOZ):
        self.save_path_YOZ = save_path_YOZ
        self.save_path_XOY = save_path_XOY
        self.save_path_XOZ = save_path_XOZ
        self.processed_files = set()  # 记录已处理的文件

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.npy'):
            # 检查文件是否已处理过
            if event.src_path in self.processed_files:
                return

            # 使用文件大小变化检测文件是否写完
            last_size = -1
            for _ in range(5):  # 最多检查5次
                try:
                    current_size = os.path.getsize(event.src_path)
                    if current_size == last_size and current_size > 0:
                        # 文件大小稳定，开始处理
                        print(f"检测到新体素文件: {event.src_path}")
                        create_single_projection(
                            event.src_path,
                            self.save_path_YOZ,
                            self.save_path_XOY,
                            self.save_path_XOZ
                        )
                        self.processed_files.add(event.src_path)
                        break
                    last_size = current_size
                    time.sleep(0.05)  # 短暂等待
                except OSError:
                    # 文件可能还在写入，继续等待
                    time.sleep(0.05)


def monitor_voxel_folder(voxel_folder, save_path_YOZ, save_path_XOY, save_path_XOZ):
    """监视体素文件夹并实时处理新文件"""
    # 确保输出文件夹存在
    os.makedirs(save_path_YOZ, exist_ok=True)
    os.makedirs(save_path_XOY, exist_ok=True)
    os.makedirs(save_path_XOZ, exist_ok=True)

    # 创建事件处理器
    event_handler = VoxelFileHandler(save_path_YOZ, save_path_XOY, save_path_XOZ)

    # 创建观察者
    observer = Observer()
    observer.schedule(event_handler, voxel_folder, recursive=False)

    # 启动观察者
    observer.start()
    print(f"开始监视体素文件夹: {voxel_folder}")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()