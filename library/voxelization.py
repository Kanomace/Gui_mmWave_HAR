import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading


def natural_sort_key(s):
    """
    自然排序键函数，确保数字按数值大小排序
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('(\d+)', s)]


def voxelization(data, grid_size, boundaries):
    """
    将点云数据转换为体素网格
    """
    # 处理空点云情况
    if data.empty:
        return np.zeros(grid_size, dtype=np.uint8)

    # 计算每个体素的大小
    voxel_size = (
        (boundaries['x'][1] - boundaries['x'][0]) / grid_size[0],
        (boundaries['y'][1] - boundaries['y'][0]) / grid_size[1],
        (boundaries['z'][1] - boundaries['z'][0]) / grid_size[2]
    )

    # 计算每个点所属的体素坐标，并限制在网格范围内
    x_indices = ((data['x'] - boundaries['x'][0]) / voxel_size[0]).astype(int)
    y_indices = ((data['y'] - boundaries['y'][0]) / voxel_size[1]).astype(int)
    z_indices = ((data['z'] - boundaries['z'][0]) / voxel_size[2]).astype(int)

    # 限制索引在有效范围内
    x_indices = np.clip(x_indices, 0, grid_size[0] - 1)
    y_indices = np.clip(y_indices, 0, grid_size[1] - 1)
    z_indices = np.clip(z_indices, 0, grid_size[2] - 1)

    # 创建体素网格
    voxel_grid = np.zeros(grid_size, dtype=np.uint8)

    # 将点云数据填充到体素网格中
    voxel_grid[x_indices, y_indices, z_indices] = 1

    return voxel_grid


def process_single_excel_file(excel_path, output_folder, grid_size, boundaries):
    """
    处理单个Excel文件
    """
    try:
        # 读取 Excel 文件数据
        df = pd.read_excel(excel_path)

        # 检查必要的列是否存在
        if not all(col in df.columns for col in ['X', 'Y', 'Z']):
            print(f"Warning: File {excel_path} doesn't contain required columns (X, Y, Z)")
            return False

        # 提取 xyz 列数据
        x = df['X']
        y = df['Y']
        z = df['Z']

        # 创建点云数据 DataFrame
        point_cloud = pd.DataFrame({'x': x, 'y': y, 'z': z})

        # 筛选符合边界范围的点云数据
        point_cloud = point_cloud[
            (point_cloud['x'] >= boundaries['x'][0]) & (point_cloud['x'] <= boundaries['x'][1]) &
            (point_cloud['y'] >= boundaries['y'][0]) & (point_cloud['y'] <= boundaries['y'][1]) &
            (point_cloud['z'] >= boundaries['z'][0]) & (point_cloud['z'] <= boundaries['z'][1])
            ]

        # 进行体素化
        voxel_grid = voxelization(point_cloud, grid_size, boundaries)

        # 保存体素文件
        output_file = os.path.join(output_folder, f'{os.path.splitext(os.path.basename(excel_path))[0]}.npy')
        np.save(output_file, voxel_grid)

        return True

    except Exception as e:
        print(f"Error occurred in file {excel_path}: {e}")
        return False


def process_excel_files(input_folder, output_folder, grid_size, boundaries, skip_existing=True):
    """
    处理指定文件夹中的所有Excel文件
    """
    # 创建保存体素文件的文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取文件夹中的Excel文件列表并按自然顺序排序
    excel_files = [f for f in os.listdir(input_folder) if f.endswith('.xlsx')]
    excel_files.sort(key=natural_sort_key)

    print(f"找到 {len(excel_files)} 个Excel文件，按自然排序:")
    for i, f in enumerate(excel_files[:10]):
        print(f"  {i + 1}: {f}")
    if len(excel_files) > 10:
        print("  ...")

    processed_count = 0
    error_count = 0

    # 使用进度条显示处理进度
    for frame, excel_file in enumerate(tqdm(excel_files, desc="Processing Excel files")):
        try:
            # 检查输出文件是否已存在
            output_file = os.path.join(output_folder, f'{os.path.splitext(excel_file)[0]}.npy')
            if skip_existing and os.path.exists(output_file):
                processed_count += 1
                continue

            # 处理单个文件
            if process_single_excel_file(
                    os.path.join(input_folder, excel_file),
                    output_folder,
                    grid_size,
                    boundaries
            ):
                processed_count += 1
            else:
                error_count += 1

        except Exception as e:
            error_count += 1
            print(f"Error occurred in file {excel_file}: {e}")

    return processed_count, error_count


class ExcelFileHandler(FileSystemEventHandler):
    """监视Excel文件变化的处理器"""

    def __init__(self, output_folder, grid_size, boundaries):
        self.output_folder = output_folder
        self.grid_size = grid_size
        self.boundaries = boundaries
        self.processed_files = set()  # 记录已处理的文件

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.xlsx'):
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
                        process_single_excel_file(event.src_path, self.output_folder, self.grid_size, self.boundaries)
                        self.processed_files.add(event.src_path)
                        break
                    last_size = current_size
                    time.sleep(0.05)  # 短暂等待
                except OSError:
                    # 文件可能还在写入，继续等待
                    time.sleep(0.05)


def monitor_folder(input_folder, output_folder, grid_size, boundaries):
    """监视文件夹并实时处理新文件"""
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 创建事件处理器
    event_handler = ExcelFileHandler(output_folder, grid_size, boundaries)

    # 创建观察者
    observer = Observer()
    observer.schedule(event_handler, input_folder, recursive=False)

    # 启动观察者
    observer.start()
    print(f"开始监视文件夹: {input_folder}")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()