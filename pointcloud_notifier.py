# pointcloud_notifier.py
import threading

_callbacks = []
_lock = threading.Lock()

def register_callback(func):
    """注册 GUI 或其他模块的回调函数"""
    print(f"[Notifier] Callback registered: {func.__name__}")
    with _lock:
        _callbacks.append(func)

def notify_new_pointcloud(data):
    """
    当新的点云或聚类结果保存时调用。
    data 可以是：
      - str: 表示普通点云文件路径
      - dict: 包含 {'type': 'clustered', 'clusters': [...], 'noise': [...]}
    """
    with _lock:
        for cb in _callbacks:
            try:
                cb(data)
            except Exception as e:
                print(f"[Notifier] Error in callback {cb.__name__}: {e}")
