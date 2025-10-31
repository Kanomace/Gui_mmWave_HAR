# pointcloud_notifier.py
import os
import threading

_callbacks = []
_lock = threading.Lock()

def register_callback(func):
    print(f"[DEBUG] register_callback: {func}")
    with _lock:
        _callbacks.append(func)

def notify_new_pointcloud(file_path):
    """当新点云文件保存时调用"""
    with _lock:
        for cb in _callbacks:
            try:
                cb(file_path)
            except Exception as e:
                print(f"[Notifier] Error in callback: {e}")
