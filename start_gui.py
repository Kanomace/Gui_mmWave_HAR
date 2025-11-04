#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
启动脚本 - 彻底抑制所有警告
"""

import os
import sys
import warnings

# ✅ Step 1: 在任何导入之前禁用所有警告
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

# ✅ Step 2: 重定向 stderr 来过滤 tkinter 警告
import io
import contextlib


class WarningFilter(io.TextIOBase):
    """过滤包含特定关键词的警告"""

    def __init__(self, original_stderr):
        self.original_stderr = original_stderr
        self.keywords = ['Glyph', 'CJK', 'DejaVu', 'UserWarning', 'font']

    def write(self, text):
        # 如果包含警告关键词，忽略
        if any(kw in text for kw in self.keywords):
            return
        # 否则正常输出
        self.original_stderr.write(text)

    def flush(self):
        self.original_stderr.flush()


# ✅ Step 3: 替换 stderr
sys.stderr = WarningFilter(sys.stderr)

# ✅ Step 4: 配置 matplotlib（在导入 GUI 之前）
import matplotlib

matplotlib.use('TkAgg', force=True)
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# ✅ Step 5: 现在安全地导入 GUI
if __name__ == "__main__":
    from gui_main import App

    print("=" * 60)
    print("mmWave Real-time Visualization System")
    print("=" * 60)
    print("[INFO] Starting GUI...")
    print("[INFO] All warnings suppressed")
    print("=" * 60)

    app = App()
    app.mainloop()