# radar_interface.py
import os
import time
from gui_parser import uartParser

class RadarToXlsxDumper:
    def __init__(self, parser_type='SDK Out of Box Demo', output_dir=None):
        """
        parser_type: 初始化 uartParser 时的类型
        output_dir: 输出根目录，由 gui_main.py 传入（带时间戳）
        """
        # 自动创建输出目录，如果没有传入就使用当前目录
        self.output_dir = output_dir or os.getcwd()
        # 创建时间戳文件夹
        self.timestamp_dir = time.strftime('%Y%m%d-%H%M%S')
        self.full_output_dir = os.path.join(self.output_dir, self.timestamp_dir)
        os.makedirs(self.full_output_dir, exist_ok=True)

        # 初始化 uartParser
        self.parser = uartParser(type=parser_type, output_dir=self.full_output_dir)
        # 默认保存 binary 和 xlsx
        self.parser.setSaveBinary(1)

    def connect_ports(self, cli_com, data_com=None):
        """
        连接雷达串口
        data_com 为 None 时，使用单端口模式
        """
        if data_com:
            self.parser.connectComPorts(cli_com, data_com)
        else:
            self.parser.connectComPort(cli_com)

    def read_frame(self):
        """
        读取一帧雷达数据，并自动保存到 timestamp 文件夹中
        """
        if self.parser.parserType == "DoubleCOMPort":
            return self.parser.readAndParseUartDoubleCOMPort()
        else:
            return self.parser.readAndParseUartSingleCOMPort()
