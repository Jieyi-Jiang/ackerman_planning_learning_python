import sys
import cv2
import cv2.mat_wrapper
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel,
                             QPushButton, QVBoxLayout, QHBoxLayout,
                             QFileDialog, QScrollArea, QStyle, QFrame)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QIcon
from PyQt5.QtCore import Qt, pyqtSignal, QRect

class ImageBox(QLabel):
    # 发射原图坐标信号
    clicked = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        # 对象居中对其
        self.setAlignment(Qt.AlignCenter)
        # 设置背景色为白色
        self.setStyleSheet("background-color: white;")
        # 失能缩放
        self.setScaledContents(False)


    def mousePressEvent(self, event):
        # 当有鼠标左键点击事件和标签中有图片时
        if event.button() == Qt.LeftButton and self.pixmap():
            # 获取图片信息
            pixmap = self.pixmap()
            # print(pixmap.rect())
            # 计算坐标转换
            # 获取标签长宽
            label_w = self.width()
            label_h = self.height()
            # 获取图像长宽
            pix_w = pixmap.width()
            pix_h = pixmap.height()
            ratio = 1.0
            display_w = int(pix_w * ratio)
            display_h = int(pix_h * ratio)
            # 计算偏移
            x_offset = (label_w - display_w) // 2
            y_offset = (label_h - display_h) // 2

            x_in_label = event.x() - x_offset
            y_in_label = event.y() - y_offset

            if 0 <= x_in_label < display_w and 0 <= y_in_label < display_h:
                x_ori = int(x_in_label / ratio)
                y_ori = int(y_in_label / ratio)
                # 发送事件
                self.clicked.emit(x_ori, y_ori)
        super().mousePressEvent(event)