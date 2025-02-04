import sys
from time import sleep
from turtle import Shape

import cv2
import cv2.mat_wrapper
import numpy as np
from PIL.ImageChops import screen
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel,
                             QPushButton, QVBoxLayout, QHBoxLayout,
                             QFileDialog, QScrollArea, QStyle, QFrame)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QIcon
from PyQt5.QtCore import Qt, pyqtSignal, QRect
import matplotlib.pyplot as plt
import heapq

from scipy import sparse
import osqp
import time

# 创建可点击对象的类，继承自标签类
class ClickableLabel(QLabel):
    # 发射原图坐标信号
    clicked = pyqtSignal(int, int)
    def __init__(self, parent=None):
        super().__init__(parent)

        # 对象居中对其
        self.setAlignment(Qt.AlignCenter)
        # 设置背景色为白色
        self.setStyleSheet("background-color: white;")
        # 失能缩放
        self.setScaledContents(True)

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
            ####################################################
            # 下面的代码有问题，需要修改
            ####################################################
            # 计算缩放率
            # ratio = min(label_w / pix_w, label_h / pix_h)
            ####################################################
            ### ratio = 1.0  就没问题了
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


def _cv2qt(cv_image:cv2.typing.MatLike)->QImage:
    h, w, c = cv_image.shape
    bytes_per_line = 3 * w
    q_img = QImage(cv_image.data, w, h, bytes_per_line,
                   QImage.Format_RGB888)
    return  q_img


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.source_image = None
        self.resized_image = None
        self.processed_image = None
        self.map_image = None
        self.original_pixmap = None
        self.current_pixmap = None
        self.cv_image = None
        self.points = []
        self.real_point = []
        self.point_selection_mode = False
        self.image_box_size = QRect()
        self.scale_ratio = 0.0
        # self.screen()

    def initUI(self):
        self.setWindowTitle('图像处理工具')
        # 无法正常加载自定义图标
        icon = QIcon("./icon.png")
        # 使用PyQt自带的图标
        # icon = QApplication.style().standardIcon()
        # print(icon)
        self.setWindowIcon(icon)
        # QApplication.primaryScreen().geometry()
        # self.setGeometry(100, 100, 800, 600)
        # print(self.screen().geometry())
        # print(QRect(0, 0, 100, 100))
        self.setGeometry(self.make_window_central( self.screen().geometry().width()/2, self.screen().geometry().height()/2) )

        # 创建控件
        # 可点击控件，用来放置滚动区域和图像
        self.label = ClickableLabel()
        # print(f'self.label: {self.label}')
        # self.scroll_area = QScrollArea()
        # self.scroll_area.setFrameShape(QFrame.Box)
        # self.scroll_area.setLineWidth(3)
        # self.scroll_area.setWidget(self.label)
        # self.scroll_area.setWidgetResizable(True)
        self.image_box_size = self.label.rect()
        print(f'the size of image box: {self.image_box_size}')
        # 按钮
        self.btn_open = QPushButton('打开图片')
        self.btn_image_process = QPushButton('处理图像')
        self.btn_select = QPushButton('选择起点终点')
        self.btn_draw = QPushButton('绘制路线')
        self.btn_resize = QPushButton('重绘')
        # 布局
        # 水平摆放控件，放置4个按钮
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.btn_open)
        btn_layout.addWidget(self.btn_image_process)
        btn_layout.addWidget(self.btn_select)
        btn_layout.addWidget(self.btn_draw)
        btn_layout.addWidget(self.btn_resize)
        # 垂直摆放控件，放置按钮和图片窗口
        main_layout = QVBoxLayout()
        main_layout.addLayout(btn_layout)
        main_layout.addWidget(self.label)
        # main_layout.addWidget(self.label)
        # 创建一个窗口部件
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)
        # 连接信号
        # 连接打开图片的信号和槽
        self.btn_open.clicked.connect(self.open_image)
        # 连接图像处理的信号和槽
        self.btn_image_process.clicked.connect(self.image_process)
        self.btn_select.clicked.connect(self.enable_point_selection)
        self.btn_draw.clicked.connect(self.draw_line)
        self.label.clicked.connect(self.handle_click)
        self.btn_resize.clicked.connect(self.resize_image)
        # self.image_box_size = self.scroll_area.rect()

    def resizeEvent(self, event):
        # self.image_box_size = self.scroll_area.rect()
        self.image_box_size = self.label.rect()
        # self.adjustSize()
        # print(self.image_box_size)

    def resize_image(self):
        if self.source_image is None:
            print('no source image.')
            return
        self.resized_image = self._resize_image(self.source_image)
        print(type(self.resized_image))
        self.cv_image = self.resized_image.copy()
        # print(self.cv_image)
        q_img = _cv2qt(self.cv_image)
        # h, w, c = self.cv_image.shape
        # bytes_per_line = 3 * w
        # q_img = QImage(self.cv_image.data, w, h, bytes_per_line,
        #                QImage.Format_RGB888)
        self.original_pixmap = QPixmap.fromImage(q_img)
        self.current_pixmap = self.original_pixmap.copy()
        self.label.setPixmap(self.current_pixmap)
        self.points.clear()
        self.point_selection_mode = False

    def make_window_central(self, width=800, height=600):
        my_screen = self.screen().geometry()
        _x = int((my_screen.width()-width)//2)
        _y = int(my_screen.height()-height)//2
        _width = int(width)
        _height = int(height)
        return QRect(_x, _y, _width, _height)

    # 调整图片大小适配图片框
    def _resize_image(self, source_image:cv2.typing.MatLike)-> cv2.typing.MatLike :
        re_image = source_image.copy()
        im_shape = re_image.shape
        im_width = im_shape[1]
        im_height = im_shape[0]
        box_width = self.image_box_size.width()
        box_height = self.image_box_size.height()
        # print(f'image box size:({self.image_box_size.width()}, {self.image_box_size.height()})')
        # print(f'image size: {source_image.shape}')
        box_ratio =  box_width / box_height
        image_ratio = im_width / im_height
        bordered_img = None
        if box_ratio > image_ratio:
            # print('1')
            self.scale_ratio = self.image_box_size.height() / im_height
            re_width = int(im_width * self.scale_ratio)
            re_height = int(im_height * self.scale_ratio)
            re_image = cv2.resize(re_image, (re_width, re_height) )
            margin_width = int((box_width - re_width ) / 2)
            top, bottom, left, right = 0, 0, margin_width, margin_width
            border_color = [255, 255, 255]
            bordered_img = cv2.copyMakeBorder(re_image, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                              value=border_color)
            # re_image = re_image.resize(x, y)
        else:
            # print('2')
            self.scale_ratio = self.image_box_size.width() / im_width
            re_width = int(im_width * self.scale_ratio)
            re_height = int(im_height * self.scale_ratio)
            re_image = cv2.resize(re_image, (re_width, re_height))
            # re_image = re_image.resize(x, y)
            margin_height = int((box_height - re_height) / 2)
            top, bottom, left, right = margin_height, margin_height, 0, 0
            border_color = [255, 255, 255]
            bordered_img = cv2.copyMakeBorder(re_image, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                              value=border_color)
        print(f'box size:({self.image_box_size.width()},{self.image_box_size.height()})')
        print(f're_image size:({re_width},{re_height})')
        return bordered_img

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, '选择图片', '', '图片文件 (*.jpg *.png *.bmp)')
        if path:
            # 以opencv的格式读取图片
            self.source_image = cv2.imread(path)
            self.source_image = cv2.cvtColor(self.source_image, cv2.COLOR_BGR2RGB)
            # print(self.cv_image.shape)
            self.resized_image = self._resize_image(self.source_image)
            # print(type(self.resized_image))
            self.cv_image = self.resized_image.copy()
            # print(self.cv_image)
            q_img = _cv2qt(self.cv_image)
            # h, w, c = self.cv_image.shape
            # bytes_per_line = 3 * w
            # q_img = QImage(self.cv_image.data, w, h, bytes_per_line,
            #                QImage.Format_RGB888)
            self.original_pixmap = QPixmap.fromImage(q_img)
            self.current_pixmap = self.original_pixmap.copy()
            self.label.setPixmap(self.current_pixmap)
            self.points.clear()
            self.point_selection_mode = False
        else:
            print('Path is invalid!')

    def image_process(self):
        if self.cv_image is not None:
            gray = cv2.cvtColor(self.cv_image, cv2.COLOR_RGB2GRAY)
            h, w = gray.shape
            q_img = QImage(gray.data, w, h, w, QImage.Format_Grayscale8)
            self.current_pixmap = QPixmap.fromImage(q_img)
            self.label.setPixmap(self.current_pixmap)

    def enable_point_selection(self):
        if self.original_pixmap:
            self.point_selection_mode = True
            self.points.clear()
            self.label.setPixmap(self.original_pixmap)
            self.current_pixmap = self.original_pixmap.copy()

    def handle_click(self, x, y):
        if self.point_selection_mode and len(self.points) < 2:
            # 绘制红点
            pixmap = self.current_pixmap.copy()
            painter = QPainter(pixmap)
            painter.setPen(QColor(255, 0, 0))
            radius = int(3*self.scale_ratio)
            # painter.drawEllipse(x - 3, y - 3, 6, 6)
            painter.drawEllipse(x - radius, y - radius, 2*radius, 2*radius)
            painter.end()
            self.label.setPixmap(pixmap)
            self.current_pixmap = pixmap
            self.points.append((x, y))
            self.real_point.append(( int(x/self.scale_ratio), int(y/self.scale_ratio) ))
            if len(self.points) == 2:
                print(self.real_point)
                self.point_selection_mode = False

    def draw_line(self):
        if len(self.points) == 2 and self.original_pixmap:
            # 在原始图片上绘制绿线
            pixmap = self.original_pixmap.copy()
            painter = QPainter(pixmap)
            painter.setPen(QColor(0, 255, 0))
            x1, y1 = self.points[0]
            x2, y2 = self.points[1]
            painter.drawLine(x1, y1, x2, y2)
            painter.end()
            self.label.setPixmap(pixmap)
            self.current_pixmap = pixmap

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    # window.image_box_size = window.scroll_area.rect()
    window.image_box_size = window.label.rect()
    sys.exit(app.exec_())