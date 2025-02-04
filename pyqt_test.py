import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel,
                             QPushButton, QVBoxLayout, QHBoxLayout,
                             QFileDialog, QScrollArea)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor
from PyQt5.QtCore import Qt, pyqtSignal


class ClickableLabel(QLabel):
    clicked = pyqtSignal(int, int)  # 发射原图坐标信号

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: white;")
        self.setScaledContents(False)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.pixmap():
            pixmap = self.pixmap()
            # 计算坐标转换
            label_w = self.width()
            label_h = self.height()
            pix_w = pixmap.width()
            pix_h = pixmap.height()

            ratio = min(label_w / pix_w, label_h / pix_h)
            display_w = int(pix_w * ratio)
            display_h = int(pix_h * ratio)

            x_offset = (label_w - display_w) // 2
            y_offset = (label_h - display_h) // 2

            x_in_label = event.x() - x_offset
            y_in_label = event.y() - y_offset

            if 0 <= x_in_label < display_w and 0 <= y_in_label < display_h:
                x_ori = int(x_in_label / ratio)
                y_ori = int(y_in_label / ratio)
                self.clicked.emit(x_ori, y_ori)
        super().mousePressEvent(event)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.original_pixmap = None
        self.current_pixmap = None
        self.cv_image = None
        self.points = []
        self.point_selection_mode = False

    def initUI(self):
        self.setWindowTitle('图像处理工具')
        self.setGeometry(100, 100, 800, 600)

        # 创建控件
        self.label = ClickableLabel()
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.label)
        self.scroll_area.setWidgetResizable(True)

        # 按钮
        self.btn_open = QPushButton('打开图片')
        self.btn_grayscale = QPushButton('灰度处理')
        self.btn_select = QPushButton('选择点')
        self.btn_draw = QPushButton('绘制连线')

        # 布局
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.btn_open)
        btn_layout.addWidget(self.btn_grayscale)
        btn_layout.addWidget(self.btn_select)
        btn_layout.addWidget(self.btn_draw)

        main_layout = QVBoxLayout()
        main_layout.addLayout(btn_layout)
        main_layout.addWidget(self.scroll_area)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # 连接信号
        self.btn_open.clicked.connect(self.open_image)
        self.btn_grayscale.clicked.connect(self.convert_grayscale)
        self.btn_select.clicked.connect(self.enable_point_selection)
        self.btn_draw.clicked.connect(self.draw_line)
        self.label.clicked.connect(self.handle_click)

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, '选择图片', '', '图片文件 (*.jpg *.png *.bmp)')
        if path:
            self.cv_image = cv2.imread(path)
            self.cv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
            h, w, c = self.cv_image.shape
            bytes_per_line = 3 * w
            q_img = QImage(self.cv_image.data, w, h, bytes_per_line,
                           QImage.Format_RGB888)
            self.original_pixmap = QPixmap.fromImage(q_img)
            self.current_pixmap = self.original_pixmap.copy()
            self.label.setPixmap(self.current_pixmap)
            self.points.clear()
            self.point_selection_mode = False

    def convert_grayscale(self):
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
            painter.drawEllipse(x - 3, y - 3, 6, 6)
            painter.end()
            self.label.setPixmap(pixmap)
            self.current_pixmap = pixmap
            self.points.append((x, y))

            if len(self.points) == 2:
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
    sys.exit(app.exec_())