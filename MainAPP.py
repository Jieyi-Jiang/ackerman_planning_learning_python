from mian_window import Ui_MainWindow
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel,
                             QPushButton, QVBoxLayout, QHBoxLayout,
                             QFileDialog, QScrollArea)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5 import QtCore, QtGui, QtWidgets

def _cv2qt_rgb(cv_image:cv2.typing.MatLike)->QImage:
    h, w, c = cv_image.shape
    bytes_per_line = 3 * w
    q_img = QImage(cv_image.data, w, h, bytes_per_line,
                   QImage.Format_RGB888)
    return  q_img
def _cv2qt_gray(cv_image:cv2.typing.MatLike)->QImage:
    h, w = cv_image.shape
    bytes_per_line = w
    q_img = QImage(cv_image.data, w, h, bytes_per_line,
                   QImage.Format_Grayscale8)
    return  q_img

class MainWindow(Ui_MainWindow, QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # parameter
        # image import and process
        # image data
        self.image_current_cv = None
        self.image_current_qt = None
        self.image_current_pixmap = None
        self.image_source_cv = None
        self.image_source_qt = None
        self.image_source_pixmap = None
        self.image_resize_cv = None
        self.image_resize_qt = None
        self.image_resize_pixmap = None
        self.image_gray_cv = None
        self.image_gray_qt = None
        self.image_gray_pixmap = None
        # image parameter
        self.image_path = ""
        self.image_detail_box_source_text = None
        # image process parameter
        self.image_process_mode = 'gray' # gray/binary
        self.bin_thr_min = 0
        self.bin_thr_max = 255
        self.bin_thr = 200
        self.image_size_source = {'width':0, 'height': 0}
        self.image_box_size = {'width':0, 'height': 0}
        self.scale_ratio = 1.0

        # map data
        self.map_grid = None
        self.map_dilate = None
        self.image_search_path = None
        self.image_smooth_path = None

        # map parameter
        self.dilate_radius = 0
        self.dilate_iterations = 0


        # function
        self.other_init()
        self.get_image_box_size()
        self.connect_list()

    def other_init(self):
        self.function_select_tab.setCurrentIndex(0)
        self.image_tab.setCurrentIndex(0)
        self.btn_gray.setChecked(True)
        pass

    def connect_list(self):
        self.btn_import_image.clicked.connect(self.handle_import_image)
        self.btn_gray.toggled.connect(self.handle_btn_gray)
        self.btn_binary.toggled.connect(self.handle_btn_binary)
        self.spin_bin_thr_min.valueChanged.connect(self.handle_spin_bin_thr_min)
        self.spin_bin_thr_max.valueChanged.connect(self.handle_spin_bin_thr_max)
        self.spin_bin_thr.valueChanged.connect(self.handle_spin_bin_thr)
        self.slider_bin_thr.valueChanged.connect(self.handle_slider_bin_thr)
        self.btn_image_process.clicked.connect(self.handle_btn_image_process)
        self.btn_inverse.clicked.connect(self.handle_btn_inverse)

    def update_interact_image(self):
        self.label_interact.setPixmap(self.image_current_pixmap)
        self.image_tab.setCurrentIndex(0)

    def update_source_image(self):
        self.label_image_source.setPixmap(self.image_resize_pixmap)

    def update_statusBar(self, msg_str:str, time_msc:int=2000):
        self.statusBar.showMessage(msg_str, time_msc)

    def update_image_detail_box(self):
            _translate = QtCore.QCoreApplication.translate
            self.image_detail_box_source_text = f"""
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">
<html><head><meta name="qrichtext" content="1" /><style type="text/css">
p, li {{ white-space: pre-wrap; }}
</style></head><body style=" font-family:'SimSun'; font-size:9pt; font-weight:400; font-style:normal;">
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">path:{self.image_path}</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">size:({self.image_size_source['width']}, {self.image_size_source['height']})</p>
<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;">cursor:(0, 0)</p></body></html>
            """
            self.image_detail_text_box.setHtml(_translate("MainWindow", self.image_detail_box_source_text))
            print('update box')

    def get_image_box_size(self):
        box_size = self.label_interact.size()
        self.image_box_size['width'] = box_size.width()
        self.image_box_size['height'] = box_size.height()
        # print(f'image box size:({self.image_box_size.width()}, {self.image_box_size.height()})')

    # 调整图片大小适配图片框
    def _resize_image(self, source_image:cv2.typing.MatLike)-> cv2.typing.MatLike :
        re_image = source_image.copy()
        im_shape = re_image.shape
        im_width = im_shape[1]
        im_height = im_shape[0]
        box_width = self.image_box_size['width']
        box_height = self.image_box_size['height']
        print(f'image size: ({im_width}, {im_height})')
        box_ratio =  box_width / box_height
        image_ratio = im_width / im_height
        bordered_img = None
        if box_ratio > image_ratio:
            print('1')
            self.scale_ratio = box_height / im_height
            re_width = int(im_width * self.scale_ratio)
            re_height = int(im_height * self.scale_ratio)
            re_image = cv2.resize(re_image, (re_width, re_height) )
            # margin_width = int((box_width - re_width ) / 2)
            # top, bottom, left, right = 0, 0, margin_width, margin_width
            # border_color = [255, 255, 255]
            # bordered_img = cv2.copyMakeBorder(re_image, top, bottom, left, right, cv2.BORDER_CONSTANT,
            #                                   value=border_color)
            # re_image = re_image.resize(x, y)
        else:
            print('2')
            self.scale_ratio = box_width / im_width
            re_width = int(im_width * self.scale_ratio)
            re_height = int(im_height * self.scale_ratio)
            re_image = cv2.resize(re_image, (re_width, re_height))
            # re_image = re_image.resize(x, y)
            # margin_height = int((box_height - re_height) / 2)
            # top, bottom, left, right = margin_height, margin_height, 0, 0
            # border_color = [255, 255, 255]
            # bordered_img = cv2.copyMakeBorder(re_image, top, bottom, left, right, cv2.BORDER_CONSTANT,
            #                                   value=border_color)
        print(f'box size:({box_width},{box_height})')
        print(f're_image size:({re_width},{re_height})')
        return re_image

    # import image
    def handle_import_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, '选择图片', '', '图片文件 (*.jpg *.png *.bmp)')
        if path:
            # 以opencv的格式读取图片
            self.image_path = path
            self.image_source_cv = cv2.imread(path)
            self.image_source_cv = cv2.cvtColor(self.image_source_cv, cv2.COLOR_BGR2RGB)
            im_shape = self.image_source_cv.shape
            self.image_size_source['width'] = im_shape[1]
            self.image_size_source['height'] = im_shape[0]
            self.image_resize_cv = self._resize_image(self.image_source_cv)
            image_resize_qt = _cv2qt_rgb(self.image_resize_cv)
            # h, w, c = self.cv_image.shape
            # bytes_per_line = 3 * w
            # q_img = QImage(self.cv_image.data, w, h, bytes_per_line,
            #                QImage.Format_RGB888)
            self.image_resize_pixmap = QPixmap.fromImage(image_resize_qt)
            self.image_current_pixmap = self.image_resize_pixmap.copy()
            self.update_interact_image()
            self.update_source_image()
            self.update_image_detail_box()
            self.update_statusBar(f"image path: {self.image_path}   |||   image size: ({self.image_size_source['width']}, {self.image_size_source['height']})")
            # self.label.setPixmap(self.current_pixmap)
            # self.points.clear()
            # self.point_selection_mode = False
        else:
            print('Path is invalid!')

    # image process
    def handle_btn_image_process(self):
        if self.image_source_cv is None:
            self.update_statusBar('no image to process', 2000)
            return
        if self.image_process_mode == 'gray':
            self.image_gray_cv = cv2.cvtColor(self.image_resize_cv, cv2.COLOR_BGR2GRAY)
            self.image_gray_qt = _cv2qt_gray(self.image_gray_cv)
            self.image_gray_pixmap = QPixmap.fromImage(self.image_gray_qt)
            self.image_current_pixmap = self.image_gray_pixmap.copy()
            self.update_interact_image()
        elif self.image_process_mode == 'binary':
            self.image_gray_cv = cv2.cvtColor(self.image_resize_cv, cv2.COLOR_BGR2GRAY)
            _, self.image_gray_cv = cv2.threshold(self.image_gray_cv, self.bin_thr, 255, cv2.THRESH_BINARY)
            self.image_gray_qt = _cv2qt_gray(self.image_gray_cv)
            self.image_gray_pixmap = QPixmap.fromImage(self.image_gray_qt)
            self.image_current_pixmap = self.image_gray_pixmap.copy()
            self.update_interact_image()
        else:
            self.update_statusBar('no image to process', 2000)

    # image process mode
    # select the process mode
    def handle_btn_gray(self):
        self.image_process_mode = 'gray'
        self.update_statusBar('set grad mode', 2000)

    def handle_btn_binary(self):
        self.image_process_mode = 'binary'
        self.update_statusBar('set binary mode', 2000)

    # set the minimum and maximum value of the binary threshold
    def handle_spin_bin_thr_min(self, value):
        self.bin_thr_min = value
        self.slider_bin_thr.setMinimum(value)
        self.update_statusBar(f'set binary threshold min: {value}', 2000)

    def handle_spin_bin_thr_max(self, value):
        self.bin_thr_max = value
        self.slider_bin_thr.setMaximum(value)
        self.update_statusBar(f'set binary threshold max: {value}', 2000)

    # set the binary value
    def handle_spin_bin_thr(self, value):
        self.bin_thr = value
        self.slider_bin_thr.setValue(value)
        self.update_statusBar(f'set binary threshold: {value}', 2000)

    def handle_slider_bin_thr(self, value):
        self.bin_thr = value
        self.spin_bin_thr.setValue(value)
        self.update_statusBar(f'set binary threshold: {value}', 2000)

    def handle_spin_dilate_radius(self, value):
        self.dilate_radius = value
        self.update_statusBar(f'set dilate radius: {value}', 2000)

    def handle_spin_spin_dilate_iterations(self, value):
        self.dilate_iterations = value
        self.update_statusBar(f'set dilate iterations: {value}', 2000)

    # map process
    #
    def handle_btn_inverse(self):
        if self.image_gray_cv is None:
            self.update_statusBar('no image to process', 2000)
            return
        else:
            self.image_gray_cv = 255 - self.image_gray_cv
            self.image_gray_qt = _cv2qt_gray(self.image_gray_cv)
            self.image_gray_pixmap = QPixmap.fromImage(self.image_gray_qt)
            self.image_current_pixmap = self.image_gray_pixmap.copy()
            self.update_interact_image()

if __name__ == '__main__':
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())