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

import cv2
import matplotlib.pyplot as plt
import heapq
from scipy import sparse
import osqp
import time


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

def _resize_image(source_image:cv2.typing.MatLike, image_box_size)-> cv2.typing.MatLike :
    re_image = source_image.copy()
    im_shape = re_image.shape
    im_width = im_shape[1]
    im_height = im_shape[0]
    box_width = image_box_size['width']
    box_height = image_box_size['height']
    print(f'image size: ({im_width}, {im_height})')
    box_ratio =  box_width / box_height
    image_ratio = im_width / im_height
    bordered_img = None
    if box_ratio > image_ratio:
        print('1')
        scale_ratio = box_height / im_height
        print(f'scale_ratio: {scale_ratio}')
        re_width = int(im_width * scale_ratio)
        re_height = int(im_height * scale_ratio)
        re_image = cv2.resize(re_image, (re_width, re_height), interpolation=cv2.INTER_AREA)
        # margin_width = int((box_width - re_width ) / 2)
        # top, bottom, left, right = 0, 0, margin_width, margin_width
        # border_color = [255, 255, 255]`
        # bordered_img = cv2.copyMakeBorder(re_image, top, bottom, left, right, cv2.BORDER_CONSTANT,
        #                                   value=border_color)
        # re_image = re_image.resize(x, y)
    else:
        print('2')
        scale_ratio = box_width / im_width
        print(f'scale_ratio: {scale_ratio}')
        re_width = int(im_width * scale_ratio)
        re_height = int(im_height * scale_ratio)
        re_image = cv2.resize(re_image, (re_width, re_height), interpolation=cv2.INTER_AREA)
        # re_image = re_image.resize(x, y)
        # margin_height = int((box_height - re_height) / 2)
        # top, bottom, left, right = margin_height, margin_height, 0, 0
        # border_color = [255, 255, 255]
        # bordered_img = cv2.copyMakeBorder(re_image, top, bottom, left, right, cv2.BORDER_CONSTANT,
        #                                   value=border_color)
    print(f'box size:({box_width},{box_height})')
    print(f're_image size:({re_width},{re_height})')
    return re_image, scale_ratio

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

        self.image_resize_gray_cv = None
        self.image_resize_gray_qt = None
        self.image_resize_gray_pixmap = None

        self.image_source_gray_cv = None
        self.image_source_gray_qt = None
        self.image_source_gray_pixmap = None

        self.image_resize_dilate_cv = None
        self.image_resize_dilate_qt = None
        self.image_resize_dilate_pixmap = None

        self.image_source_dilate_cv = None
        self.image_source_dilate_qt = None
        self.image_source_dilate_pixmap = None

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
        self.map_grid_list = None
        self.map_dilate = None
        self.map_dilate_list = None
        self.open_area = None
        self.image_search_path = None
        self.image_smooth_path = None

        # map parameter
        self.dilate_radius = 0
        self.dilate_iterations = 0
        self.src_point = {'x':0.0, 'y':0.0}
        self.dst_point = {'x':0.0, 'y':0.0}
        self.unreachable_thr = 255
        self.w_g = 0.6
        self.w_h = 1.0
        self.distance_method = 'euclidean' # 'euclidean' / 'manhattan' / 'diagonal'
        self.search_method = 'eight' # 'eight' / 'four'
        self.search_cost_time = None
        # matplotlib
        # self.map_grid_plt, self.map_grid_plt_ax = plt.subplots(figsize=(8, 8), dpi=200)
        # self.map_dilate_plt, self.map_dilate_plt_ax = plt.subplots(figsize=(8, 8), dpi=200)
        # self.cmap_1 = plt.get_cmap('YlGnBu')
        # self.cmap_2 = plt.get_cmap('summer')

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
        self.btn_dilate.clicked.connect(self.handle_btn_dilate)
        self.btn_inverse.clicked.connect(self.handle_btn_inverse)
        self.spin_dilate_radius.valueChanged.connect(self.handle_spin_dilate_radius)
        self.spin_dilate_iterations.valueChanged.connect(self.handle_spin_dilate_iterations)
        self.spin_dst_x.valueChanged.connect(self.handle_spin_dst_x)
        self.spin_dst_y.valueChanged.connect(self.handle_spin_dst_y)
        self.btn_set_src.clicked.connect(self.handle_btn_set_src)
        self.spin_dst_x.valueChanged.connect(self.handle_spin_dst_x)
        self.spin_dst_y.valueChanged.connect(self.handle_spin_dst_y)
        self.btn_set_dst.clicked.connect(self.handle_btn_set_dst)
        self.btn_start_search.clicked.connect(self.handle_btn_start_search)
        self.btn_set_search_param.clicked.connect(self.handle_btn_set_search_param)
        self.spin_unreachable_threshold.valueChanged.connect(self.handle_spin_unreachable_threshold)
        self.spin_weight_g.valueChanged.connect(self.handle_spin_weight_g)
        self.spin_weight_h.valueChanged.connect(self.handle_spin_weight_h)
        self.combo_search_method.activated.connect(self.handle_combo_search_method)
        self.combo_distance_method.activated.connect(self.handle_combo_distance_method)
        self.image_tab.currentChanged.connect(self.handle_image_tab)

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

    # # 调整图片大小适配图片框
    # def _resize_image(self, source_image:cv2.typing.MatLike)-> cv2.typing.MatLike :
    #     re_image = source_image.copy()
    #     im_shape = re_image.shape
    #     im_width = im_shape[1]
    #     im_height = im_shape[0]
    #     box_width = self.image_box_size['width']
    #     box_height = self.image_box_size['height']
    #     print(f'image size: ({im_width}, {im_height})')
    #     box_ratio =  box_width / box_height
    #     image_ratio = im_width / im_height
    #     bordered_img = None
    #     if box_ratio > image_ratio:
    #         print('1')
    #         self.scale_ratio = box_height / im_height
    #         print(f'scale_ratio: {self.scale_ratio}')
    #         re_width = int(im_width * self.scale_ratio)
    #         re_height = int(im_height * self.scale_ratio)
    #         re_image = cv2.resize(re_image, (re_width, re_height) )
    #         # margin_width = int((box_width - re_width ) / 2)
    #         # top, bottom, left, right = 0, 0, margin_width, margin_width
    #         # border_color = [255, 255, 255]
    #         # bordered_img = cv2.copyMakeBorder(re_image, top, bottom, left, right, cv2.BORDER_CONSTANT,
    #         #                                   value=border_color)
    #         # re_image = re_image.resize(x, y)
    #     else:
    #         print('2')
    #         self.scale_ratio = box_width / im_width
    #         print(f'scale_ratio: {self.scale_ratio}')
    #         re_width = int(im_width * self.scale_ratio)
    #         re_height = int(im_height * self.scale_ratio)
    #         re_image = cv2.resize(re_image, (re_width, re_height))
    #         # re_image = re_image.resize(x, y)
    #         # margin_height = int((box_height - re_height) / 2)
    #         # top, bottom, left, right = margin_height, margin_height, 0, 0
    #         # border_color = [255, 255, 255]
    #         # bordered_img = cv2.copyMakeBorder(re_image, top, bottom, left, right, cv2.BORDER_CONSTANT,
    #         #                                   value=border_color)
    #     print(f'box size:({box_width},{box_height})')
    #     print(f're_image size:({re_width},{re_height})')
    #     return re_image

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

            self.image_resize_cv, self.scale_ratio = _resize_image(self.image_source_cv, self.image_box_size)
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
            # resize
            self.image_resize_gray_cv = cv2.cvtColor(self.image_resize_cv, cv2.COLOR_BGR2GRAY)
            self.image_resize_gray_qt = _cv2qt_gray(self.image_resize_gray_cv)
            self.image_resize_gray_pixmap = QPixmap.fromImage(self.image_resize_gray_qt)
            self.image_current_pixmap = self.image_resize_gray_pixmap.copy()
            self.update_interact_image()
            # source
            self.image_source_gray_cv = cv2.cvtColor(self.image_source_cv, cv2.COLOR_BGR2GRAY)
            self.image_source_gray_qt = _cv2qt_gray(self.image_source_gray_cv)
            self.map_grid_list = np.array(self.image_source_gray_cv.tolist())
        elif self.image_process_mode == 'binary':
            # resize
            self.image_resize_gray_cv = cv2.cvtColor(self.image_resize_cv, cv2.COLOR_BGR2GRAY)
            _, self.image_resize_gray_cv = cv2.threshold(self.image_resize_gray_cv, self.bin_thr, 255, cv2.THRESH_BINARY)
            self.image_resize_gray_qt = _cv2qt_gray(self.image_resize_gray_cv)
            self.image_resize_gray_pixmap = QPixmap.fromImage(self.image_resize_gray_qt)
            self.image_current_pixmap = self.image_resize_gray_pixmap.copy()
            self.update_interact_image()
            # source
            self.image_source_gray_cv = cv2.cvtColor(self.image_source_cv, cv2.COLOR_BGR2GRAY)
            _, self.image_source_gray_cv = cv2.threshold(self.image_source_gray_cv, self.bin_thr, 255,
                                                         cv2.THRESH_BINARY)
            self.image_source_gray_qt = _cv2qt_gray(self.image_source_gray_cv)
            self.map_grid_list = np.array(self.image_source_gray_cv.tolist())
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

    # map process
    # dilate process
    def handle_btn_dilate(self):
        if self.image_resize_gray_cv is None:
            self.update_statusBar('no image to process', 2000)
            return
        else:
            # resize
            kernel_1 = np.ones((int(self.dilate_radius * self.scale_ratio), int(self.dilate_radius * self.scale_ratio)), dtype=np.uint8)
            kernel_2 = np.ones((self.dilate_radius, self.dilate_radius), dtype=np.uint8)
            # print(kernel_1.shape)
            # self.image_resize_dilate_cv = cv2.dilate(self.image_resize_gray_cv, kernel_1, iterations=self.dilate_iterations)
            # self.image_resize_dilate_qt = _cv2qt_gray(self.image_resize_dilate_cv)
            # self.image_resize_dilate_pixmap = QPixmap.fromImage(self.image_resize_dilate_qt)
            # self.image_current_pixmap = self.image_resize_dilate_pixmap.copy()
            # self.update_interact_image()
            # source
            self.image_source_dilate_cv = cv2.dilate(self.image_source_gray_cv, kernel_2, iterations=self.dilate_iterations)
            self.image_source_dilate_qt = _cv2qt_gray(self.image_source_dilate_cv)
            self.map_dilate_list = np.array(self.image_source_dilate_cv)
            # resize
            self.image_resize_dilate_cv, _ = _resize_image(self.image_source_dilate_cv, self.image_box_size)
            self.image_resize_dilate_qt = _cv2qt_gray(self.image_resize_dilate_cv)
            self.image_resize_dilate_pixmap = QPixmap.fromImage(self.image_resize_dilate_qt)
            self.image_current_pixmap = self.image_resize_dilate_pixmap.copy()
            self.update_interact_image()

    # 黑白反转
    def handle_btn_inverse(self):
        if self.image_resize_gray_cv is None:
            self.update_statusBar('no image to process', 2000)
            return
        else:
            # # resize
            # self.image_resize_gray_cv = 255 - self.image_resize_gray_cv
            # self.image_resize_gray_qt = _cv2qt_gray(self.image_resize_gray_cv)
            # self.image_resize_gray_pixmap = QPixmap.fromImage(self.image_resize_gray_qt)
            # self.image_current_pixmap = self.image_resize_gray_pixmap.copy()
            # self.update_interact_image()
            # source
            self.image_source_gray_cv = 255 - self.image_source_gray_cv
            self.image_source_gray_qt = _cv2qt_gray(self.image_source_gray_cv)
            self.map_grid_list = np.array(self.image_source_gray_cv)
            # resize
            self.image_resize_gray_cv, _ = _resize_image(self.image_source_gray_cv, self.image_box_size)
            self.image_resize_gray_qt = _cv2qt_gray(self.image_resize_gray_cv)
            self.image_resize_gray_pixmap = QPixmap.fromImage(self.image_resize_gray_qt)
            self.image_current_pixmap = self.image_resize_gray_pixmap.copy()
            self.update_interact_image()

    # set dilate parameter
    def handle_spin_dilate_radius(self, value):
        self.dilate_radius = value
        self.update_statusBar(f'set dilate radius: {value}', 2000)

    def handle_spin_dilate_iterations(self, value):
        self.dilate_iterations = value
        self.update_statusBar(f'set dilate iterations: {value}', 2000)

    # set start point
    def handle_spin_src_x(self, value):
        self.src_point['x'] = value
    def handle_spin_src_y(self, value):
        self.src_point['y'] = value
    # plan to use this function to handle graphical interaction, but now just update the spin value
    def handle_btn_set_src(self):
        self.src_point['x'] = self.spin_src_x.value()
        self.src_point['y'] = self.spin_src_y.value()
        self.update_statusBar(f'Start: ({self.src_point['x']}, {self.src_point['y']})')
    # set end point
    def handle_spin_dst_x(self, value):
        self.dst_point['x'] = value

    def handle_spin_dst_y(self, value):
        self.dst_point['y'] = value
    # plan to use this function to handle graphical interaction, but now just update the spin value
    def handle_btn_set_dst(self):
        self.dst_point['x'] = self.spin_dst_x.value()
        self.dst_point['y'] = self.spin_dst_y.value()
        self.update_statusBar(f'Target: ({self.dst_point['x']}, {self.dst_point['y']})')

    # set search parameter
    def handle_btn_set_search_param(self):
        print('handle_btn_set_search_param')
        pass
    def handle_btn_start_search(self):
        print('handle_btn_start_search')

        pass
    def handle_spin_unreachable_threshold(self, value):
        self.unreachable_thr = value
        self.update_statusBar(f'set unreachable threshold: {self.unreachable_thr}', 2000)

    def handle_spin_weight_g(self, value):
        self.w_g = value
        self.update_statusBar(f'set weight g: {self.w_g:.2f}', 2000)

    def handle_spin_weight_h(self, value):
        self.w_h = value
        self.update_statusBar(f'set weight h: {self.w_h:.2f}', 2000)

    def handle_combo_distance_method(self, index:int):
        if index == 0:
            self.distance_method = 'euclidean'
            self.update_statusBar(f'set distance method: {self.distance_method}', 2000)
        elif index == 1:
            self.distance_method = 'diagonal'
            self.update_statusBar(f'set distance method: {self.distance_method}', 2000)
        elif index == 2:
            self.distance_method = 'manhattan'
            self.update_statusBar(f'set distance method: {self.distance_method}', 2000)
        else:
            self.update_statusBar(f'unknown distance method: {index}', 2000)

    def handle_combo_search_method(self, index:int):
        if index == 0:
            self.search_method = 'eight'
            self.update_statusBar(f'set search method: {self.search_method}', 2000)
        elif index == 1:
            self.search_method = 'four'
            self.update_statusBar(f'set search method: {self.search_method}', 2000)
        else:
            self.update_statusBar(f'unknown search method: {index}', 2000)

    # set optimal parameters
    def handle_btn_set_optimal_param(self):
        pass
    def handle_btn_smooth_path(self):
        pass
    def handle_spin_weight_smooth(self, value):
        pass
    def handle_slider_weight_smooth(self, value):
        pass
    def handle_spin_weight_length(self, value):
        pass
    def handle_slider_weight_length(self, value):
        pass
    def handle_spin_weight_overlap(self, value):
        pass
    def handle_slider_weight_overlap(self, value):
        pass

    def handle_image_tab(self, index:int):
        print(f'handle_image_tab {index}')
        if index == 0:
            pass
        elif index == 1:
            pass
        elif index == 2:
            if self.map_grid_list is None:
                return
            else:
                fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
                cmap = plt.get_cmap('YlGnBu')
                ax.imshow(self.map_grid_list, cmap=cmap, interpolation='nearest')
                path = './map_grid.png'
                plt.savefig(path)
                image_source_cv = cv2.imread(path)
                image_source_cv = cv2.cvtColor(image_source_cv, cv2.COLOR_BGR2RGB)

                image_resize_cv, _ = _resize_image(image_source_cv, self.image_box_size)
                image_resize_qt = _cv2qt_rgb(image_resize_cv)
                self.image_source_gray_pixmap = QPixmap.fromImage(image_resize_qt)
                # self.image_current_pixmap = self.image_resize_pixmap.copy()
                self.label_map_grid.setPixmap(self.image_source_gray_pixmap)
        elif index == 3:
            if self.map_dilate_list is None:
                return
            else:
                fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
                cmap = plt.get_cmap('YlGnBu')
                ax.imshow(self.map_dilate_list, cmap=cmap, interpolation='nearest')
                path = './map_dilate.png'
                plt.savefig(path)
                image_source_cv = cv2.imread(path)
                image_source_cv = cv2.cvtColor(image_source_cv, cv2.COLOR_BGR2RGB)

                image_resize_cv, _ = _resize_image(image_source_cv, self.image_box_size)
                image_resize_qt = _cv2qt_rgb(image_resize_cv)
                self.image_source_dilate_pixmap = QPixmap.fromImage(image_resize_qt)
                # self.image_current_pixmap = self.image_resize_pixmap.copy()
                self.label_map_dilate.setPixmap(self.image_source_dilate_pixmap)
        elif index == 4:
            pass
        elif index == 5:
            pass
        else:
            pass

if __name__ == '__main__':
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())