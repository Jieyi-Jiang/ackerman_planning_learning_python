import os

from numpy import dtype

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

# 八邻域搜索
#  ------------------------------------------>
#  |  x(i-1, j-1)   x(x, j-1)   x(i+1, j-1)  |      0   1   2
#  |  x(i-1, j)     x(i, j)     x(i+1, j)    |      7   X   3
#  |  x(i-1, j+1)   x(i, j+1)   x(i+1, j+1)  |      6   5   4
#  <------------------------------------------

# 四邻域搜索
#  ------------------------------------------>
#  |                x(x, j-1)                |          0
#  |  x(i-1, j)     x(i, j)     x(i+1, j)    |      3   X   1
#  |                x(i, j+1)                |          2
#  <------------------------------------------


# optimal the algorithm based on grid map
class Node:
    def __init__(self, x, y, cost, parent=None):
        self.position = [x, y]
        self.parent = parent
        self.cost = cost
        self.g = 0.0  # 从起点到当前节点的实际代价
        self.h = 0.0  # 从当前节点到终点的启发式代价
        self.f = 0.0  # 总代价 f = g + h

    def __eq__(self, other):
        return self.f == other.f

    def __lt__(self, other):
        return self.f < other.f

    def __repr__(self):
        return f"Node({self.position}, cost={self.cost}, g={self.g}, h={self.h}, f={self.f})"

    def same_point(self, other):
        if self.position[0] == other.position[0] and self.position[1] == other.position[1]:
            return True
        else:
            return False



def distance(a, b, method='euclidean'):
    if method == 'euclidean':
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    elif method == 'manhattan':
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    elif method == 'diagonal':
        x_dis = abs(a[0] - b[0])
        y_dis = abs(a[1] - b[1])
        return abs(x_dis - y_dis) + np.sqrt(2)*min([x_dis, y_dis])


def a_in_list(a:Node, node_list : list[Node]):
    for i in range(0, len(node_list)):
        node = node_list[i]
        if a.same_point(node):
            return i, node
    return -1, None

def a_star_search(start, goal, grid, threshold=np.inf, w_g = 1.0, w_h = 1.0, dis_method='euclidean', search_method='four'):
    open_list = []
    closed_list = []
    grid_shape = grid.shape
    open_area = np.zeros(grid_shape, dtype=np.uint8)
    closed_area = np.zeros(grid_shape, dtype=np.uint8)
    start_node = Node(start[0], start[1], grid[start[0]][start[1]])
    start_node.h = distance(start_node.position, start_node.position, dis_method)
    start_node.f = start_node.h
    goal_node = Node(goal[0], goal[1], grid[goal[0]][goal[1]])
    if search_method == 'eight':
        search_list_dir = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)]
    elif search_method == 'four':
        search_list_dir = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    else:
        search_list_dir = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    print('start:', start_node)
    print('goal:', goal_node)
    print(f"w_g: {w_g}, w_h:{w_h}, dis_method: {dis_method}, search_method: {search_method}")
    heapq.heappush(open_list, start_node)
    open_area[start[0],  start[1]] = 1
    cnt = 0
    while open_list:
        # print(cnt)
        cnt += 1
        current_node = heapq.heappop(open_list)
        # print(f"{cnt} select: ", current_node)
        if current_node.same_point(goal_node):
            total_cost = current_node.g
            print("total cost:", total_cost)
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            # print('open list =====================================================\n')
            # for node in open_list:
            #     print(node)
            # print('closed list ===================================================\n', closed_list)
            # for node in closed_list:
            #     print(node)
            return open_area, closed_area, path[::-1]

        closed_list.append(current_node)
        closed_area[current_node.position[0], current_node.position[1]] = 1
        # 八邻域搜索
        #  ------------------------------------------>
        #  |  x(i-1, j-1)   x(x, j-1)   x(i+1, j-1)  |      0   1   2
        #  |  x(i-1, j)     x(i, j)     x(i+1, j)    |      7   X   3
        #  |  x(i-1, j+1)   x(i, j+1)   x(i+1, j+1)  |      6   5   4
        #  <------------------------------------------

        # 四邻域搜索
        #  ------------------------------------------>
        #  |                x(x, j-1)                |          0
        #  |  x(i-1, j)     x(i, j)     x(i+1, j)    |      3   X   1
        #  |                x(i, j+1)                |          2
        #  <------------------------------------------

        # for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
        for di, dj in search_list_dir:
            neighbor_position = [current_node.position[0] + di, current_node.position[1] + dj]
            # 超出地图范围则忽略
            if not (0 <= neighbor_position[0] < grid_shape[0] and 0 <= neighbor_position[1] < grid_shape[1]):
                continue
            cost = grid[neighbor_position[0], neighbor_position[1]]
            neighbor_node = Node(neighbor_position[0], neighbor_position[1], cost, current_node)
            neighbor_node.g = current_node.g + neighbor_node.cost
            neighbor_node.h = distance(neighbor_node.position, goal_node.position, dis_method)
            # neighbor_node.h = distance(neighbor_node.position, goal_node.position, 'diagonal')
            neighbor_node.f = w_g * neighbor_node.g + w_h * neighbor_node.h

            # 超过阈值视为障碍物，忽略
            if cost >= threshold:
                continue

            # 如果在open表中，则处理
            n_index, old_node = a_in_list(neighbor_node, open_list)
            if n_index == -1:
                pass
            elif neighbor_node.g < old_node.g:
                open_list[n_index].g = neighbor_node.g
                open_list[n_index].f = neighbor_node.f
                open_list[n_index].parent = neighbor_node.parent
                continue
            else:
                continue

            # 如果在closed表中，则处理
            n_index, old_node = a_in_list(neighbor_node, closed_list)
            if n_index == -1:
                pass
            elif neighbor_node.g < old_node.g:
                closed_list[n_index].g = neighbor_node.g
                closed_list[n_index].f = neighbor_node.f
                closed_list[n_index].parent = neighbor_node.parent
                continue
                # del closed_list[n_index]
            else:
                continue


            # if not any(node.g <= neighbor_node.g for node in open_list):
            #     heapq.heappush(open_list, neighbor_node)
            heapq.heappush(open_list, neighbor_node)
            open_area[neighbor_node.position[0], neighbor_node.position[1]] = 1
            # print(neighbor_node)
            # print(open_list)
    return None

def make_matrix_P(n=100, w_A=1.0, w_B=1.0, w_C=1.0):
    m = n*2
    A = np.zeros((m, m))
    B = np.zeros((m, m))
    C = np.eye(m, m)
    for i in range(0, n - 1):
        A[i:i+2, i:i+2] += [[1, -1], [-1, 1]]
    A[n:, n:] = A[0:n, 0:n]
    for i in range(0, n - 2):
        B[i:i + 3, i:i + 3] += [[1, -2, 1], [-2, 4, -2], [1, -2, 1]]
    B[n:, n:] = B[0:n, 0:n]
    _P = w_A*A + w_B*B + w_C*C
    _P = _P * 2.0
    # point_xy = np.zeros((m, 1))
    # point_xy[0:n, 0] = pts[:, 0]
    # point_xy[n:, 0] = pts[:, 1]
    # _P = _P * point_xy
    return sparse.csc_matrix(_P)

def make_matrix_Q(pts, n=100):
    m = n*2
    _Q = np.zeros((m, 1))
    for i in range(0, m):
        _Q[i, 0] = -2
    point_xy = np.zeros((m, 1))
    point_xy[0:n, 0] = pts[:, 0]
    point_xy[n:, 0] = pts[:, 1]
    _Q = _Q * point_xy
    return _Q

def make_matrix_A(pts, n=100):
    m = n*2
    # A = np.zeros((m, m))
    _A = np.eye(m, m)
    return sparse.csc_matrix(_A)

def make_matrix_l(pts, n=100, R=np.inf):
    m = n*2
    _l = np.zeros((m, 1))
    point_xy = np.zeros((m, 1))
    point_xy[0:n, 0] = pts[:, 0]
    point_xy[n:, 0] = pts[:, 1]
    # for i in range(0, m):
    #     _l[i, 0] = -np.inf
    _l = point_xy - R
    # 等式约束起点和终点
    _l[0, 0]    = pts[0, 0]
    _l[n-1, 0]  = pts[n-1, 0]
    _l[n, 0]    = pts[0, 1]
    _l[m-1, 0]  = pts[n-1, 1]
    return _l

def make_matrix_u(pts, n=100, R=np.inf):
    m = n*2
    _u = np.zeros((m, 1))
    point_xy = np.zeros((m, 1))
    point_xy[0:n, 0] = pts[:, 0]
    point_xy[n:, 0] = pts[:, 1]
    # for i in range(0, m):
    #     _u[i, 0] = +np.inf
    _u = point_xy + R
    # 等式约束起点和终点
    _u[0, 0]   = pts[0, 0]
    _u[n-1, 0] = pts[n - 1, 0]
    _u[n, 0]   = pts[0, 1]
    _u[m-1, 0] = pts[n - 1, 1]
    return _u


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
    # print(f'image size: ({im_width}, {im_height})')
    box_ratio =  box_width / box_height
    image_ratio = im_width / im_height
    if box_ratio > image_ratio:
        # print('1')
        scale_ratio = box_height / im_height
        # print(f'scale_ratio: {scale_ratio}')
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
        # print('2')
        scale_ratio = box_width / im_width
        # print(f'scale_ratio: {scale_ratio}')
        re_width = int(im_width * scale_ratio)
        re_height = int(im_height * scale_ratio)
        re_image = cv2.resize(re_image, (re_width, re_height), interpolation=cv2.INTER_AREA)
        # re_image = re_image.resize(x, y)
        # margin_height = int((box_height - re_height) / 2)
        # top, bottom, left, right = margin_height, margin_height, 0, 0
        # border_color = [255, 255, 255]
        # bordered_img = cv2.copyMakeBorder(re_image, top, bottom, left, right, cv2.BORDER_CONSTANT,
        #                                   value=border_color)
    # print(f'box size:({box_width},{box_height})')
    # print(f're_image size:({re_width},{re_height})')
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

        self.image_search_pixmap = None
        self.image_optimal_pixmap = None

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
        self.unreachable_thr = 40
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

        # search parameter
        self.search_time = 0
        self.open_area = None
        self.closed_area = None
        self.search_path = None

        # optimal parameter
        self.smooth_cost = 1.0
        self.length_cost = 1.0
        self.overlap_cost = 1.0
        self.position_constrain = 1.0
        self.smooth_result = {'x':None, 'y':None}
        # function
        self.other_init()
        self.get_image_box_size()
        self.connect_list()

    def other_init(self):
        self.function_select_tab.setCurrentIndex(0)
        self.image_tab.setCurrentIndex(0)
        self.btn_gray.setChecked(True)
        self.spin_unreachable_threshold.setValue(self.unreachable_thr)
        self.spin_bin_thr.setValue(self.bin_thr)
        self.slider_bin_thr.setValue(self.bin_thr)
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
        # self.btn_set_search_param.clicked.connect(self.handle_btn_set_search_param)
        self.spin_unreachable_threshold.valueChanged.connect(self.handle_spin_unreachable_threshold)
        self.spin_weight_g.valueChanged.connect(self.handle_spin_weight_g)
        self.spin_weight_h.valueChanged.connect(self.handle_spin_weight_h)
        self.combo_search_method.activated.connect(self.handle_combo_search_method)
        self.combo_distance_method.activated.connect(self.handle_combo_distance_method)
        self.image_tab.currentChanged.connect(self.handle_image_tab)
        # optimal param set
        self.spin_weight_smooth.valueChanged.connect(self.handle_spin_weight_smooth)
        self.slider_weight_smooth.valueChanged.connect(self.handle_slider_weight_smooth)
        self.spin_weight_length.valueChanged.connect(self.handle_spin_weight_length)
        self.slider_weight_length.valueChanged.connect(self.handle_slider_weight_length)
        self.spin_weight_overlap.valueChanged.connect(self.handle_spin_weight_overlap)
        self.slider_weight_overlap.valueChanged.connect(self.handle_slider_weight_overlap)
        self.spin_position_constrain.valueChanged.connect(self.handle_spin_position_constrain)
        self.slider_positionr_constrain.valueChanged.connect(self.handle_slider_position_constrain)
        self.btn_smooth_path.clicked.connect(self.handle_btn_smooth_path)
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
            self.spin_src_x.setMaximum(self.image_size_source['width'] - 1)
            self.spin_src_y.setMaximum(self.image_size_source['height'] - 1)
            self.spin_dst_x.setMaximum(self.image_size_source['width'] - 1)
            self.spin_dst_y.setMaximum(self.image_size_source['height'] - 1)
        else:
            print('Path is invalid!')

    # image process
    def handle_btn_image_process(self):
        if self.image_source_cv is None:
            self.update_statusBar('no image to process', 2000)
            return
        if self.image_process_mode == 'gray':
            # source
            self.image_source_gray_cv = cv2.cvtColor(self.image_source_cv, cv2.COLOR_BGR2GRAY)
            print(type(self.image_source_gray_cv))
            self.image_source_gray_qt = _cv2qt_gray(self.image_source_gray_cv)
            self.map_grid_list = np.array(self.image_source_gray_cv.tolist(), dtype=np.uint32)
            self.map_dilate_list = self.map_grid_list
            # resize
            # self.image_resize_gray_cv, _ = _resize_image(self.image_source_gray_cv, self.image_box_size)
            # print(type(self.image_resize_gray_cv))
            # self.image_resize_gray_qt = _cv2qt_gray(self.image_resize_gray_cv)
            # self.image_resize_gray_pixmap = QPixmap.fromImage(self.image_resize_gray_qt)
            # self.image_current_pixmap = self.image_resize_gray_pixmap.copy()
            ############## matplotlib
            fig, ax = plt.subplots(figsize=(7, 7), dpi=200)
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
            os.remove(path)
            ###############
            self.image_current_pixmap = self.image_source_gray_pixmap
            self.update_interact_image()
        elif self.image_process_mode == 'binary':
            # source
            self.image_source_gray_cv = cv2.cvtColor(self.image_source_cv, cv2.COLOR_BGR2GRAY)
            shape = self.image_source_gray_cv.shape
            for i in range(0, shape[0]):
                for j in range(0, shape[1]):
                    if self.image_source_gray_cv[i][j] > self.bin_thr:
                        self.image_source_gray_cv[i][j] = 255
                    else:
                        self.image_source_gray_cv[i][j] = 0
            # _, self.image_source_gray_cv = cv2.threshold(self.image_source_gray_cv, self.bin_thr, 255,
            #                                              cv2.THRESH_BINARY)
            self.image_source_gray_qt = _cv2qt_gray(self.image_source_gray_cv)
            self.map_grid_list = np.array(self.image_source_gray_cv.tolist(), dtype=np.uint32)
            print(type(self.map_grid_list[0][0]))
            self.map_dilate_list = self.map_grid_list
            # resize
            # self.image_resize_gray_cv, _ = _resize_image(self.image_source_gray_cv, self.image_box_size)
            # self.image_resize_gray_qt = _cv2qt_gray(self.image_resize_gray_cv)
            # self.image_resize_gray_pixmap = QPixmap.fromImage(self.image_resize_gray_qt)
            # self.image_current_pixmap = self.image_resize_gray_pixmap.copy()
            # self.update_interact_image()
            ############## matplotlib
            fig, ax = plt.subplots(figsize=(7, 7), dpi=200)
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
            os.remove(path)
            ###############
            self.image_current_pixmap = self.image_source_gray_pixmap
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

    # map process
    # dilate process
    def handle_btn_dilate(self):
        if self.image_source_gray_cv is None:
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
            self.map_dilate_list = np.array(self.image_source_dilate_cv, dtype=np.uint32)
            # resize
            # self.image_resize_dilate_cv, _ = _resize_image(self.image_source_dilate_cv, self.image_box_size)
            # self.image_resize_dilate_qt = _cv2qt_gray(self.image_resize_dilate_cv)
            # self.image_resize_dilate_pixmap = QPixmap.fromImage(self.image_resize_dilate_qt)
            # self.image_tab.setCurrentIndex(3)
            # self.image_current_pixmap = self.image_resize_dilate_pixmap.copy()
            # self.update_interact_image()
            #################################
            fig, ax = plt.subplots(figsize=(7, 7), dpi=200)
            cmap = plt.get_cmap('YlGnBu')
            ax.imshow(self.map_dilate_list, cmap=cmap, interpolation='nearest')
            path = './map_dilate.png'
            plt.savefig(path)
            image_source_cv = cv2.imread(path)
            image_source_cv = cv2.cvtColor(image_source_cv, cv2.COLOR_BGR2RGB)

            image_resize_cv, _ = _resize_image(image_source_cv, self.image_box_size)
            image_resize_qt = _cv2qt_rgb(image_resize_cv)
            self.image_source_dilate_pixmap = QPixmap.fromImage(image_resize_qt)
            self.label_map_dilate.setPixmap(self.image_source_dilate_pixmap)
            os.remove(path)
            ########################
            self.image_current_pixmap = self.image_source_dilate_pixmap
            self.update_interact_image()

    # 黑白反转
    def handle_btn_inverse(self):
        if self.image_source_gray_cv is None:
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
            self.map_grid_list = np.array(self.image_source_gray_cv, dtype=np.uint32)
            self.map_dilate_list = self.map_grid_list
            # # resize
            # self.image_resize_gray_cv, _ = _resize_image(self.image_source_gray_cv, self.image_box_size)
            # self.image_resize_gray_qt = _cv2qt_gray(self.image_resize_gray_cv)
            # self.image_resize_gray_pixmap = QPixmap.fromImage(self.image_resize_gray_qt)
            # self.image_current_pixmap = self.image_resize_gray_pixmap.copy()
            # self.update_interact_image()
            ############## matplotlib
            fig, ax = plt.subplots(figsize=(7, 7), dpi=200)
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
            os.remove(path)
            ###############
            self.image_current_pixmap = self.image_source_gray_pixmap
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

    # # set search parameter
    # def handle_btn_set_search_param(self):
    #     print('handle_btn_set_search_param')
    #     pass

    def handle_btn_start_search(self):
        if self.map_dilate_list is None:
            self.update_statusBar("map is null")
            return
        self.open_area, self.closed_area, self.search_path  = None, None, None
        start_time = time.time()
        start = [self.src_point['y'], self.src_point['x']]
        end = [self.dst_point['y'], self.dst_point['x']]
        print(type(self.map_grid_list[0][0]))
        self.map_dilate_list = self.map_dilate_list + 1
        ret_val = a_star_search(
            start,
            end,
            self.map_dilate_list,
            self.unreachable_thr,
            self.w_g,
            self.w_h,
            self.distance_method,
            self.search_method
        )
        end_time = time.time()
        self.search_time = end_time - start_time
        self.update_statusBar(f"A*算法执行时间: {self.search_time} 秒")
        if ret_val is None:
            self.update_statusBar('fail to search')
            return
        self.open_area, self.closed_area, self.search_path = ret_val
        print(len(self.search_path))
        # print('handle_btn_start_search')
        for i in self.search_path:
            temp = i[0]
            i[0] = i[1]
            i[1] = temp
        ################################
        if self.search_path is None:
            self.update_statusBar('search path is null!')
            return
        fig, ax = plt.subplots(figsize=(7, 7), dpi=200)
        cmap = plt.get_cmap('YlGnBu')
        ax.imshow(self.map_grid_list, cmap=cmap, interpolation='nearest')
        cmap = plt.get_cmap('summer')
        masked1 = np.ma.masked_where(self.open_area == 0, self.open_area)
        ax.imshow(masked1, cmap=cmap, interpolation='nearest', alpha=0.5)
        path_x, path_y = zip(*self.search_path)
        ax.plot(path_x, path_y, color='blue', linestyle='--', linewidth=1)
        ax.plot(self.src_point['x'], self.src_point['y'], color='blue', marker='o', markersize=10)
        ax.plot(self.dst_point['x'], self.dst_point['y'], color='green', marker='*', markersize=20)
        path = './map_search.png'
        plt.savefig(path)
        # plt.show()
        image_source_cv = cv2.imread(path)
        image_source_cv = cv2.cvtColor(image_source_cv, cv2.COLOR_BGR2RGB)

        image_resize_cv, _ = _resize_image(image_source_cv, self.image_box_size)
        image_resize_qt = _cv2qt_rgb(image_resize_cv)
        self.image_search_pixmap = QPixmap.fromImage(image_resize_qt)
        # self.image_current_pixmap = self.image_resize_pixmap.copy()
        self.label_map_search.setPixmap(self.image_search_pixmap)
        os.remove(path)
        ###################
        self.image_current_pixmap = self.image_search_pixmap
        self.update_interact_image()

    def handle_spin_unreachable_threshold(self, value:int):
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
    # def handle_btn_set_optimal_param(self):
    #     pass
    def handle_btn_smooth_path(self):
        if self.search_path is None:
            self.update_statusBar('search path is null!')
            return
        points = np.array(self.search_path)
        p_shape = points.shape
        v_num = p_shape[0]
        P = make_matrix_P(v_num, 1.0, 1.0, 1.0)
        Q = make_matrix_Q(points, v_num)
        A = make_matrix_A(points, v_num)
        R = np.inf
        R = 2
        l = make_matrix_l(points, v_num, R)
        u = make_matrix_u(points, v_num, R)
        prob = osqp.OSQP()
        prob.setup(P, Q, A, l, u, alpha=1.0)
        res = prob.solve()
        result = res.x
        result_x = result[0:v_num]
        result_y = result[v_num:]
        self.smooth_result['x'] = result_x
        self.smooth_result['y'] = result_y
        #####################################
        if self.smooth_result['x'] is None:
            self.update_statusBar('optimal path is null!')
            return
        fig, ax = plt.subplots(figsize=(7, 7), dpi=200)
        cmap = plt.get_cmap('YlGnBu')
        ax.imshow(self.map_grid_list, cmap=cmap, interpolation='nearest')
        cmap = plt.get_cmap('summer')
        masked1 = np.ma.masked_where(self.open_area == 0, self.open_area)
        ax.imshow(masked1, cmap=cmap, interpolation='nearest', alpha=0.5)
        path_x, path_y = zip(*self.search_path)
        ax.plot(path_x, path_y, color='blue', linestyle='--', linewidth=1)
        ax.plot(self.smooth_result['x'], self.smooth_result['y'], color='red', linestyle='-', linewidth=2)
        ax.plot(self.src_point['x'], self.src_point['y'], color='blue', marker='o', markersize=10)
        ax.plot(self.dst_point['x'], self.dst_point['y'], color='green', marker='*', markersize=20)
        path = './map_path_smooth.png'
        plt.savefig(path)
        # plt.show()
        image_source_cv = cv2.imread(path)
        image_source_cv = cv2.cvtColor(image_source_cv, cv2.COLOR_BGR2RGB)

        image_resize_cv, _ = _resize_image(image_source_cv, self.image_box_size)
        image_resize_qt = _cv2qt_rgb(image_resize_cv)
        self.image_optimal_pixmap = QPixmap.fromImage(image_resize_qt)
        # self.image_current_pixmap = self.image_resize_pixmap.copy()
        self.label_smooth.setPixmap(self.image_optimal_pixmap)
        os.remove(path)
        #####################
        self.image_current_pixmap = self.image_optimal_pixmap
        self.update_interact_image()

    cost_wight_min = 0.0
    cost_weight_max = 10
    def handle_spin_weight_smooth(self, value):
        self.smooth_cost = value
        set_value = int(value * 10)
        self.slider_weight_smooth.setValue(set_value)
        pass
    def handle_slider_weight_smooth(self, value):
        self.smooth_cost = value
        set_value = float(value / 10)
        self.spin_weight_smooth.setValue(set_value)
        pass
    def handle_spin_weight_length(self, value):
        self.length_cost = value
        set_value = int(value * 10)
        self.slider_weight_length.setValue(set_value)
        pass
    def handle_slider_weight_length(self, value):
        self.length_cost = value
        set_value = float(value / 10)
        self.spin_weight_length.setValue(set_value)
        pass
    def handle_spin_weight_overlap(self, value):
        self.overlap_cost = value
        set_value = int(value * 10)
        self.slider_weight_overlap.setValue(set_value)
        pass
    def handle_slider_weight_overlap(self, value):
        self.overlap_cost = value
        set_value = float(value / 10)
        self.spin_weight_overlap.setValue(set_value)
        pass
    def handle_spin_position_constrain(self, value):
        self.position_constrain = value
        set_value = int(value * 10)
        self.slider_positionr_constrain.setValue(set_value)
    def handle_slider_position_constrain(self, value):
        self.position_constrain = value
        set_value = float(value / 10)
        self.spin_position_constrain.setValue(set_value)

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
                fig, ax = plt.subplots(figsize=(7, 7), dpi=200)
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
                os.remove(path)
        elif index == 3:
            if self.map_dilate_list is None:
                return
            else:
                fig, ax = plt.subplots(figsize=(7, 7), dpi=200)
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
                os.remove(path)
        elif index == 4:
            if self.search_path is None:
                self.update_statusBar('search path is null!')
                return
            fig, ax = plt.subplots(figsize=(7, 7), dpi=200)
            cmap = plt.get_cmap('YlGnBu')
            ax.imshow(self.map_grid_list, cmap=cmap, interpolation='nearest')
            cmap = plt.get_cmap('summer')
            masked1 = np.ma.masked_where(self.open_area == 0, self.open_area)
            ax.imshow(masked1, cmap=cmap, interpolation='nearest', alpha=0.5)
            path_x, path_y = zip(*self.search_path)
            ax.plot(path_x, path_y, color='blue', linestyle='--', linewidth=1)
            ax.plot(self.src_point['x'], self.src_point['y'], color='blue', marker='o', markersize=10)
            ax.plot(self.dst_point['x'], self.dst_point['y'], color='green', marker='*', markersize=20)
            path = './map_search.png'
            plt.savefig(path)
            # plt.show()
            image_source_cv = cv2.imread(path)
            image_source_cv = cv2.cvtColor(image_source_cv, cv2.COLOR_BGR2RGB)

            image_resize_cv, _ = _resize_image(image_source_cv, self.image_box_size)
            image_resize_qt = _cv2qt_rgb(image_resize_cv)
            self.image_search_pixmap = QPixmap.fromImage(image_resize_qt)
            # self.image_current_pixmap = self.image_resize_pixmap.copy()
            self.label_map_search.setPixmap(self.image_search_pixmap)
            os.remove(path)

        elif index == 5:
            if self.smooth_result['x'] is None :
                self.update_statusBar('optimal path is null!')
                return
            fig, ax = plt.subplots(figsize=(7, 7), dpi=200)
            cmap = plt.get_cmap('YlGnBu')
            ax.imshow(self.map_grid_list, cmap=cmap, interpolation='nearest')
            cmap = plt.get_cmap('summer')
            masked1 = np.ma.masked_where(self.open_area == 0, self.open_area)
            ax.imshow(masked1, cmap=cmap, interpolation='nearest', alpha=0.5)
            path_x, path_y = zip(*self.search_path)
            ax.plot(path_x, path_y, color='blue', linestyle='--', linewidth=1)
            ax.plot(self.smooth_result['x'], self.smooth_result['y'], color='red', linestyle='-', linewidth=2)
            ax.plot(self.src_point['x'], self.src_point['y'], color='blue', marker='o', markersize=10)
            ax.plot(self.dst_point['x'], self.dst_point['y'], color='green', marker='*', markersize=20)
            path = './map_path_smooth.png'
            plt.savefig(path)
            # plt.show()
            image_source_cv = cv2.imread(path)
            image_source_cv = cv2.cvtColor(image_source_cv, cv2.COLOR_BGR2RGB)

            image_resize_cv, _ = _resize_image(image_source_cv, self.image_box_size)
            image_resize_qt = _cv2qt_rgb(image_resize_cv)
            self.image_optimal_pixmap = QPixmap.fromImage(image_resize_qt)
            # self.image_current_pixmap = self.image_resize_pixmap.copy()
            self.label_smooth.setPixmap(self.image_optimal_pixmap)
            os.remove(path)
        else:
            pass

if __name__ == '__main__':
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())