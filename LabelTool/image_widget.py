# -*- coding: utf-8 -*-
import sys
import os
if hasattr(sys, 'frozen'):
    os.environ['PATH'] = sys._MEIPASS + ";" + os.environ['PATH']
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui, uic
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import copy
import xml.etree.cElementTree as et
import os
import cv2
import math
from PIL import Image

# ui配置文件
cUi, cBase = uic.loadUiType("image_widget.ui")

# 主界面
class CImageWidget(QWidget, cUi):
    def __init__(self):
        # 设置UI
        QMainWindow.__init__(self)
        cUi.__init__(self)
        self.setupUi(self)

        #image信息
        self.img_path = ''
        self.img_name = ''
        self.img = None

        # 已标注信息
        #self.box_list = []
        self.kp_list = []

        #待标注信息
        self.current_class = 0
        #self.start_label = False
        #self.current_box = [0,0,0,0,0]
        self.current_kp = [0, 0, 0]

    def closeEvent(self, event):
        pass

    def set_info(self, image_path, kp_list=None):
        if image_path is None:
            self.img_path = ''
            self.img_name = ''
            self.img = None
            self.kp_list = []
            #self.start_label = False
            self.current_kp = [0, 0, 0]
        else:
            self.img_path = image_path
            self.img_name = os.path.basename(image_path) #.split('.')[0]
            self.img = QPixmap(self.img_path)
            if kp_list is not None:
                self.kp_list = kp_list
            else:
                self.kp_list = []
        self.update()

    def set_current_cls(self, cls):
        #self.current_kp[2] = cls
        self.current_class = cls

    def get_info(self):
        return self.img_name, self.kp_list

    def draw_background(self, painter):
        pen = QPen()
        pen.setColor(QColor(0, 0, 0))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawRect(0, 0, self.width(), self.height())

    def draw_image(self, painter):
        if self.img is not None:
            painter.drawPixmap(QtCore.QRect(0, 0, self.width(), self.height()), self.img)
            painter.drawText(10,20,str(self.img_name))

    def draw_kp_info(self, painter):
        for kp in self.kp_list:
            painter.setPen(QColor(255, 0, 0))
            pen = QPen()
            pen.setColor(QColor(255, 0, 0))
            pen.setWidth(3)
            painter.setPen(pen)
            painter.drawPoint(kp[0] * self.width(), kp[1] * self.height())
            painter.drawText(kp[0] * self.width(), kp[1] * self.height(), str(kp[2]))
        '''
        if self.start_label:
            kp = self.current_kp
            pen = QPen()
            pen.setColor(QColor(255, 0, 0))
            pen.setWidth(3)
            painter.setPen(pen)
            painter.drawPoint(kp[0] * self.width(), kp[1] * self.height())
            painter.drawText(kp[0] * self.width(), kp[1] * self.height(), str(kp[2]))
        '''

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        self.draw_background(painter)
        self.draw_image(painter)
        self.draw_kp_info(painter)

    def mousePressEvent(self, e):
        if self.img is None:
            return
        if e.button() == QtCore.Qt.LeftButton:
            #self.start_label = True
            self.current_kp[0] = e.pos().x() / self.width()
            self.current_kp[1] = e.pos().y() / self.height()
            self.kp_list.append([self.current_kp[0], self.current_kp[1], self.current_class])
        if e.button() == QtCore.Qt.RightButton and len(self.kp_list) > 0:
            self.kp_list.pop()
        self.update()

if __name__ == "__main__":
    cApp = QApplication(sys.argv)
    cImageWidget = CImageWidget()
    cImageWidget.show()
    cImageWidget.set_info('./1.jpg')
    sys.exit(cApp.exec_())