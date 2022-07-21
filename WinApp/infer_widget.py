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
import time
from PIL import Image
import threading
import time
try:
    import queue
except ImportError:
    import Queue as queue
from alg_warp import AlgWarp


# ui配置文件
cUi, cBase = uic.loadUiType("infer_widget.ui")

# 主界面
class InferWidget(QWidget, cUi):
    def __init__(self): #, main_widget=None):
        # 设置UI
        QMainWindow.__init__(self)
        cUi.__init__(self)
        self.setupUi(self)
        self.comboBoxCamera.addItem('0')
        self.comboBoxCamera.addItem('1')
        self.comboBoxCamera.addItem('2')
        
        self.allow_callback = False

        self.video_cap = None
        self.camera_cap = None

        self.thread_handle = None
        self.thread_flag = False
        self.image_queue = queue.Queue(maxsize=1)
        
        self.qpixmap = None
        self.infor = None

        self.setWindowTitle('仰卧起坐计数 github: OpenSitup')

        # alg wrap
        self.alg_warp = AlgWarp(self.image_queue, self.slot_alg_result)
        self.alg_warp.start()


    def closeEvent(self, event):
        reply = QMessageBox.question(self,
                                    '本程序',
                                    "是否要退出程序？",
                                    QMessageBox.Yes | QMessageBox.No,
                                    QMessageBox.No)
                                    
        if reply == QMessageBox.Yes:
            self.alg_warp.stop()
            event.accept()
        else:
            event.ignore()

    @pyqtSlot()
    def on_btnPhoto_clicked(self):
        print('on_btnPhoto_clicked')
        self.allow_callback = True
        img_path = QFileDialog.getOpenFileName(self,  "选取图片", "./", "Images (*.jpg);;Images (*.png)") 
        img_path = img_path[0]
        if img_path != '':
            img = cv2.imread(img_path)        
            self.put_image(img)
    
    @pyqtSlot()
    def on_btnVideo_clicked(self):
        print('on_btnVideo_clicked')
        self.allow_callback = True
        if self.thread_handle != None:
            print('thread is running now, ignore')
            return
        video_path = QFileDialog.getOpenFileName(self,  "选取视频", "./", "Videos (*.*);;Videos (*.mp4);;Videos (*.3gp)") 
        video_path = video_path[0]
        self.put_image({'cmd':'analyze_start'})
        if video_path != '':
            self.thread_flag = True
            self.thread_handle = threading.Thread(target=self.thread_func, args=({'video':video_path},)) 
            self.thread_handle.start()
                    
    @pyqtSlot()    
    def on_btnCamera_clicked(self):
        print('on_btnCamera_clicked')
        self.allow_callback = True
        if self.thread_handle != None:
            print('thread is running now, ignore')
            return
        self.put_image({'cmd':'analyze_start'})
        self.thread_flag = True
        self.thread_handle = threading.Thread(target=self.thread_func, args=({'camera':0},)) 
        self.thread_handle.start()
                    
    @pyqtSlot()    
    def on_btnStop_clicked(self):
        self.put_image({'cmd':'analyze_stop'})
        self.stop()
        
    def thread_func(self, args):
        if 'camera' in args.keys() and self.camera_cap is None:
            camera_id = args['camera']
            self.camera_cap = cv2.VideoCapture(camera_id)
            while self.thread_flag:
                ret, img = self.camera_cap.read()
                if ret is False:                    
                    break
                self.put_image(img)          
            self.thread_flag = False
            self.camera_cap.release()
            self.camera_cap = None
            self.thread_handle = None
        elif 'video' in args.keys() and self.video_cap is None:
            video_path = args['video']
            self.video_cap = cv2.VideoCapture(video_path)
            fps = self.video_cap.get(cv2.CAP_PROP_FPS)
            if fps < 10 or fps > 120:
                fps = 30.0
            time_interval = 1.0 / fps
            while self.thread_flag:
                ret, img = self.video_cap.read()
                time.sleep(time_interval)
                if ret is False:
                    break
                self.put_image(img)

            self.thread_flag = False
            self.video_cap.release()
            self.video_cap = None
            self.thread_handle = None
        elif 'dir' in args.keys():
            for file_path in self.file_in_dir:
                img = cv2.imread(file_path)
                while not self.put_image(img) and self.thread_flag:
                    time.sleep(0.1)
            self.thread_flag = False
            self.thread_handle = None
        else:
            pass
                      
    def slot_alg_result(self, ret_img, ret_info, time_spend):
        if not self.allow_callback:
            return
        img = ret_img
        height, width, bytesPerComponent = img.shape
        bytesPerLine = bytesPerComponent * width
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        image = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.qpixmap = QPixmap.fromImage(image) 
        self.infor = ret_info
        self.update()
    
    def stop(self):
        #self.image_queue.queue.clear()
        #time.sleep(0.1)
        self.allow_callback = False
        time.sleep(0.1)
        self.qpixmap = None
        self.thread_flag = False
        if self.thread_handle is not None:
            self.thread_handle.join()
            self.thread_handle = None
        self.update()

    def put_image(self, img):
        if self.image_queue.full():
            return False
        else:
            self.image_queue.put(img)
            return True
        
    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        self.draw_image(painter)
        self.draw_infor(painter)
        
    def draw_image(self, painter):
        pen = QPen()
        font = QFont("Microsoft YaHei")
        if self.qpixmap is None:
            painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))
            painter.setBrush(QBrush(Qt.SolidPattern))
            painter.setBrush(QColor(100, 100, 100, 0))
            painter.drawRect(QtCore.QRect(0, 0, self.width(), self.height()))
        else:
            painter.drawPixmap(QtCore.QRect(0, 0, self.width(), self.height()), self.qpixmap)
             
    def draw_infor(self, painter):
        if self.infor is None:
            return
        pen = QPen()
        pen.setColor(QColor(255, 0, 0))
        painter.setPen(pen)
        font = QFont("Microsoft YaHei")
        pointsize = font.pointSize()
        font.setPixelSize(pointsize*self.width() / 400.0)
        painter.setFont(font)
        painter.drawText(self.width()*0.01, self.height()*0.1, '%s'%(self.infor))

if __name__ == "__main__":
    cApp = QApplication(sys.argv)
    cInferWidget = InferWidget()
    cInferWidget.show()
    sys.exit(cApp.exec_())