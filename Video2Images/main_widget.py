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
import threading

# ui配置文件
cUi, cBase = uic.loadUiType("main_widget.ui")

# 主界面
class CImageWidget(QWidget, cUi):

    info_sig = pyqtSignal(dict)

    def __init__(self):
        # 设置UI
        QMainWindow.__init__(self)
        cUi.__init__(self)
        self.setupUi(self)
        self.videos_dir = ''
        self.images_dir = ''
        self.thread_flag = False
        self.thread_handle = None
        self.info_sig.connect(self.info_slot)
        
        self.progressNow.setMaximum(100)
        self.progressTotal.setMaximum(100)
        self.progressNow.setValue(0)
        self.progressTotal.setValue(0)
        
        self.setWindowTitle('视频转图片小工具  作者：理工堆堆星 联系：cjnewstar111')
        
    def closeEvent(self, event):
        pass
    
    def thread_func(self, videos_dir, images_dir):
        print('>>[info] thread_func start')
        print('>>[info] videos_dir is', videos_dir)
        print('>>[info] images_dir is', images_dir)
        
        all_videos = os.listdir(videos_dir)
        videos_count = len(all_videos)
        for i, video_name in enumerate(all_videos):
            total_progress = int(i * 100 / videos_count)
            video_path = os.path.join(videos_dir, video_name)
            capture = cv2.VideoCapture(video_path)
            frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            if frame_count < 1:
                frame_count = 100000000
            frame_seq = 0
            while True:
                #print('-----> frame_count,', frame_count)
                ret, frame = capture.read()
                if not ret:
                    self.info_sig.emit({'video_name':video_name, 'total_progress': total_progress, 'now_progress': 100})
                    break
                frame_name = video_name + '_%d.jpg'%frame_seq
                frame_path = os.path.join(images_dir, frame_name)
                cv2.imwrite(frame_path, frame)
                now_progress = int(frame_seq * 100 / frame_count)
                self.info_sig.emit({'video_name':video_name, 'total_progress': total_progress, 'now_progress': now_progress})
                frame_seq += 1
                
        self.info_sig.emit({'video_name':video_name, 'total_progress': 100, 'now_progress': 100})
        self.thread_flag = False
        self.thread_handle = None
        print('>>[info] thread_func stop')
            
    @pyqtSlot() 
    def on_btnOpenVideosDir_clicked(self):
        print('info:on_btnOpenVideosDir_clicked')
        videos_dir = QFileDialog.getExistingDirectory(self, u"请选择视频所在文件夹", os.getcwd())
        if os.path.exists(videos_dir):
            self.editVideosDir.setText(videos_dir)
        else:
            pass

    @pyqtSlot() 
    def on_btnOpenImagesDir_clicked(self):
        print('info:on_btnOpenImagesDir_clicked')
        images_dir = QFileDialog.getExistingDirectory(self, u"请选择图片保存文件夹", os.getcwd())
        if os.path.exists(images_dir):
            self.editImagesDir.setText(images_dir)
        else:
            pass
            
    @pyqtSlot() 
    def on_btnStartConvert_clicked(self):
        print('info:on_btnStartConvert_clicked')
        if self.thread_flag is False:
            
            videos_dir = self.editVideosDir.text()
            images_dir = self.editImagesDir.text()
            
            if not os.path.exists(videos_dir):
                reply = QMessageBox.warning(self,
                        u'警告', 
                        u'请选择有效的视频所在文件夹', 
                        QMessageBox.Yes)
                return
            
            if not os.path.exists(images_dir):
                reply = QMessageBox.warning(self,
                        u'警告', 
                        u'请选择有效的图片保存文件夹', 
                        QMessageBox.Yes)
                return
            
            self.thread_flag = True
            self.thread_handle = threading.Thread(target=self.thread_func, args=(videos_dir, images_dir,))
            self.thread_handle.start()
        else:
            pass

    def info_slot(self, info_dict):
        video_name = info_dict['video_name']
        total_progress = info_dict['total_progress']
        now_progress = info_dict['now_progress']
        self.editNowVideo.setText(video_name)
        self.progressNow.setValue(now_progress)
        self.progressTotal.setValue(total_progress)


if __name__ == "__main__":
    cApp = QApplication(sys.argv)
    cImageWidget = CImageWidget()
    cImageWidget.show()
    sys.exit(cApp.exec_())