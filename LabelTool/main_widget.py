# -*- coding: utf-8 -*-
import sys
import os
if hasattr(sys, 'frozen'):
    os.environ['PATH'] = sys._MEIPASS + ";" + os.environ['PATH']
from PyQt5.QtWidgets import * #QApplication, QWidget, QPushButton, QMainWindow, QVBoxLayout, QHBoxLayout
from PyQt5 import QtCore, QtGui, uic
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from image_widget import *
import shutil
import glob

# ui配置文件
cUi, cBase = uic.loadUiType("main_widget.ui")

# 主界面
class CMainWidget(QWidget, cUi):
    def __init__(self):
        # 设置UI
        QMainWindow.__init__(self)
        cUi.__init__(self)
        self.setupUi(self)
        self.image_dir = ''
        self.label_file = ''
        self.label_info = {}
        self.image_widgets = []
        self.batch_index = 0

        vbox = QVBoxLayout()
        for i in range(4):
            hbox = QHBoxLayout()
            for j in range(4):
                self.image_widgets.append(CImageWidget())
                hbox.addWidget(self.image_widgets[-1])
            vbox.addLayout(hbox)
        self.frame.setLayout(vbox)
        self.btn_open.clicked.connect(self.slot_btn_open)
        self.btn_back.clicked.connect(self.slot_btn_pre)
        self.btn_next.clicked.connect(self.slot_btn_next)
        self.edit_cls.textChanged.connect(self.slot_edit_change)

        self.btn_back.hide()
        self.btn_next.hide()

    def closeEvent(self, event):
        self.save_kp_info()
        self.write_label_file()
        pass

    def read_label_file(self):
        if os.path.exists(self.label_file):
            with open(self.label_file, 'r') as f:
                for line in f:
                    info = line.strip('\r\n')
                    if len(info) == 0:
                        continue
                    domains = info.split(' ')
                    name = domains[0]

                    kps = []
                    if len(domains) > 1:
                        kps_str = domains[1:]
                        assert (len(kps_str) % 3 == 0)
                        kp_count = int(len(kps_str) / 3)
                        for i in range(kp_count):
                            kp_str = kps_str[i * 3:(i + 1) * 3]
                            kp = [float(x) for x in kp_str]
                            kps.append(kp)
                    self.label_info[name] = kps

    def write_label_file(self):
        with open(self.label_file, 'w') as f:
            for key in self.label_info.keys():
                info = str(key)
                if self.label_info[key] is not None:
                    for kp in self.label_info[key]:
                        info += ' %.2f %.2f %.2f'%(kp[0],kp[1],kp[2])
                f.write(info + '\r')

    def save_kp_info(self):
        for image_win in self.image_widgets:
            name, kps = image_win.get_info()
            if name is not None and len(name) > 0:
                self.label_info[name] = kps

    def slot_btn_open(self):
        self.image_dir = QFileDialog.getExistingDirectory(self, "选择文件夹", "C:\\Users\\newst\\Desktop\\test_data")
        if os.path.exists(self.image_dir):
            self.btn_back.show()
            self.btn_next.show()
            files = os.listdir(self.image_dir)
            for img_name in files:
                if str(img_name).endswith('txt'):
                    continue
                self.label_info[str(img_name)] = None

            self.label_file = self.image_dir + "/label.txt"
            self.read_label_file()
            self.slot_btn_next()

    def slot_btn_next(self):
        self.save_kp_info()
        image_names = self.update_batch_index(next=True, pre=False)
        if image_names is not None:
            for i in range(16):
                if i + 1 <= len(image_names):
                    img_path = self.image_dir + '/' + image_names[i]
                    self.image_widgets[i].set_info(img_path, self.label_info[image_names[i]])
                else:
                    self.image_widgets[i].set_info(None, None)

    def slot_btn_pre(self):
        self.save_kp_info()
        image_names = self.update_batch_index(next=False, pre=True)
        if image_names is not None:
            for i in range(16):
                if i + 1 <= len(image_names):
                    img_path = self.image_dir + '/' + image_names[i]
                    self.image_widgets[i].set_info(img_path, self.label_info[image_names[i]])
                else:
                    self.image_widgets[i].set_info(None, None)

    def slot_edit_change(self):
        for image_win in self.image_widgets:
            image_win.set_current_cls(int(self.edit_cls.text()))

    def update_batch_index(self, next=True, pre=False):
        if len(self.label_info.keys()) == 0:
            return None
        assert (next != pre)
        self.total_batch = math.ceil(len(self.label_info.keys()) / 16)
        if next:
            if self.batch_index == self.total_batch:
                return None
            if self.batch_index == self.total_batch - 1:
                batch_count = len(self.label_info.keys()) % 16
                image_names = list(self.label_info.keys())[self.batch_index * 16: self.batch_index * 16 + batch_count]
                self.batch_index += 1
            if self.batch_index < self.total_batch - 1:
                batch_count = 16
                image_names = list(self.label_info.keys())[self.batch_index * 16: self.batch_index * 16 + batch_count]
                self.batch_index += 1
        else:
            if self.batch_index == 1:
                return None
            else:
                self.batch_index -= 1
                image_names = list(self.label_info.keys())[(self.batch_index-1) * 16: (self.batch_index-1) * 16 + 16]

        self.label_jindu.setText('%d/%d'%(self.batch_index, self.total_batch))
        return image_names



if __name__ == "__main__":
    cApp = QApplication(sys.argv)
    cMainWidget = CMainWidget()
    cMainWidget.show()
    sys.exit(cApp.exec_())