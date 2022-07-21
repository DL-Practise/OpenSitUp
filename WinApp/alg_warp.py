# -*- coding: utf-8 -*-
import sys
import sys
sys.path.append('./movenet')
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
from alg_onnx.api import SitupDet
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    import queue
except ImportError:
    import Queue as queue
import json



class AlgWarp():
    def __init__(self, image_queue, callback_func): 
        self.img_queue = image_queue
        self.callback_func = callback_func
        self.thread_handle = None
        self.thread_flag = False
        self.alg = SitupDet()     
        self.analyze_flag = False   
        self.situp_count = 0
        self.last_pos = 'unknown'

    def thread_func(self, args):
        while self.thread_flag:
            try:
                img = self.img_queue.get(block=True, timeout=0.5)
                if type(img) is dict:
                    #print('thread running: get a cmd')
                    if img['cmd'] == 'analyze_start':
                        self.analyze_init()
                    if img['cmd'] == 'analyze_stop':
                        self.analyze_fini()
                else:
                    time_start = time.time()
                    result_img, result_info = self.infer(img)
                    time_spend = time.time() - time_start
                    self.callback_func(result_img, result_info, time_spend)
            except queue.Empty:
                pass
        
        self.thread_flag = False
        self.thread_handle = None
                      
    def start(self):
        self.thread_flag = True
        self.thread_handle = threading.Thread(target=self.thread_func, args=(None,)) 
        self.thread_handle.start()
    
    def stop(self):
        self.thread_flag = False
        if self.thread_handle is not None:
            self.thread_handle.join()
            self.thread_handle = None

    def analyze_init(self):
        self.analyze_flag = True   
        self.situp_count = 0
        self.last_pos = 'unknown'

    def analyze_fini(self):
        self.analyze_flag = False   

    def infer(self, img):
        img, pos = self.alg.infer(img)
        if self.analyze_flag:
            now_pos_head, now_pos_knee, now_pos_crotch = pos
            if now_pos_head[0] > now_pos_knee[0] and now_pos_crotch[0] > now_pos_knee[0]:
                self.last_pos = 'step1'
            if now_pos_head[0] < now_pos_knee[0] and now_pos_crotch[0] > now_pos_knee[0]:
                if self.last_pos == 'step1':
                    self.situp_count += 1
                self.last_pos = 'step2'

            info = '仰卧起坐个数：%d'%(self.situp_count)
        else:
            info = '图片模式，只检测关键点，不计数'

        return img, info
            