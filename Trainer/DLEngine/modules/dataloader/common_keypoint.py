import os
from torch.utils.data import Dataset
import cv2
import numpy as np
import random
from .transform import *
#from transform import *
import copy
import logging
import torch
import csv
import copy


class CommonKeyPoint(Dataset):
    #read the label files
    def __init__(self, phase, arg_dict):
        self.phase = phase
        self.img_root = arg_dict['data_dir']
        self.label_file = arg_dict['data_label']
        self.resize = arg_dict['resize']
        self.mean = arg_dict['mean']
        self.std = arg_dict['std']

        if self.phase == 'train':
            self.transforms = [
                ResizeImage(self.resize),
                NormalizeImage(self.mean,self.std),]
        elif self.phase == 'eval':
            self.transforms = [
                ResizeImage(self.resize),
                NormalizeImage(self.mean,self.std),]
        else:
            logging.error('unsupport phase: %s'%phase)
            exit(0)
    
        self.read_label_file()

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, item):
        img_name = self.img_name[item]
        label_ori = self.img_label[item]
        label = copy.deepcopy(label_ori)
        img_ori = cv2.imread(self.img_root + '/' + img_name, cv2.IMREAD_COLOR)
        if img_ori is None:
            raise ValueError('cv2 read failed: ', img_name)
        ori_h, ori_w = img_ori.shape[0:2]
        img = img_ori.astype(np.float32)
        for t in self.transforms:
            img, _ =  t(img, None)
        img = img.transpose(2,0,1)

        label[:, :, 0] =  label[:, :, 0] / ori_w
        label[:, :, 1] =  label[:, :, 1] / ori_h
        label = np.clip(label, label.min(), 1.0)

        heatmap = np.zeros((13, 56, 56)).astype(np.float32)
        radio = 5
        sigma = 1
        for c in range(13):
            heatmap_c = heatmap[c]
            for l in range(label.shape[0]):
                labels_c = label[l][c]
                labels_e = labels_c[2]
                if labels_e > 0:
                    labels_x = math.floor(labels_c[0] * (55))
                    labels_y = math.floor(labels_c[1] * (55))
                    for i in range(labels_y-radio, labels_y+radio):
                        for j in range(labels_x-radio, labels_x+radio):
                            if i < 0 or i > 55 or j < 0 or j > 55:
                                continue
                            heatmap_c[i][j] = np.exp(-0.5 * ((i - labels_y)**2 + (j - labels_x)**2) / sigma**2 ) 


        ##################################
        #import matplotlib
        #matplotlib.use('Agg')
        #import matplotlib.pyplot as plt
        #img_resize = cv2.resize(img_ori, (224, 224))[:,:,[2,1,0]]
        #for i in range(13):
        #    plt.clf()
        #    plt.imshow(img_resize)
        #    heatmap_resize = cv2.resize(heatmap[i], (224, 224))
        #    plt.imshow(heatmap_resize, alpha=0.5)
        #    plt.savefig('./train_kp_%d.jpg'%i)
        ##################################

        return img, heatmap

    def read_image(self, img_path):
        img_ori = cv2.imread(img_path)
        if img_ori is None:
            raise ValueError('cv2 read failed: ', img_name)
        ori_w, ori_h = img_ori.shape[0:2]
        img = img_ori.astype(np.float32)
        for t in self.transforms:
            img, _ =  t(img, None)
        img = img.transpose(2,0,1)
        return img, img_ori, ori_w, ori_h 

    def read_label_file(self):

        def get_one_line(line_info):
            img_name, personNumber, bndbox, nose, left_eye, right_eye, left_ear, right_ear, \
                left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist, \
                left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle = line_info

            nose = np.array([float(i) for i in nose.split('_')])
            left_shoulder = np.array([float(i) for i in left_shoulder.split('_')])
            right_shoulder = np.array([float(i) for i in right_shoulder.split('_')])
            left_hip = np.array([float(i) for i in left_hip.split('_')])
            right_hip = np.array([float(i) for i in right_hip.split('_')])
            left_elbow = np.array([float(i) for i in left_elbow.split('_')])
            right_elbow = np.array([float(i) for i in right_elbow.split('_')])
            left_wrist = np.array([float(i) for i in left_wrist.split('_')])
            right_wrist = np.array([float(i) for i in right_wrist.split('_')])
            left_knee = np.array([float(i) for i in left_knee.split('_')])
            right_knee = np.array([float(i) for i in right_knee.split('_')])
            left_ankle = np.array([float(i) for i in left_ankle.split('_')])
            right_ankle = np.array([float(i) for i in right_ankle.split('_')])
            
            pose_list = np.stack([nose, left_shoulder, right_shoulder, left_hip, right_hip, left_elbow, right_elbow, \
                        left_wrist, right_wrist, left_knee, right_knee, left_ankle, right_ankle], axis=0)
            
            return img_name, int(personNumber), pose_list

        self.img_label = []
        self.img_name = []
        with open(self.label_file, 'r') as f:
            reader = csv.reader(f)
            total_lines = []
            for line in reader:
                total_lines.append(line)

            i = 1 # the 0 line is no use
            while i < len(total_lines):
                labels = []
                img_name, personNumber, pose_list = get_one_line(total_lines[i])
                labels.append(pose_list)

                self.img_name.append(img_name.split('\'')[1])

                if personNumber > 1:
                    for num in range(1, personNumber):
                        img_name, personNumber, pose_list = get_one_line(total_lines[i+num])
                        labels.append(pose_list)
                i += personNumber
                self.img_label.append(np.stack(labels, axis=0))

    #@staticmethod
    def collate_fn(self, batch):
        imgs, labels = list(zip(*batch))
        imgs = torch.from_numpy(np.stack(imgs, 0))
        labels = torch.from_numpy(np.stack(labels, 0))
        return [imgs, labels]

if __name__ == '__main__':

    demo_dict = {
        'data_name': 'CommonKeyPoint',
        'num_workers': 6,
        'data_dir': '/data/data_set/COCO/train2017/',
        'data_label': '/data/zhengxing/my_dl/tools/get_coco_keypoints/coco_one_person_keypoints.csv',
        'batch_size': 16,
        'resize': [224, 224], # w and h
        'mean': [104,117,123],
        'std': [1,1,1]}

    dataset = CommonKeyPoint('train', demo_dict)
    dataset.test_getitem(0)

