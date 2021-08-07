import os
from torch.utils.data import Dataset
import cv2
import numpy as np
import random
from .transform import *
import copy
import logging
import torch
'''
demo_dict = {
        'data_name': 'CommonCls',
        'num_workers': 6,
        'data_dir': '/data/data_set/cifar10/train/',
        'data_label': '/data/data_set/cifar10/train.txt',
        'batch_size': 256,
        'resize': [224, 224], # w and h
        'crop': [224, 224], # w and h
        'mean': [104,117,123],
        'std': [1,1,1],},
'''
class CommonCls(Dataset):
    #read the label files
    def __init__(self, phase, arg_dict):
        self.phase = phase
        self.img_root = arg_dict['data_dir']
        self.label_file = arg_dict['data_label']
        self.resize = arg_dict['resize']
        self.crop = arg_dict['crop']
        self.mean = arg_dict['mean']
        self.std = arg_dict['std']

        if self.phase == 'train':
            self.transforms = [
                ResizeImage(self.resize),
                RandomCropFix(self.crop),
                RandomHorizontalFlip(),
                NormalizeImage(self.mean,self.std),]
        elif self.phase == 'eval':
            self.transforms = [
                ResizeImage(self.resize),
                CenterCropFix(self.crop),
                NormalizeImage(self.mean,self.std),]
        else:
            logging.error('unsupport phase: %s'%phase)
            exit(0)

        self.read_label_file()

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, item):
        img_name = self.img_name[item]
        label = self.img_label[item]

        img = cv2.imread(img_name, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError('cv2 read failed: ', img_name)
        img_size = img.shape[0:2]
        img = img.astype(np.float32)
        for t in self.transforms:
            img, _ =  t(img, None)

        img = img.transpose(2,0,1)
        return img, label

    def read_image(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError('cv2 read failed: ', image_path)
        img_size = img.shape[0:2]
        img = img.astype(np.float32)
        for t in self.transforms:
            img, _ = t(img, None)

        img = img.transpose(2, 0, 1)
        return img

    def read_label_file(self):
        with open(self.label_file) as input_file:
            lines = input_file.readlines()

            self.img_name = [os.path.join(self.img_root, line.strip().split(' ')[0]) for line in lines]
            self.img_label = [int(line.strip().split(' ')[-1]) for line in lines]

    #@staticmethod
    def collate_fn(self, batch):
        imgs, labels = list(zip(*batch))
        imgs = torch.from_numpy(np.stack(imgs, 0))
        labels = torch.from_numpy(np.stack(labels, 0))
        return [imgs, labels]