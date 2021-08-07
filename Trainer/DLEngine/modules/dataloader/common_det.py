import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import sys
import cv2
from .transform import *
import logging

class CommonDet(Dataset):
    # read the label files
    def __init__(self, phase, arg_dict):
        self.phase = phase
        self.img_root = arg_dict['data_dir']
        self.label_file = arg_dict['data_label']
        self.resize = arg_dict['resize']
        self.mean = arg_dict['mean']
        self.std = arg_dict['std']
        if 'epoch_expand' not in arg_dict.keys():
            self.epoch_expand = 1
        else:
            self.epoch_expand = arg_dict['epoch_expand']

        if self.phase == 'train':
            self.transforms = [
                ResizeImage(self.resize),
                RandomHorizontalFlip(),
                RandomSwapChannels(),
                RandomContrast(),
                RandomHSV(),
                NormalizeImage(self.mean,self.std),]
        elif self.phase == 'test':
            self.transforms = [
                ResizeImage(self.resize),
                NormalizeImage(self.mean,self.std),]
        else:
            logging.error('unsupport phase: %s' % phase)
            exit(0)

        self.read_label_file()

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, item):
        try:
            img_name = self.img_names[item]
            labels = np.array(self.img_labels[item]).astype(np.float32)
            if not img_name.endswith('.jpg') and not img_name.endswith('.JPG') :
                img_name += '.jpg'
            img = cv2.imread(img_name, cv2.IMREAD_COLOR)
            img_size = img.shape[0:2]
        except:
            raise ValueError('cv2 read failed: ', img_name)

        img = img.astype(np.float32)
        for t in self.transforms:
            img, labels = t(img, labels)

        img = img.transpose(2, 0, 1)
        new_labels = {'file':img_name, 'size':img_size, 'labels':[], 'boxes':[], 'format':'xyxy'}
        
        for label in labels:
            box = np.array(label[0:4]).astype(np.float32)
            label = int(label[4])
            new_labels['boxes'].append(box)
            new_labels['labels'].append(label)
        return img, new_labels

    # read the label files
    def read_label_file(self):
        self.img_names = []
        self.img_labels = []

        with open(self.label_file) as input_file:
            lines = input_file.readlines()
            for line in lines:
                # read box(ignore box_count domain)
                fields = line.strip().split(' ')[1:]
                if int(len(fields) / 5) == 0:
                    fields = []

                if len(fields) % 5 == 1:
                    fields = fields[1:]
                try:
                    fields = [int(float(i)) for i in fields]
                except:
                    print(fields)

                sample_labels = []
                for i in range(0, len(fields), 5):
                    sample_labels.append(fields[i:i + 5])

                self.img_names.append(self.img_root + line.strip().split(' ')[0])
                self.img_labels.append(sample_labels)

        #expand the list
        if self.epoch_expand > 1:
            logging.info('expand the dateset to %d large'%self.epoch_expand)
            for i in range(self.epoch_expand - 1):
                self.img_names.extend(self.img_names)
                self.img_labels.extend(self.img_labels)

    def read_infer_image(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        labels = None
        for t in self.transforms_infer:
            img, labels = t(img, labels)

        img = img.transpose(2, 0, 1)
        img_t = torch.from_numpy(img).unsqueeze(0)
        return img_t

    #@staticmethod
    def collate_fn(self, batch):
        imgs, labels = list(zip(*batch))
        imgs = torch.from_numpy(np.stack(imgs, 0))
        return [imgs, labels]

    