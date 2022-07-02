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
import random


class KPRandomPadCrop(object):
    def __init__(self, ratio=0.25, pad_value=[128, 128, 128]):
        assert (ratio > 0 and ratio <= 1)
        self.ratio = ratio
        self.pad_value = pad_value

    def __call__(self, image, labels=None):
        if random.randint(0,1):
            h, w = image.shape[:2]
            top_offset = int(h * random.uniform(0, self.ratio))
            bottom_offset = int(h * random.uniform(0, self.ratio))
            left_offset = int(w * random.uniform(0, self.ratio))
            right_offset = int(w * random.uniform(0, self.ratio))
            # pad
            if True: #random.randint(0,1):
                image = cv2.copyMakeBorder(image, top_offset, bottom_offset, left_offset, right_offset, cv2.BORDER_CONSTANT, value=self.pad_value)
                if labels is not None and len(labels) > 0:
                    labels[:, 0] = (labels[:, 0] * w + left_offset) / (w + left_offset + right_offset)
                    labels[:, 1] = (labels[:, 1] * h + top_offset) / (h + top_offset + bottom_offset)
            # crop
            else:
                image = image[top_offset:h - bottom_offset, left_offset:w-right_offset]
                if labels is not None and len(labels) > 0:
                    labels[:, 0] = (labels[:, 0] * w - left_offset) / (w - left_offset - right_offset)
                    labels[:, 1] = (labels[:, 1] * h - top_offset) / (h - top_offset - bottom_offset)

        return image, labels

class KPRandomHorizontalFlip(object):
    def __init__(self):
        pass

    def __call__(self, image, labels=None):
        if random.randint(0, 1):
            image = cv2.flip(image, 1)
            h, w = image.shape[:2]
            if labels is not None and len(labels) > 0:
                labels[:, 0] = 1.0 - labels[:, 0]
        return image, labels

class KPResizeImage(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, labels=None, resize=None):
        h, w = image.shape[:2]
        if resize is None:
            scale_h = self.size[1] / h
            scale_w = self.size[0] / w
            image = cv2.resize(image, tuple(self.size))
        else:
            scale_h = resize[1] / h
            scale_w = resize[0] / w
            image = cv2.resize(image, tuple(resize))
        return image, labels

class KPRandomNegMixUp(object):
    def __init__(self, ratio=0.5, neg_dir='/data/zhengxing/my_dl/dataset/sit_up/coco_neg'):
        self.ratio = ratio
        self.neg_dir = neg_dir
        self.neg_images = []
        files = os.listdir(self.neg_dir)
        for file in files:
            if str(file).endswith('.jpg') or str(file).endswith('.png'):
                self.neg_images.append(str(file))

    def __call__(self, image, labels):
        if random.randint(0, 1):
            h, w = image.shape[:2]
            neg_name = random.choice(self.neg_images)
            neg_path = self.neg_dir + '/' + neg_name
            neg_img = cv2.imread(neg_path)
            neg_img = cv2.resize(neg_img, (w, h)).astype(np.float32)
            neg_alpha = random.uniform(0, self.ratio)
            ori_alpha = 1 - neg_alpha
            gamma = 0
            img_add = cv2.addWeighted(image, ori_alpha, neg_img, neg_alpha, gamma)
            return img_add, labels
        else:
            return image, labels

class KPRandomSwapChannels(object):
    def __init__(self):
        self.swaps = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, labels=None):
        if random.randint(0, 1):
            index = random.randint(0, len(self.swaps) - 1)
            image = image[:, :, self.swaps[index]]
        return image, labels

class KPRandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, labels=None):
        if random.randint(0, 1):
            alpha = random.uniform(self.lower, self.upper)
            image = image.astype(np.float32) * alpha
        return image, labels

class KPRandomHSV(object):
    def __init__(self, hue=0.1, saturation=1.5, value=1.5):
        self.hue = hue
        self.saturation = saturation
        self.value = value

    def __call__(self, image, labels=None):
        if random.randint(0, 1):
            dh = random.uniform(-self.hue, self.hue)
            ds = random.uniform(1, self.saturation)
            if random.random() < 0.5:
                ds = 1 / ds
            dv = random.uniform(1, self.value)
            if random.random() < 0.5:
                dv = 1 / dv

            image = image.astype(np.float32) / 255.0
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            def wrap_hue(x):
                x[x >= 360.0] -= 360.0
                x[x < 0.0] += 360.0
                return x

            image[:, :, 0] = wrap_hue(image[:, :, 0] + (360.0 * dh))
            image[:, :, 1] = np.clip(ds * image[:, :, 1], 0.0, 1.0)
            image[:, :, 2] = np.clip(dv * image[:, :, 2], 0.0, 1.0)

            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
            image = (image * 255.0)
        return image, labels

class PersionKeypointTxt(Dataset):
    #read the label files
    def __init__(self, phase, arg_dict):
        self.phase = phase
        self.img_root = arg_dict['data_dir']
        self.label_file = arg_dict['data_label']
        self.resize = arg_dict['resize']
        self.resize_w = self.resize[0]
        self.resize_h = self.resize[1]
        self.mean = arg_dict['mean']
        self.std = arg_dict['std']
        self.kp_num = arg_dict['kp_num']
        self.gauss_ratio = arg_dict['gauss_ratio']
        self.gauss_sigma = arg_dict['gauss_sigma']
        self.heatmap = arg_dict['heatmap']
        self.heatmap_w = self.heatmap[0]
        self.heatmap_h = self.heatmap[1]
        if 'data_len_expand' in arg_dict.keys():
            self.data_len_expand = arg_dict['data_len_expand']
        else:
            self.data_len_expand = 1

        if self.phase == 'train':
            self.transforms = [
                KPRandomHorizontalFlip(),
                KPRandomPadCrop(ratio=0.25, pad_value = self.mean),
                KPResizeImage(self.resize),
                #KPRandomNegMixUp(ratio=0.5, neg_dir='../../dataset/sit_up/images_coco_neg/'),
                KPRandomSwapChannels(),
                KPRandomContrast(),
                KPRandomHSV(),
                NormalizeImage(self.mean,self.std),]
        elif self.phase == 'eval' or self.phase == 'infer':
            self.transforms = [
                KPResizeImage(self.resize),
                NormalizeImage(self.mean,self.std),]
        else:
            logging.error('unsupport phase: %s'%phase)
            exit(0)
        if self.phase != 'infer':
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
            img, _ =  t(img, label)
        img = img.transpose(2,0,1)
        
        heatmap = np.zeros((self.kp_num, self.heatmap_h, self.heatmap_w)).astype(np.float32)

        #for c in range(self.kp_num):
        for k in range(label.shape[0]):
            labels_c = label[k]
            labels_kp_name = int(labels_c[2])
            heatmap_c = heatmap[labels_kp_name]
            labels_x = math.floor(labels_c[0] * (self.heatmap_w - 1))
            labels_y = math.floor(labels_c[1] * (self.heatmap_h - 1))
            for i in range(labels_y-self.gauss_ratio, labels_y+self.gauss_ratio):
                for j in range(labels_x-self.gauss_ratio, labels_x+self.gauss_ratio):
                    if i < 0 or i > 55 or j < 0 or j > 55:
                        continue
                    heatmap_c[i][j] = max(heatmap_c[i][j], np.exp(-0.5 * ((i - labels_y)**2 + (j - labels_x)**2) / self.gauss_sigma**2))
                    #heatmap_c[i][j] = np.exp(-0.5 * ((i - labels_y)**2 + (j - labels_x)**2) / self.gauss_sigma**2)

        ##################################
        #import matplotlib
        #matplotlib.use('Agg')
        #import matplotlib.pyplot as plt
        #img_resize = cv2.resize(img_ori, (224, 224))[:,:,[2,1,0]]

        #img_show = (img.transpose(1,2,0) * self.std + self.mean).astype(np.uint8)[:,:,[2,1,0]]

        #for i in range(2):
        #    plt.clf()
        #    plt.imshow(img_show)
        #    heatmap_resize = cv2.resize(heatmap[i], (224, 224))
        #    plt.imshow(heatmap_resize, alpha=0.5)
        #    plt.savefig('./dataread_kp_%d.jpg'%i)
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
        self.img_label = []
        self.img_name = []
        with open(self.label_file, 'r') as f:
            for line in f.readlines():
                domains = line.strip().split(' ')
                self.img_name.append(domains[0])
                kp_str = domains[1:]
                assert(len(kp_str) % 3 == 0)
                labels = []
                for seq in range(int(len(kp_str) / 3)):
                    kp_list = kp_str[seq*3:(seq+1)*3]
                    label = np.array([float(i) for i in kp_list ])
                    labels.append(label)
                self.img_label.append(np.stack(labels, axis=0))

        self.img_name = self.img_name * self.data_len_expand
        self.img_label = self.img_label * self.data_len_expand


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

