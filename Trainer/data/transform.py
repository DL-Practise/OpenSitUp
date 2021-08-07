import cv2
import numpy as np
import random
import copy
import math
from scipy.stats import mode
from math import fabs, sin, cos, radians

#common 光学变换
class RandomSwapChannels(object):
    def __init__(self):
        self.swaps = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, labels=None):
        if random.randint(0, 1):
            index = random.randint(0, len(self.swaps) - 1)
            image = image[:, :, self.swaps[index]]
        return image, labels

class RandomContrast(object):
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

class RandomHSV(object):
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

'''
class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, labels=None):
        if random.randint(0, 1):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, labels
'''

class RandomBrightness(object):
    def __init__(self, delta=220):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, labels=None):
        if random.randint(0, 1):
            old_tyupe = image.dtype
            image = image.astype(np.float32)
            delta = random.uniform(0, self.delta)
            image += delta
            image[image > 255] = 255
            image = image.astype(old_tyupe)
        return image, labels

class RandomSaltNoise(object):

    def __init__(self, max_noise_rate=0.1):
        self.max_noise_rate = max_noise_rate

    def __call__(self, img, labels=None):

        if random.randint(0, 1):
            noise_count = int(img.size * self.max_noise_rate)
            for k in range(noise_count):
                i = int(np.random.random() * img.shape[1])
                j = int(np.random.random() * img.shape[0])
                value = np.random.random() * 100
                if img.ndim == 2:
                    img[j, i] = value
                elif img.ndim == 3:
                    img[j, i, 0] = value
                    img[j, i, 1] = value
                    img[j, i, 2] = value

        return img, labels

class RgbToGray(object):
    def __init__(self):
        pass

    def __call__(self, img, labels=None):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        img = img.reshape(img.shape[0], img.shape[1], 1)
        return img, labels

class NormalizeImage(object):
    def __init__(self, mean, std):
        self.mean = np.array(mean).astype(np.float32)
        self.std = np.array(std).astype(np.float32)
        pass

    def __call__(self, image, labels=None):
        image = image - self.mean
        image = image / self.std
        return image, labels


#common 几何变换
class ResizeImage(object):
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

        if labels is not None and len(labels) > 0:
            labels[:, 0] = labels[:, 0] * scale_w
            labels[:, 1] = labels[:, 1] * scale_h
            labels[:, 2] = labels[:, 2] * scale_w
            labels[:, 3] = labels[:, 3] * scale_h

        return image, labels


class RandomResizePadding(object):
    """随机缩放填充
    先改变图像的宽高比，在填充，保证图像宽高比不变
    改变图像主体内容的宽高比，而不改变图片的宽高比
    输入opencv图像和非归一化xyxy坐标
    """

    def __init__(self, stretch_list=[1.1, 1.2, 1.3, 1.4, 1.5], padding_value=[255, 255, 255]):
        self.stretch_list = stretch_list
        self.padding = padding_value

    def __call__(self, img, labels):
        if random.randint(0, 1):
            ratio = self.stretch_list[random.randint(0, len(self.stretch_list) - 1)]
            height = img.shape[0]
            widht = img.shape[1]

            # height < width
            if random.randint(0, 1):
                img = cv2.resize(img, (int(widht * ratio), height))
                det = int(((height * ratio) - height) / 2)
                img = cv2.copyMakeBorder(img, det, det, 0, 0, cv2.BORDER_CONSTANT, value=self.padding)
                if labels is not None and len(labels) > 0:
                    labels[:, 1:4:2] += det
                    labels[:, 0:3:2] = labels[:, 0:3:2].astype(np.float32) * ratio

            # height > width
            else:
                img = cv2.resize(img, (widht, int(height * ratio)))
                det = int(((widht * ratio) - widht) / 2)
                img = cv2.copyMakeBorder(img, 0, 0, det, det, cv2.BORDER_CONSTANT, value=self.padding)
                if labels is not None and len(labels) > 0:
                    labels[:, 0:3:2] += det
                    labels[:, 1:4:2] = labels[:, 1:4:2].astype(np.float32) * ratio

        return img, labels

class RandomCropFix(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, labels=None):
        h, w = image.shape[:2]
        crop_w, crop_h = self.size
        if crop_w > w or crop_h > h:
            raise ValueError('RdndomCrop failed')
        h_offset = random.randint(0, h - crop_h)
        w_offset = random.randint(0, w - crop_w)
        image = image[h_offset:h_offset + crop_h, w_offset:w_offset + crop_w]

        if labels is not None and len(labels) > 0:
            labels[:, 0:3:2] -= w_offset
            labels[:, 1:4:2] -= h_offset
            labels[:, 0:3:2] = np.clip(labels[:, 0:3:2], 0, crop_w)
            labels[:, 1:4:2] = np.clip(labels[:, 1:4:2], 0, crop_h)

        return image, labels

class RandomCropRatio(object):
    def __init__(self, ratio):
        assert (ratio > 0 and ratio <= 1)
        self.ratio = ratio

    def __call__(self, image, labels=None):
        h, w = image.shape[:2]
        crop_w, crop_h = (int(float(w) * self.ratio), int(float(h) * self.ratio))
        if crop_w > w or crop_h > h:
            raise ValueError('RdndomCrop failed')
        h_offset = random.randint(0, h - crop_h)
        w_offset = random.randint(0, w - crop_w)
        image = image[h_offset:h_offset + crop_h, w_offset:w_offset + crop_w]

        if labels is not None and len(labels) > 0:
            labels[:, 0:3:2] -= w_offset
            labels[:, 1:4:2] -= h_offset
            labels[:, 0:3:2] = np.clip(labels[:, 0:3:2], 0, crop_w)
            labels[:, 1:4:2] = np.clip(labels[:, 1:4:2], 0, crop_h)

        return image, labels

class CenterCropFix(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, labels=None):
        h, w = image.shape[:2]
        crop_w, crop_h = self.size
        if crop_w > w or crop_h > h:
            raise ValueError('RdndomCrop failed')
        h_offset = int((h - crop_h) / 2)
        w_offset = int((w - crop_w) / 2)
        image = image[h_offset:h_offset + crop_h, w_offset:w_offset + crop_w]

        if labels is not None and len(labels) > 0:
            labels[:, 0:3:2] -= w_offset
            labels[:, 1:4:2] -= h_offset
            labels[:, 0:3:2] = np.clip(labels[:, 0:3:2], 0, crop_w)
            labels[:, 1:4:2] = np.clip(labels[:, 1:4:2], 0, crop_h)

        return image, labels

class RandomHorizontalFlip(object):
    def __init__(self):
        pass

    def __call__(self, image, labels=None):
        if random.randint(0, 1):
            image = cv2.flip(image, 1)
            if labels is not None and len(labels) > 0:
                labels_old = copy.deepcopy(labels)
                h, w = image.shape[:2]
                labels[:, 0] = w - labels_old[:, 2]
                labels[:, 2] = w - labels_old[:, 0]
                pass
        return image, labels

class RandomExpand(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, labels):
        if random.randint(0, 1):
            height, width, depth = image.shape
            ratio = random.uniform(1, 3)
            left = random.uniform(0, width * ratio - width)
            top = random.uniform(0, height * ratio - height)

            expand_image = np.zeros(
                (int(height * ratio), int(width * ratio), depth),
                dtype=image.dtype)
            expand_image[:, :, :] = self.mean
            expand_image[int(top):int(top + height),
            int(left):int(left + width)] = image
            image = expand_image

            if labels is not None and len(labels) > 0:
                labels[:, 0] += int(left)
                labels[:, 1] += int(top)
                labels[:, 2] += int(left)
                labels[:, 3] += int(top)
        return image, labels

# just for cls

class ClsRandomPaddingWidth(object):
    def __init__(self, min_ratio=-0.3, max_ratio=0.3):
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def __call__(self, image, labels):
        height, width, depth = image.shape
        ratio = random.uniform(self.min_ratio, self.max_ratio)

        if ratio < 0:
            w_start = int(width * (-ratio) / 2)
            w_stop = int(width - width * (-ratio) / 2)
            return image[:, w_start:w_stop, :], labels
        else:
            det = int(width * (ratio) / 2)
            img = cv2.copyMakeBorder(image, 0, 0, det, det, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            return img, labels

class ClsRandomRotate(object):
    """随机旋转（只针对图像分类）
    输入opencv图像
    """

    def __init__(self, max_degree=45, filled_color=-1):
        self.max_degree = max_degree
        self.filled_color = filled_color

    def __call__(self, img, labels=None):

        if random.randint(0, 1):
            self.degree = random.randint(-self.max_degree, self.max_degree)
            # 获取旋转后4角的填充色
            if self.filled_color == -1:
                self.filled_color = mode([img[0, 0], img[0, -1],
                                          img[-1, 0], img[-1, -1]]).mode[0]
            if np.array(self.filled_color).shape[0] == 2:
                if isinstance(self.filled_color, int):
                    self.filled_color = (self.filled_color, self.filled_color, self.filled_color)
            else:
                self.filled_color = tuple([int(i) for i in self.filled_color])

            height, width = img.shape[:2]

            # 旋转后的尺寸
            height_new = int(width * fabs(sin(radians(self.degree))) +
                             height * fabs(cos(radians(self.degree))))
            width_new = int(height * fabs(sin(radians(self.degree))) +
                            width * fabs(cos(radians(self.degree))))

            mat_rotation = cv2.getRotationMatrix2D((width / 2, height / 2), self.degree, 1)

            mat_rotation[0, 2] += (width_new - width) / 2
            mat_rotation[1, 2] += (height_new - height) / 2

            # Pay attention to the type of elements of filler_color, which should be
            # the int in pure python, instead of those in numpy.
            img = cv2.warpAffine(img, mat_rotation, (width_new, height_new),
                                 borderValue=self.filled_color)
            # 填充四个角
            mask = np.zeros((height_new + 2, width_new + 2), np.uint8)
            mask[:] = 0
            seed_points = [(0, 0), (0, height_new - 1), (width_new - 1, 0),
                           (width_new - 1, height_new - 1)]
            for i in seed_points:
                cv2.floodFill(img, mask, i, self.filled_color)

        return img, labels

class ClsRandomRotate180(object):
    def __init__(self):
        pass
    def __call__(self, img, labels=None):
        img = cv2.flip(img, 0)
        img = cv2.flip(img, 1)
        return img, labels

class ClsAffine(object):
    def __init__(self):
        pass
    def __call__(self, image, labels):
        if random.randint(0, 1):
            height, width, depth = image.shape
            ratio_h = random.uniform(-0.08, +0.08)
            ratio_w = random.uniform(-0.08, +0.08)
            matSrc = np.float32([[0,0],[0,height-1],[width-1,0]])
            matDst = np.float32([[ratio_h*height, ratio_w*width],[0,height+1],[width+1,0]])
            matAffine = cv2.getAffineTransform(matSrc,matDst)
            dst = cv2.warpAffine(image,matAffine,(width,height), borderValue=(255,255,255))
            return dst, labels
        else:
            return image, labels



# just for det
class DetIgnoreBoxes(object):
    def __init__(self, min_area=0, ignore_class=[]):
        self.min_area = min_area
        self.ignore_class = ignore_class
    def __call__(self, image, labels):
        ignore_indexs = []
        if self.min_area > 0:
            assert(labels is not None)
            for i,label in enumerate(labels):
                if int(label[4]) in self.ignore_class or \
                   (label[2]-label[0])*(label[3]-label[1]) < self.min_area:
                    ignore_indexs.append(i)
        if len(ignore_indexs) > 0:
            #print('*** ignore %d boxes'%(len(ignore_indexs)))
            labels = np.delete(labels, ignore_indexs, axis = 0)

        return image, labels