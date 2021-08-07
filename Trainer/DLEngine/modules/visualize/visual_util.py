import numpy as np
import time
import math
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
import os
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

vis_enable = False
vis_writer = None
vis_count = {'image_with_box': 0}

def visual_init(proj_dir):
    global vis_writer
    global vis_enable
    vis_writer = SummaryWriter(log_dir=proj_dir + '/tensorboard')
    vis_enable = True

def visual_enable():
    global vis_enable
    return vis_enable

def visual_add_scale(name, value, iter):
    if not visual_enable():
        return 

    global vis_writer
    vis_writer.add_scalar(name, value, iter)

def visual_add_image_with_box(image, boxes, iter, max_cout=1):
    if not visual_enable():
        return 

    global vis_writer
    global vis_count
    if vis_count['image_with_box'] >= max_cout:
        return
    image_np = image.detach().cpu().numpy().transpose(1, 2, 0)
    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
    fig = plt.figure('image', figsize=(4, 4))
    plt.clf()
    plt.imshow(image_np)
    for box in boxes:
        x1 = box[0].item()
        y1 = box[1].item()
        x2 = box[2].item()
        y2 = box[3].item()
        plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], color='red')
    vis_writer.add_figure(tag='image', figure=fig, global_step=iter)
    vis_count['image_with_box'] += 1

def visual_add_yolov3_targets(coord_mask, conf_pos_mask, conf_neg_mask, cls_mask, tcoord, tconf, tcls):
    if not visual_enable():
        return 
    print(coord_mask, conf_pos_mask, conf_neg_mask, cls_mask, tcoord, tconf, tcls)

def visual_add_image_with_heatmap(images, preds, labels, mean, std, epoch):
    if not visual_enable():
        return 
    fig = plt.figure(figsize=(10, 10), dpi=100)
    plt.clf()

    label0 = labels[0].cpu().detach().numpy()
    pred0 = preds[0].cpu().detach().numpy()
    image0 = images[0].cpu().detach().numpy().transpose(1,2,0)
    image0 = image0 * std
    image0 = image0 + mean
    image0 = image0.astype(np.uint8)
    h, w = image0.shape[0:2]

    kp_num = label0.shape[0]
    for kp_c in range(kp_num):
        plt.subplot(2, kp_num, kp_c + 1)
        plt.imshow(image0)
        plt.imshow(cv2.resize(pred0[kp_c], (w, h)), alpha=0.5)
    
    for kp_c in range(kp_num):
        plt.subplot(2, kp_num, kp_num + kp_c + 1)
        plt.imshow(image0)
        plt.imshow(cv2.resize(label0[kp_c], (w, h)), alpha=0.5)

    plt.savefig('./tran_%d.jpg'%epoch)
    vis_writer.add_figure(tag='train', figure=fig, global_step=epoch)       

def draw_det_img(img_paths, preds, labels, title='', save_path=None):
    if not visual_enable():
        return 
    img_count = len(img_paths)
    assert(img_count) <= 16
    side = int(math.sqrt(img_count))
    plt.figure(figsize=(side*5, side*5))
    plt.suptitle(title)
    for i in range(side):
        for j in range(side):
            index = i*side + j
            img = cv2.imread(img_paths[index])
            plt.subplot(side, side, index + 1)
            plt.title(os.path.basename(img_paths[index]))
            plt.imshow(img)

            for cls in preds.keys():
                for box in preds[cls]:
                    img_id = int(box[1])
                    if img_id != index:
                        continue
                    prob = float(box[0])
                    x1 = int(box[2])
                    y1 = int(box[3])
                    x2 = int(box[4])
                    y2 = int(box[5])
                    plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], color='red')
                    plt.text(x1, y1, '%d:%.2f' % (cls, prob), color='red')

            if index in labels.keys():
                for cls in labels[index].keys():
                    for box in labels[index][cls]:
                        x1 = int(box[0])
                        y1 = int(box[1])
                        x2 = int(box[2])
                        y2 = int(box[3])
                        plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], color='blue')
                        plt.text(x1, y1, '%d' % (cls), color='blue')
    plt.savefig(save_path)
    plt.show()

def draw_pr_curve(aps, recalls, precisions, title='', save_path=None):
    if not visual_enable():
        return 
    class_count = len(aps)
    assert(class_count) <= 16
    side = int(math.sqrt(class_count))
    plt.figure(figsize=(side*5, side*5))
    plt.suptitle(title)
    for i in range(class_count):
        plt.subplot(side, side, i + 1)
        plt.title('class=%d ap=%.4f'%(i, aps[i]))
        print('precisions for class: ',i)
        print(precisions[i])
        print('recalls for class: ', i)
        print(recalls[i])
        plt.plot(recalls[i], precisions[i], color='red')
        plt.xlabel("recall")
        plt.ylabel("precision")

    plt.savefig(save_path)
    plt.show()
