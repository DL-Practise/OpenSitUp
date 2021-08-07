import sys
import os
import importlib
import torch
from torch import nn
from DLEngine.eval_project import EvalProject
from DLEngine.modules.cfg_parse.cfg_parse import parse_cfg_file
import models
#from torchstat import stat
import data
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import math

############################################################
MODEL_FILE='save/keypoint_shufflenetv2_heatmap_2stages_224_1.0_3kps-20210402020950/model_48.pkl'
CFG_FILE='cfgs/key_point/keypoint_shufflenetv2_heatmap_2stages_224_1.0_3kps.py'
IMG_PATH='/data/zhengxing/my_dl/dataset/sit_up/test_images/ttt'
#IMG_PATH='/data/zhengxing/temp/key_point_images/'
############################################################

def load_pre_train_ignore_name(net, pre_train):
    if pre_train == '':
        print('the pre_train is null, skip')
        return
    else:
        print('the pre_train is %s' % pre_train)
        new_dict = {}
        pretrained_model = torch.load(pre_train, map_location=torch.device('cpu'))

        pre_keys = pretrained_model.keys()
        net_keys = net.state_dict().keys()
        print('net keys len:%d, pretrain keys len:%d' % (len(net_keys), len(pre_keys)))
        if len(net_keys) != len(pre_keys):
            print(
                'key lens not same, maybe the pytorch version for pretrain and net are difficent; use name load')
            for key_net in net_keys:
                strip_key_net = key_net.replace('module.', '')
                if strip_key_net not in pre_keys:
                    print('op: %s not exist in pretrain, ignore' % (key_net))
                    new_dict[key_net] = net.state_dict()[key_net]
                    continue
                else:
                    net_shape = str(net.state_dict()[key_net].shape).replace('torch.Size', '')
                    pre_shape = str(pretrained_model[strip_key_net].shape).replace('torch.Size', '')
                    if net.state_dict()[key_net].shape != pretrained_model[strip_key_net].shape:
                        print('op: %s exist in pretrain but shape difficenet(%s:%s), ignore' % (
                        key_net, net_shape, pre_shape))
                        new_dict[key_net] = net.state_dict()[key_net]
                    else:
                        print(
                            'op: %s exist in pretrain and shape same(%s:%s), load' % (key_net, net_shape, pre_shape))
                        new_dict[key_net] = pretrained_model[strip_key_net]

        else:
            for key_pre, key_net in zip(pretrained_model.keys(), net.state_dict().keys()):
                if net.state_dict()[key_net].shape == pretrained_model[key_pre].shape:
                    new_dict[key_net] = pretrained_model[key_pre]
                    print('op: %s shape same, load weights' % (key_net))
                else:
                    new_dict[key_net] = net.state_dict()[key_net]
                    print('op: %s:%s shape diffient(%s:%s), ignore weights' %
                                 (key_net, key_pre,
                                  str(net.state_dict()[key_net].shape).replace('torch.Size', ''),
                                  str(pretrained_model[key_pre].shape).replace('torch.Size', '')))

        net.load_state_dict(new_dict, strict=False)

def get_need_test_images(the_path):
    img_list = []
    if os.path.isdir(the_path):
        files = os.listdir(the_path)
        for file in files:
            if str(file).endswith('.jpg') or str(file).endswith('.png'):
                img_list.append(the_path + '/' + str(file))
    elif os.path.isfile(the_path):
        img_list.append(the_path)
    if len(img_list) > 16:
        return img_list[0:16]
    return img_list

def key_point_postproc(img_ori, net_out, thres, i, total):
    
    def draw_line(pose_1, pose_2, img_size):
        if pose_1[2] > 0 and pose_2[2] > 0:
            plt.plot([pose_1[0]*img_size, pose_2[0]*img_size],[pose_1[1]*img_size, pose_2[1]*img_size,], color='red')

    def draw_head(pose_1, pose_2, img_size):
        if pose_1[2] > 0 and pose_2[2] > 0:
            plt.plot([pose_1[0]*img_size, pose_2[0]*img_size],[pose_1[1]*img_size, pose_2[1]*img_size,], color='red')

    side = math.ceil(math.sqrt(total))
    plt.subplot(side, side, i+1)
    plt.axis('off')

    preds = net_out.squeeze().cpu().detach().numpy()
    img_resize = cv2.resize(img_ori, (224, 224))[:,:,[2,1,0]]
    results = []
    for c in range(13):
        info = preds[c]
        max_score = info.max()
        if max_score > thres:
            max_pos_ = info.argmax()
            max_pos = np.unravel_index(max_pos_, info.shape)
            pos_y = max_pos[0] / 55.0
            pos_x = max_pos[1] / 55.0
            results.append([pos_x, pos_y, 1.0])
        else:
            results.append([-1.0, -1.0, 0.0])

    plt.imshow(img_resize)    
    if results[1][2] > 0 and results[2][2] > 0 and results[0][2] > 0:
        result_temp = [0.0, 0.0, 1.0]
        result_temp[0] = (results[1][0] + results[2][0]) / 2.0
        result_temp[1] = (results[1][1] + results[2][1]) / 2.0
        draw_line(results[0], result_temp , 224)

    draw_line(results[1], results[2], 224)
    draw_line(results[1], results[3], 224)
    draw_line(results[2], results[4], 224)
    draw_line(results[3], results[4], 224)
    draw_line(results[1], results[5], 224)
    draw_line(results[5], results[7], 224)
    draw_line(results[2], results[6], 224)
    draw_line(results[6], results[8], 224)
    draw_line(results[3], results[9], 224)
    draw_line(results[9], results[11], 224)
    draw_line(results[4], results[10], 224)
    draw_line(results[10], results[12], 224)

def key_point_postproc_simple(img_ori, net_out, thres, i, total):
    show_heatmap = False
    show_scatter = True

    preds = net_out.squeeze().cpu().detach().numpy()
    kp_num = preds.shape[0]
    img_resize = cv2.resize(img_ori, (224, 224))[:,:,[2,1,0]]

    side = math.ceil(math.sqrt(total))
    plt.subplot(side, side, i+1)
    plt.axis('off')

    if show_heatmap:
        img_show = None
        heatmap_show = None
        for c in range(kp_num):
            heatmap = cv2.resize(preds[c], (224, 224))

            if img_show is None:
                img_show = img_resize
            else:
                img_show = np.concatenate((img_show,img_resize),axis=1)

            if heatmap_show is None:
                heatmap_show = heatmap
            else:
                heatmap_show = np.concatenate((heatmap_show,heatmap),axis=1)
            
        plt.imshow(img_show)  
        plt.imshow(heatmap_show, alpha=0.5)
            
    
    if show_scatter:
        plt.imshow(img_resize)  
        all_points_x = []
        all_points_y = []
        for c in range(kp_num):
            max_score = preds[c].max()
            if max_score > thres:
                max_pos_ = preds[c].argmax()
                max_pos = np.unravel_index(max_pos_, preds[c].shape)
                pos_y = max_pos[0] * 224 / (preds[c].shape[0] - 1)
                pos_x = max_pos[1] * 224 / (preds[c].shape[1] - 1)
                all_points_x.append(pos_x)
                all_points_y.append(pos_y)
                if c == 0:
                    plt.scatter(pos_x, pos_y, s=80, c='r')
                if c == 1:
                    plt.scatter(pos_x, pos_y, s=80, c='g')
                if c == 2:
                    plt.scatter(pos_x, pos_y, s=80, c='b')
        if len(all_points_x) == 3:
            plt.plot([all_points_x[0],all_points_x[2],all_points_x[1]], [all_points_y[0],all_points_y[2],all_points_y[1]], c='b')
        

if __name__ == '__main__':
    # import cfg_dicts from cfg_file
    cfg_dicts = parse_cfg_file(CFG_FILE)

    # create net
    model_dict = cfg_dicts.model_dict
    model_name = model_dict['net']
    model_args = model_dict['net_arg']
    print(model_name)
    print(model_args)
    net = models.__dict__[model_name](model_args)
    print(net)
    load_pre_train_ignore_name(net, MODEL_FILE)
    net.eval()
    net.cuda()

    # create dataloader
    data_name = cfg_dicts.data_dict['eval']['data_name']
    dataset = data.__dict__[data_name]('eval', cfg_dicts.data_dict['eval'])

    # get img lists and process
    plt.figure(figsize=(20, 20), dpi=100)
    img_list = get_need_test_images(IMG_PATH)
    for i, img_path in enumerate(img_list):
        img, img_ori, ori_w, ori_h = dataset.read_image(img_path)
        img_t = torch.from_numpy(img).unsqueeze(0).cuda()

        # the network inference
        preds = net(img_t)

        # keypoint postproc
        key_point_postproc_simple(img_ori, preds, 0.1,  i, len(img_list))

    plt.savefig('./infer.jpg')

