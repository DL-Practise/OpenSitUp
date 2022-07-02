import sys
import os
import torch
import math

def calculate_oks(result_info):

    preds = result_info['preds']
    labels = result_info['labels']
    assert(len(preds) == len(labels))


    pred_kp_list = preds
    gt_kt_list = labels

    total_len = 0.0
    last_gt_point = None
    for gt_point in gt_kt_list:
        if last_gt_point is not None:
            total_len += math.sqrt((last_gt_point[0] - gt_point[0])**2 + (last_gt_point[1] - gt_point[1])**2)
        last_gt_point = gt_point

    norm_dis_list = []
    for pred_kp, gt_kp in zip(pred_kp_list, gt_kt_list)
        dis = math.sqrt((pred_kp[0] - gt_kp[0])**2 + (pred_kp[1] - gt_kp[1])**2)
        norm_dis = dis / total_len
        norm_dis_list.append(norm_dis)

    oks_list = []
    for norm_dis in norm_dis_list:
        oks_list.append(math.exp( -norm_dis / 2*0.5**2 ))
    oks = np.array(oks_list).mean()

    return oks
