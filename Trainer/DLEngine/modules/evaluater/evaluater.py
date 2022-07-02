import torch
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
import torch.distributed as dist
import time
import os
import logging
import shutil
from DLEngine.modules.metric.map import calculate_map
from DLEngine.modules.metric.top1 import calculate_top1
from DLEngine.modules.metric.oks import calculate_oks


class Evaluater():
    def __init__(self, net, dataloders, device, eval_dict):
        self.net = net
        self.eval_loader = dataloders
        self.device = device
        self.eval_dict = eval_dict

    def run(self):
        self.net.eval()
        result_info = {'preds': [], 'labels': []}
        for i, (images, labels) in enumerate(self.eval_loader):
            print("evaluate: %d/%d"%(i+1, len(self.eval_loader)))
            images = images.to(self.device)
            labels = labels.to(self.device)
            preds = self.net.eval_step(images)
            result_info['preds'].append(preds.detach())
            result_info['labels'].append(labels)
        if self.eval_dict['eval_type'] == 'top1':
            top1 = calculate_top1(result_info)
            print("reslut: top1=%.4f" % (top1))
        elif self.eval_dict['eval_type'] == 'oks':
            oks = calculate_oks(result_info)
            print("reslut: osk=%.4f" % (oks))
        else:
            print("unknown eval type: %s" % (self.eval_dict['eval_type']))

