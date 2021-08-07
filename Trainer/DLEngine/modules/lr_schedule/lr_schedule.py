import sys
import torch
import math
import logging
import numpy as np

class LrSchedule:
    def __init__(self, optimizer, opt_dict):
        self.opt_dict = opt_dict
        self.lr_policy = opt_dict['lr_policy']
        self.warmup_iter = opt_dict['warmup_iter']
        self.opt = optimizer
        self.get_base_lr()
        self.lr_multy = 0.0


    def get_base_lr(self):
        self.base_lr_list = []
        for param_group in self.opt.param_groups:
            self.base_lr_list.append(param_group['lr'])

    def update_lr(self, cur_iter, max_iter):
        # warmup lr
        warmup_iter = int(self.warmup_iter * max_iter)
        if warmup_iter < 1:
            warmup_iter = 1
        if cur_iter <= warmup_iter:
            #new_lr = self.base_lr * cur_iter / warmup_iter
            new_lr_multy =  cur_iter / warmup_iter
        else:
            if self.lr_policy == "step":
                lr_rate = float(self.opt_dict['lr_rate'])
                steps = self.opt_dict['lr_steps']
                steps = [int(s * max_iter) for s in steps]
                for i, step in enumerate(steps):
                    if cur_iter == step:
                        #new_lr = self.base_lr * lr_rate ** (i + 1)
                        new_lr_multy = lr_rate ** (i + 1)

            elif self.lr_policy == "cos":
                #new_lr = 0.5 * self.base_lr * (1.0 + math.cos(cur_iter * 3.1415926 / max_iter))
                new_lr_multy = 0.5 * (1.0 + math.cos(cur_iter * 3.1415926 / max_iter))
            else:
                logging.error("unsupport lr policy: %s"%self.lr_policy)
                exit(0)

        if 'new_lr_multy' in locals().keys():
            self.lr_multy = new_lr_multy
            for i, param_group in enumerate(self.opt.param_groups):
                param_group['lr'] = self.base_lr_list[i] * new_lr_multy

        return np.unique(np.array(self.base_lr_list) * self.lr_multy)
