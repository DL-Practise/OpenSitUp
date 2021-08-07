import sys
import os
import torch
import argparse
import shutil
import time
import logging
from .modules.optimizer.optimizer import create_optimizer
from .modules.lr_schedule.lr_schedule import LrSchedule
from .modules.dataloader.dataloader import create_val_dataloader
from .modules.evaluater.evaluater import Evaluater
from .modules.cfg_parse.cfg_parse import parse_cfg_file
import importlib


class EvalProject():
    def __init__(self, net, cfg_file, val_dataset):
        self.cfg_file = cfg_file
        self.net = net
        cfg_dicts = parse_cfg_file(cfg_file)
        self.model_dict = cfg_dicts.model_dict
        self.data_dict = cfg_dicts.data_dict
        self.eval_dict = cfg_dicts.train_dict['eval']
        self.device = cfg_dicts.train_dict['device']
        self.val_dataset = val_dataset
        self.proj_init()
        self.eval_init()


    def proj_init(self):
        # get the device and set device
        assert(self.device == 'cuda' or self.device == 'cpu')
        if self.device == 'cuda':
            torch.cuda.set_device(0)

    def eval_init(self):
        # 1. create dataloader
        print('eval_init: create dataloader')
        self.eval_dataloader = create_val_dataloader(self.val_dataset,
                                                     self.data_dict['eval'])

        # 2. distributed the net
        print('eval_init: move net to ', self.device)
        self.net.to(self.device)


        # 6. create the trainer
        print('eval_init: create evaluater')
        self.evaluater= Evaluater(self.net,
                                  self.eval_dataloader,
                                  self.device,
                                  self.eval_dict)

    def eval(self):
        self.evaluater.run()
