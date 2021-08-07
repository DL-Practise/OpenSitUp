import sys
import os
import importlib
import torch
from torch import nn
from DLEngine.train_project import TrainProject
from DLEngine.modules.cfg_parse.cfg_parse import parse_cfg_file
import models
import data
from torchstat import stat
#from thop import profile


############################################################
CFG_FILE='cfgs/key_point/keypoint_shufflenetv2_heatmap_224_1.0_3kps.py'
############################################################


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

    # calculate net flops
    #stat(net, (3, 224, 224))
    #input = torch.randn(1,3, 224, 224)
    #macs,params = profile(net, inputs=(input,))
    #print(macs, params)
    #exit(0)

    # create dataset
    data_name = cfg_dicts.data_dict['train']['data_name']
    dataset_train = data.__dict__[data_name]('train', cfg_dicts.data_dict['train'])
    if cfg_dicts.train_dict['eval']['eval_enable']:
        data_name = cfg_dicts.data_dict['eval']['data_name']
        dataset_eval = data.__dict__[data_name]('eval', cfg_dicts.data_dict['eval'])
    else:
        dataset_eval = None

    # create train project
    train_project = TrainProject(net, CFG_FILE, dataset_train, dataset_eval)

    # train
    train_project.train()
