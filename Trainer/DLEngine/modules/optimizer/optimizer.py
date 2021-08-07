import sys
sys.path.append('./')
sys.path.append('../')
import torch
import logging
import numpy as np
import math


def check_only_train(op_name, only_train_list):
    if len(only_train_list) == 0:
        return True
    else:
        for only_train_name in only_train_list:
            if only_train_name in op_name:
                return True
        return False

def check_lr_multy(op_name, lr_multy_map):
    for name in lr_multy_map.keys():
        if name in op_name:
            lr_multy = lr_multy_map[name]
            return lr_multy
    else:
        return 1.0


def create_optimizer(module, opt_dict):
    opt_type = opt_dict['opt_type']
    base_lr = opt_dict['base_lr']
    weight_decay = opt_dict['weight_decay']
    if 'only_train' not in opt_dict.keys():
        only_train = []
    else:
        only_train = opt_dict['only_train']
    if 'lr_multy' not in opt_dict.keys():
        lr_multy_map = {}
    else:
        lr_multy_map = opt_dict['lr_multy']

    params = []
    for name, parameter in module.named_parameters():
        if parameter.requires_grad and check_only_train(name, only_train):
            lr_multy = check_lr_multy(name, lr_multy_map)
            logging.info("op: %s need train with lr_multy: %.2f" % (name,lr_multy))
            if 'bias' in name:
                #params += [{'params': [parameter], 'lr': 2*base_lr*lr_multy, 'weight_decay': 0}]
                params += [{'params': [parameter], 'lr': base_lr * lr_multy, 'weight_decay': 0}]
            else:
                params += [{'params': [parameter], 'lr': base_lr*lr_multy, 'weight_decay': weight_decay}]
        else:
            logging.info("op: %s do not need train" % name)
            parameter.requires_grad = False

    if opt_type == "sgd":
        logging.info("use optimize : momentum sgd")
        momentum = opt_dict['momentum']
        optimizer = torch.optim.SGD(params, momentum=momentum)

    elif opt_type == "nag":
        logging.info("use optimize : momentum sgd with nag")
        momentum = opt_dict['momentum']
        optimizer = torch.optim.SGD(params, momentum=momentum, nesterov=True)

    elif opt_type == "adagrad":
        logging.info("use optimize : moment adagrad")
        optimizer = torch.optim.Adagrad(params)

    elif opt_type == "adam":
        logging.info("use optimize : adam")
        momentum = opt_dict['momentum']
        momentum2 = opt_dict['momentum2']
        optimizer = torch.optim.Adam(params, betas=(momentum, momentum2),eps=1e-08)
    else:
        logging.error("unsupport opt type: %s"%opt_type)
        exit(0)

    return optimizer

