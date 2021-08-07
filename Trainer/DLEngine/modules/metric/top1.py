import sys
import os
import torch

def calculate_top1(result_info):
    preds = result_info['preds']
    labels = result_info['labels']
    assert(len(preds) == len(labels))
    correct = 0
    total = 0
    for i in range(len(preds)):
        pred = preds[i]
        label = labels[i]
        pred_class = pred.argmax(dim=1)



        correct += torch.eq(pred_class, label).sum().float().item()
        total += pred.shape[0]
    top1 = correct / total
    return top1
