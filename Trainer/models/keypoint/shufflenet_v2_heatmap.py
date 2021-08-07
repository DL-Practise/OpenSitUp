import torch
import torch.nn as nn
import math
import numpy as np
import os
import cv2
from DLEngine.modules.visualize.visual_util import *


__all__ = ['ShuffleNetV2HeatMap']


def channel_shuffle(x, groups):
    # type: (torch.Tensor, int) -> torch.Tensor
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride):
        super(InvertedResidual, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out


class FocalLoss(nn.Module):
    def __init__(self, weight=None, reduction='mean', gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()


class DUC(nn.Module):
    '''
    Initialize: inplanes, planes, upscale_factor
    OUTPUT: (planes // upscale_factor^2) * ht * wd
    '''

    def __init__(self, inplanes, planes, upscale_factor=2):
        super(DUC, self).__init__()
        self.conv = nn.Conv2d(
            inplanes, planes, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(planes, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pixel_shuffle(x)
        return x

class ShuffleNetV2HeatMap(nn.Module):
    def __init__(self, arg_dict):
        super(ShuffleNetV2HeatMap, self).__init__()

        self.kp_num = arg_dict['kp_num']
        width_mult = arg_dict['channel_ratio']
        inverted_residual=InvertedResidual
		
        if width_mult == 0.5:
            stages_repeats = [4, 8, 4]
            stages_out_channels = [24, 48, 96, 192, 1024]
        elif width_mult == 1.0:
            stages_repeats = [4, 8, 4]
            stages_out_channels = [24, 116, 232, 464, 1024]
        elif width_mult == 1.5:
            stages_repeats = [4, 8, 4]
            stages_out_channels = [24, 176, 352, 704, 1024]
        elif width_mult == 2.0:
            stages_repeats = [4, 8, 4]
            stages_out_channels = [24, 244, 488, 976, 2048]
        elif width_mult == 0.25:
            stages_repeats = [4, 8, 4]
            stages_out_channels = [24, 28, 48, 96, 512]
        else:
            assert(False)
		

        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels

        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(
                stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [inverted_residual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(inverted_residual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]

        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

        self.conv_compress = nn.Conv2d(1024, 256, 1, 1, 0, bias=False)
        self.duc1 = DUC(256, 512, upscale_factor=2)
        self.duc2 = DUC(128, 256, upscale_factor=2)
        #self.duc3 = DUC(64, 128, upscale_factor=2)
        self.conv_result = nn.Conv2d(64, self.kp_num , 1, 1, 0, bias=False)
        self.loss_func = torch.nn.MSELoss(size_average=None, reduce=None, reduction='sum')


    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = self.conv_compress(x)
        x = self.duc1(x)
        x = self.duc2(x)
        #x = self.duc3(x)
        x = self.conv_result(x)
        return x

    def forward(self, x):
        x = self._forward_impl(x)
        if self.training:
            return x
        else:
            heatmaps = nn.functional.sigmoid(x)
            return heatmaps

    def train_step(self, images, labels, local_rank):  
        preds = self.forward(images)
        preds = nn.functional.sigmoid(preds)
        batch = labels.shape[0]

        # visual
        epoch = int(os.environ['epoch'])
        epoch_changed = os.environ['epoch_changed']
        if epoch_changed == 'true' and local_rank == 0:
            visual_add_image_with_heatmap(images, preds, labels, mean=[103.53,116.28,123.675], std=[57.375,57.12,58.395], epoch=epoch)
            os.environ['epoch_changed'] = 'false'

        preds_head = preds[:, 0, :, :]
        preds_knee = preds[:, 1, :, :]
        preds_loin = preds[:, 2, :, :]
        labels_head = labels[:, 0, :, :]
        labels_knee = labels[:, 1, :, :]
        labels_loin = labels[:, 2, :, :]
        head_pos_mask = labels_head > 0
        head_neg_mask = labels_head == 0
        knee_pos_mask = labels_knee > 0
        knee_neg_mask = labels_knee == 0
        loin_pos_mask = labels_loin > 0
        loin_neg_mask = labels_loin == 0

        loss_head_pos = self.loss_func(preds_head[head_pos_mask], labels_head[head_pos_mask]) / labels.shape[0]
        loss_head_neg = self.loss_func(preds_head[head_neg_mask], labels_head[head_neg_mask]) / labels.shape[0]
        loss_knee_pos = self.loss_func(preds_knee[knee_pos_mask], labels_knee[knee_pos_mask]) / labels.shape[0]
        loss_knee_neg = self.loss_func(preds_knee[knee_neg_mask], labels_knee[knee_neg_mask]) / labels.shape[0]
        loss_loin_pos = self.loss_func(preds_loin[loin_pos_mask], labels_loin[loin_pos_mask]) / labels.shape[0]
        loss_loin_neg = self.loss_func(preds_loin[loin_neg_mask], labels_loin[loin_neg_mask]) / labels.shape[0]
        
        loss_total = loss_head_pos + 2.0*loss_knee_pos + loss_loin_pos + 0.1*loss_head_neg  + 0.2*loss_knee_neg + 0.1*loss_loin_neg
        return {'total': loss_total, 
                'h_pos':loss_head_pos, 
                'k_pos':loss_knee_pos,
                'l_pos': loss_loin_pos,
                'h_neg':loss_head_neg, 
                'k_neg:':loss_knee_neg,
                'l_neg:':loss_loin_neg}

    def eval_step(self, images):
        out = self.forward(images)
        pass
       
