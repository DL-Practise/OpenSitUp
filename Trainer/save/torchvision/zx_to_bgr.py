import sys
import os
import torch

model_file='shufflenetv2_x1-5666bf0f80.pth'

model_weights = torch.load(model_file, map_location=torch.device('cpu'))
for i, key in enumerate(model_weights.keys()):
    if i == 0:
        print('start to trans for weight:', key)
        params = model_weights[key]
        nOut = params.shape[0]
        nIn = params.shape[1]
        assert(nIn == 3)
        bgr_params = torch.zeros_like(params)
        for n in range(nOut):
            bgr_params[n][0] = params[n][2]
            bgr_params[n][1] = params[n][1]
            bgr_params[n][2] = params[n][0]
        model_weights[key] = bgr_params
    else:
        break

torch.save(model_weights, model_file.replace('.pth', '_bgr.pth'))