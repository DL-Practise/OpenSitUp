opt_dict = {
    'opt_type': 'sgd',
    'momentum': 0.9,
    'base_lr': 0.01,
    'lr_policy': 'step',
    'lr_steps': [0.65,0.85],
    'lr_rate': 0.1,
    'warmup_iter': 0.01,
    'weight_decay': 0.0005,
    #'only_train': ['conv_compress', 'duc1', 'duc2', 'duc3', 'conv_result']
    'only_train': []
}

model_dict = {
    'net': 'ShuffleNetV2HeatMap',
    'net_arg': {'kp_num':3, 'channel_ratio':1.0,},
    'pre_train': 'save/torchvision/shufflenetv2_x1-5666bf0f80_bgr.pth',
}

data_dict = {
    'train':{
        'data_name': 'PersionKeypointTxt',
        'num_workers': 6,
        'data_dir': '../DataSet/',
        'data_label': '../DataSet/label_train.txt',
        'batch_size': 32,
        'resize': [224, 224], # w and h
        'mean': [103.53,116.28,123.675],
        'std': [57.375,57.12,58.395],
        'kp_num':3,
        'gauss_ratio': 2,
        'gauss_sigma': 1,
        'heatmap': [28, 28], # w and h
        'data_len_expand': 1, #100, 
        },
    'eval':{
        'data_name': 'PersionKeypointTxt',
        'num_workers': 6,
        'data_dir': '../DataSet/',
        'data_label': '../DataSet/label_val.txt',
        'batch_size': 16,
        'resize': [224, 224], # w and h
        'mean': [103.53,116.28,123.675],
        'std': [57.375,57.12,58.395],
        'kp_num':2,
        'gauss_ratio': 2,
        'gauss_sigma': 1,
        'heatmap': [28, 28], # w and h
        },
}

train_dict = {
    'device': 'cpu',  #'cuda' or 'cpu'
    'enable_visual': False,
    'save_dir': '',
    'max_epoch': 24,
    'train_display': 10,
    'train_save': 1,
    'eval': {
        'eval_enable': True,
        'start_eval': 1,
        'eval_epoch': 1,
        'eval_type': 'oks'
    },
}
