import sys
import os
sys.path.append('./')
sys.path.append('../')
import torch
import matplotlib
matplotlib.use('Agg')
import os
from DLEngine.modules.cfg_parse.cfg_parse import parse_cfg_file
import models
import numpy as np

################config info#####################################
MODEL_DIR='save/keypoint_shufflenetv2_heatmap_224_1.0_3kps-20210402005406/'
NCNN_DIR='../../deploy/ncnn/ncnn-20201230/build_ubuntu18/'
MODEL_TYPE='keypoint' #classify / detection / keypoint
DETAIL_LOG=True
NEED_ONNX_SIM=True
QUANT_ENABLE=False
QUANT_IMG_DIR=''
#################################################################

def get_file_form_dir(dir_path, houzhui):
    files = os.listdir(dir_path)
    file_list = []
    for file in files:
        if str(file).endswith(houzhui) :
            file_list.append(os.path.join(dir_path, file))

    if len(file_list) == 0:
        return None
    elif len(file_list) == 1:
        return file_list[0]
    else:
        print('\nthere are more then one %s files, please select one'%(houzhui))
        for i,file in enumerate(file_list):
            print(i, file)
        print('\nplease input the seq')
        x = int(input())
        return file_list[x]


def load_pre_train_ignore_name(net, pre_train):
    if pre_train == '':
        print('the pre_train is null, skip')
        return
    else:
        print('the pre_train is %s' % pre_train)
        new_dict = {}
        pretrained_model = torch.load(pre_train, map_location=torch.device('cpu'))

        pre_keys = pretrained_model.keys()
        net_keys = net.state_dict().keys()
        print('net keys len:%d, pretrain keys len:%d' % (len(net_keys), len(pre_keys)))
        if len(net_keys) != len(pre_keys):
            print(
                'key lens not same, maybe the pytorch version for pretrain and net are difficent; use name load')
            for key_net in net_keys:
                strip_key_net = key_net.replace('module.', '')
                if strip_key_net not in pre_keys:
                    print('op: %s not exist in pretrain, ignore' % (key_net))
                    new_dict[key_net] = net.state_dict()[key_net]
                    continue
                else:
                    net_shape = str(net.state_dict()[key_net].shape).replace('torch.Size', '')
                    pre_shape = str(pretrained_model[strip_key_net].shape).replace('torch.Size', '')
                    if net.state_dict()[key_net].shape != pretrained_model[strip_key_net].shape:
                        print('op: %s exist in pretrain but shape difficenet(%s:%s), ignore' % (
                        key_net, net_shape, pre_shape))
                        new_dict[key_net] = net.state_dict()[key_net]
                    else:
                        print(
                            'op: %s exist in pretrain and shape same(%s:%s), load' % (key_net, net_shape, pre_shape))
                        new_dict[key_net] = pretrained_model[strip_key_net]

        else:
            for key_pre, key_net in zip(pretrained_model.keys(), net.state_dict().keys()):
                if net.state_dict()[key_net].shape == pretrained_model[key_pre].shape:
                    new_dict[key_net] = pretrained_model[key_pre]
                    print('op: %s shape same, load weights' % (key_net))
                else:
                    new_dict[key_net] = net.state_dict()[key_net]
                    print('op: %s:%s shape diffient(%s:%s), ignore weights' %
                                 (key_net, key_pre,
                                  str(net.state_dict()[key_net].shape).replace('torch.Size', ''),
                                  str(pretrained_model[key_pre].shape).replace('torch.Size', '')))

        net.load_state_dict(new_dict, strict=False)

if __name__ == '__main__':

    cfg_file = get_file_form_dir(MODEL_DIR, '.py')
    model_file = get_file_form_dir(MODEL_DIR, '.pkl')
    cfg_dicts = parse_cfg_file(cfg_file)
    onnx_file = model_file.replace('.pkl', '.onnx')
    onnx_sim_file = model_file.replace('.pkl', '_sim.onnx')
    ncnn_param_file = model_file.replace('.pkl', '.param')
    ncnn_param_bin_file = model_file.replace('.pkl', '.param.bin')
    ncnn_bin_file = model_file.replace('.pkl', '.bin')
    ncnn_id_file = model_file.replace('.pkl', '.id.h')
    ncnn_mem_file = model_file.replace('.pkl', '.mem.h')
    ncnn_cfg = model_file.replace('.pkl', '.cfg')
    onnx2ncnn = NCNN_DIR + '/tools/onnx/onnx2ncnn'
    ncnn2mem = NCNN_DIR + '/tools/ncnn2mem'
    ncnn2stable = NCNN_DIR + '/tools/quantize/ncnn2table'
    ncnn2int8 = NCNN_DIR + '/tools/quantize/ncnn2int8'
    print('@')
    print('@ 1.get all the file names:')
    print('@ config file:         ', cfg_file)
    print('@ pytroch model file:  ', model_file)
    print('@ onnx_file:           ', onnx_file)
    print('@ onnx_sim_file:       ', onnx_sim_file)
    print('@ ncnn_param_file:     ', ncnn_param_file)
    print('@ ncnn_param_bin_file: ', ncnn_param_bin_file)
    print('@ ncnn_bin_file:       ', ncnn_bin_file)
    print('@ ncnn_id_file:        ', ncnn_id_file)
    print('@ ncnn_mem_file:       ', ncnn_mem_file)
    print('@ onnx2ncnn:           ', onnx2ncnn)
    print('@ ncnn2mem:            ', ncnn2mem)
    print('@ ncnn2stable:         ', ncnn2stable)
    print('@ ncnn2int8:           ', ncnn2int8)


    # create net
    model_dict = cfg_dicts.model_dict
    model_name = model_dict['net']
    model_args = model_dict['net_arg']
    if 'torchvision' in model_name:
        assert ('num_classes' in model_args.keys())
        cmd = 'net = models.%s(pretrained=False, num_classes=%d)' % (model_name, model_args['num_classes'])
        exec(cmd)
    else:
        cmd = 'net = models.%s(model_args)' % (model_name)
        exec(cmd)
    load_pre_train_ignore_name(net, model_file)
    net.eval()
    print('@')
    print('@ 2.create pytroch net(%s) and load model'%(model_name))


    # create the input tensor
    input_w, input_h = cfg_dicts.data_dict['eval']['resize']
    input_shape = (1, 3, input_h, input_w)
    input = torch.FloatTensor(input_shape[0],input_shape[1],input_shape[2],input_shape[3])
    input = input.to('cpu')
    print('@')
    print('@ 3.create the input tensor')
    print('@ input_shape: ', input.shape)


    # export to onnx file
    torch.onnx.export(net,input,onnx_file,verbose=DETAIL_LOG)
    print('@')
    print('@ 4.export to onnx model')


    # sim the onnx file
    print('@')
    if NEED_ONNX_SIM:
        cmd = 'python3 -m onnxsim ' + str(onnx_file) + ' ' + str(onnx_sim_file)
        ret = os.system(str(cmd))
        #print(ret)
        new_onnx_file = onnx_sim_file
        print('@ 5.need to sim the onnx model')
    else:
        new_onnx_file = onnx_file
        print('@ 5.do not need to sim the onnx model')
    print('@ use this onnx file: ', new_onnx_file)


    # trans onnx to ncnn
    cmd = str(onnx2ncnn) + ' ' + str(new_onnx_file) + ' ' + str(ncnn_param_file) + ' ' + str(ncnn_bin_file)
    ret = os.system(str(cmd))
    print('@')
    print('@ 6.trans onnx model to ncnn')
    print('@ cmd is: ', cmd)


    # quantize the model
    print('@')
    print('@ 7.quantize the ncnn model')
    if QUANT_ENABLE:
        img_size = '%d,%d'%(input_shape[3], input_shape[2])
        mean_value = cfg_dicts.data_dict['eval']['mean']
        std_value = cfg_dicts.data_dict['eval']['std']
        table_file = model_file.replace('.pkl', '.table')
        print('@ img dir is : ', QUANT_IMG_DIR)
        print('@ img_size is : ', img_size)
        print('@ mean_value is : ', mean_value)
        print('@ std_value is : ', std_value)

        cmd = str(ncnn2stable) + \
              ' --param=' + str(ncnn_param_file) + \
              ' --bin=' + str(ncnn_bin_file) + \
              ' --images=' + str(QUANT_IMG_DIR) + \
              ' --output=' + str(table_file) + \
              ' --mean=' + str(mean_value) + \
              ' --norm=' + str(std_value) + \
              ' --size=' + str(img_size) + \
              ' --swapRB' + \
              ' --thread=2'
        ret = os.system(str(cmd))

        old_param_file = ncnn_param_file
        old_bin_file = ncnn_bin_file
        ncnn_param_file = old_param_file.replace('.param', '_quant.param')
        ncnn_bin_file = old_bin_file.replace('.bin', '_quant.bin')
        cmd = str(ncnn2int8) + \
              ' ' + str(old_param_file) + \
              ' ' + str(old_bin_file) + \
              ' ' + str(ncnn_param_file) + \
              ' ' + str(ncnn_bin_file) + \
              ' ' + str(table_file)
        print(cmd)
        ret = os.system(str(cmd))
    else:
        print('@ do not need to quantize')


    # encirpt the ncnn model
    cmd = str(ncnn2mem) + ' ' + str(ncnn_param_file) + ' ' + str(ncnn_bin_file) + ' ' + str(ncnn_id_file) + ' ' + str(ncnn_mem_file)
    ret = os.system(str(cmd))
    cmd = 'mv ./*.param.bin ' + str(MODEL_DIR)
    ret = os.system(str(cmd))
    print('@')
    print('@ 8.encript ncnn model')


    # create the ncnn cfg file
    with open(ncnn_id_file, 'r') as f:
        for line in f.readlines():
            if 'const int ' in line.strip():
                out_blob_id = int(line.strip().split('=')[1].split(';')[0])
    

    if MODEL_TYPE == 'classify':
        shape = cfg_dicts.data_dict['eval']['crop']
    elif MODEL_TYPE == 'detection':
        shape = cfg_dicts.data_dict['eval']['crop']
    elif MODEL_TYPE == 'keypoint':
        shape = cfg_dicts.data_dict['eval']['resize']

    mean = cfg_dicts.data_dict['eval']['mean']
    std = 1.0 / np.array(cfg_dicts.data_dict['eval']['std'])
    with open(ncnn_cfg, 'w') as f:
        f.writelines('[ncnn]\n')
        f.writelines('model_type = %s\n'%MODEL_TYPE)
        f.writelines('#w and h for input\n')
        f.writelines('input_shape = %d,%d\n'%(shape[0], shape[1]))
        f.writelines('out_index = %d\n' % (out_blob_id))
        f.writelines('mean = %.6f,%.6f,%.6f\n' % (mean[0], mean[1], mean[2]))
        f.writelines('std = %.6f,%.6f,%.6f\n' % (std[0], std[1], std[2]))