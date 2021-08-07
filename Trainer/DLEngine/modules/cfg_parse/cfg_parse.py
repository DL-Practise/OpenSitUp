import sys
import os
import importlib
import shutil
import time

def parse_cfg_file(cfg_file_path):
    # import cfg_dicts from cfg_file
    dir_path = os.path.dirname(os.path.abspath(cfg_file_path))
    new_cfg_path = dir_path + '/temp_cfg.py'
    shutil.copyfile(cfg_file_path, new_cfg_path)
    time.sleep(1)
    sys.path.append(dir_path)
    cfg_dicts = importlib.import_module('temp_cfg')
    #os.remove(new_cfg_path)
    return cfg_dicts
