import sys
import os
import torch
import argparse
import shutil
import time
import logging
from .modules.optimizer.optimizer import create_optimizer
from .modules.lr_schedule.lr_schedule import LrSchedule
from .modules.dataloader.dataloader import create_train_dataloader, create_val_dataloader
from .modules.trainer.trainer import Trainer
from .modules.cfg_parse.cfg_parse import parse_cfg_file


class TrainProject():
    def __init__(self, net, cfg_file, train_dataset, val_dataset):
        # import cfg_dicts from cfg_file
        cfg_dicts = parse_cfg_file(cfg_file)

        self.opt_dict = cfg_dicts.opt_dict
        self.model_dict = cfg_dicts.model_dict
        self.data_dict = cfg_dicts.data_dict
        self.train_dict = cfg_dicts.train_dict
        self.cfg_file = cfg_file
        self.eval_enable = self.train_dict['eval']['eval_enable']
        self.net = net
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.proj_init()
        self.train_init()

    def load_pre_train(self):
        pre_train = self.model_dict['pre_train']
        if pre_train == '':
            return
        else:
            pretrained_model = torch.load(pre_train)
            is_pretrain_distribute = False
            for key in pretrained_model.keys():
                if 'module.' in key:
                    is_pretrain_distribute = True

            is_net_distribute = False
            for key in self.net.state_dict().keys():
                if 'module.' in key:
                    is_net_distribute = True

            if is_pretrain_distribute and not is_net_distribute:
                pretrained_model = {k.replace('module.', ''): v for k, v in pretrained_model.items()}

            if not is_pretrain_distribute and is_net_distribute:
                pretrained_model = {'module.' + key: v for k, v in pretrained_model.items()}

            self.net.load_state_dict(pretrained_model, strict=False)

    def load_pre_train_ignore_name(self):
        pre_train = self.model_dict['pre_train']
        if pre_train == '':
            logging.info('the pre_train is null, skip')
            return
        else:
            logging.info('the pre_train is %s'%pre_train)
            new_dict = {}
            pretrained_model = torch.load(pre_train, map_location=torch.device('cpu'))

            pre_keys = pretrained_model.keys()
            net_keys = self.net.state_dict().keys()
            logging.info('net keys len:%d, pretrain keys len:%d'%(len(net_keys), len(pre_keys)))
            if len(net_keys) != len(pre_keys):
                logging.info('key lens not same, maybe the pytorch version for pretrain and net are difficent; use name load')
                for key_net in net_keys:
                    strip_key_net = key_net.replace('module.', '')
                    if strip_key_net not in pre_keys:
                        logging.info('op: %s not exist in pretrain, ignore'%(key_net))
                        new_dict[key_net] = self.net.state_dict()[key_net]
                        continue
                    else:
                        net_shape = str(self.net.state_dict()[key_net].shape).replace('torch.Size', '')
                        pre_shape = str(pretrained_model[strip_key_net].shape).replace('torch.Size', '')
                        if self.net.state_dict()[key_net].shape != pretrained_model[strip_key_net].shape:
                            logging.info('op: %s exist in pretrain but shape difficenet(%s:%s), ignore' % (key_net,net_shape,pre_shape))
                            new_dict[key_net] = self.net.state_dict()[key_net]
                        else:
                            logging.info('op: %s exist in pretrain and shape same(%s:%s), load' % (key_net, net_shape, pre_shape))
                            new_dict[key_net] = pretrained_model[strip_key_net]

            else:
                for key_pre, key_net in zip(pretrained_model.keys(), self.net.state_dict().keys()):
                    if self.net.state_dict()[key_net].shape == pretrained_model[key_pre].shape:
                        new_dict[key_net] = pretrained_model[key_pre]
                        logging.info('op: %s shape same, load weights'%(key_net))
                    else:
                        new_dict[key_net] = self.net.state_dict()[key_net]
                        logging.info('op: %s:%s shape diffient(%s:%s), ignore weights'%
                              (key_net, key_pre,
                               str(self.net.state_dict()[key_net].shape).replace('torch.Size', ''),
                               str(pretrained_model[key_pre].shape).replace('torch.Size', '')))

            self.net.load_state_dict(new_dict, strict=False)

    def proj_init(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--local_rank', type=int, default=0)
        args = parser.parse_args()
        self.local_rank = args.local_rank
        self.world_size = int(os.getenv('WORLD_SIZE', 1))

        # create train dir
        self.proj_dir = self.train_dict['save_dir']
        if self.proj_dir == '':
            cfg_name = os.path.splitext(os.path.basename(self.cfg_file))[0]
            time_name = time.strftime("%Y%m%d%H%M%S", time.localtime())
            self.proj_dir = './save/' + cfg_name + '-' + time_name
            self.train_dict['save_dir'] = self.proj_dir

        if self.local_rank == 0 and not os.path.exists(self.proj_dir):
            os.makedirs(self.proj_dir)
            shutil.copy(self.cfg_file, self.proj_dir)

        # get the device and set device
        self.device = self.train_dict['device']
        assert(self.device == 'cuda' or self.device == 'cpu')
        if self.device == 'cuda':
            torch.cuda.set_device(self.local_rank)

        # init log
        while not os.path.exists(self.proj_dir):
            time.sleep(1)
        logging.basicConfig(level=logging.DEBUG,
                            filename=self.proj_dir + '/log.txt',
                            filemode='w',
                            format='%(asctime)s %(levelname)s: %(message)s')
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

        # print info
        logging.info('proj_init: device=%s' % self.device)
        logging.info('proj_init: rank=%d;world_size=%d' % (self.local_rank, self.world_size))
        logging.info('proj_init: cfg_file=%s' % (self.cfg_file))

        # print cfg file
        if self.local_rank == 0:
            with open(self.cfg_file, 'r') as f:
                for line in f.readlines():
                    logging.info(line.strip())
                    
    def train_init(self):
        # 1. create dataloader
        logging.info('train_init: create dataloader')
        self.train_loader, self.train_sampler = create_train_dataloader(self.train_dataset,
                                                                        self.data_dict['train'],
                                                                        self.local_rank,
                                                                        self.world_size)
        if self.eval_enable:
            self.eval_dataloader = create_val_dataloader(self.val_dataset,
                                                          self.data_dict['eval'])
        else:
            self.eval_dataloader = None
 
        # 2. distributed the net
        logging.info('train_init: distributed the net')
        self.net.to(self.device)
        if self.world_size > 1:
            assert(self.device == "cuda")
            torch.distributed.init_process_group(backend="nccl",
                                                 world_size=self.world_size,
                                                 init_method='env://')
            torch.distributed.barrier()
            self.net = torch.nn.parallel.DistributedDataParallel(self.net,
                                                                 find_unused_parameters=True,
                                                                 device_ids=[self.local_rank],
                                                                 output_device=self.local_rank)
        # 3. load net pre model
        logging.info('train_init: load pretrain model')
        self.load_pre_train_ignore_name()

        # 4. create opt
        logging.info('train_init: create optimizer')
        self.opt = create_optimizer(self.net, self.opt_dict)

        # 5. create lr schedelu
        logging.info('train_init: create lr_schedule')
        self.lr_schedule = LrSchedule(self.opt, self.opt_dict)

        # 6. create the trainer
        logging.info('train_init: create trainer')
        self.trainer = Trainer(self.proj_dir,
                               self.opt,
                               self.lr_schedule,
                               self.net,
                               [self.train_loader, self.train_sampler,self.eval_dataloader],
                               self.train_dict,
                               self.device,
                               self.local_rank,
                               self.world_size)
    def train(self):
        self.trainer.run()
