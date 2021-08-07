import torch
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
import torch.distributed as dist
import time
import os
import logging
import shutil
from DLEngine.modules.visualize.visual_util import *
from DLEngine.modules.metric.map import calculate_map
from DLEngine.modules.metric.top1 import calculate_top1



class Trainer():
    def __init__(self, proj_dir, opt, lr_schedule, net, dataloders, train_dict, device, local_rank, world_size):
        self.pro_dir = proj_dir
        self.opt = opt
        self.lr_schedule = lr_schedule
        self.net = net
        self.train_loader = dataloders[0]
        self.train_sample = dataloders[1]
        self.eval_loader = dataloders[2]
        self.train_dict = train_dict
        self.device = device
        self.local_rank = local_rank
        self.world_size = world_size


    def run(self):
        max_epoch = self.train_dict['max_epoch']
        display_iter = self.train_dict['train_display']
        save_epoch = self.train_dict['train_save']
        eval_enable = self.train_dict['eval']['eval_enable']
        eval_start = self.train_dict['eval']['start_eval']
        eval_epoch = self.train_dict['eval']['eval_epoch']
        eval_type = self.train_dict['eval']['eval_type']
        train_batch = self.train_loader.batch_size
        self.save_dir = self.train_dict['save_dir']
        enable_visual = self.train_dict['enable_visual']
        if enable_visual and self.local_rank == 0:
            visual_init(self.pro_dir)

        self.net.train()
        iters_per_epoch = len(self.train_loader)
        max_iters = max_epoch * iters_per_epoch
        old_time = time.time()
        for epoch in range(1, max_epoch + 1):
            self.train_sample.set_epoch(epoch)
            os.environ['epoch'] = str(epoch)
            os.environ['epoch_changed'] = 'true'
            for i, (images, labels) in enumerate(self.train_loader):
                os.environ['iter'] = str(i)
                # update lr
                iter = (epoch - 1) * iters_per_epoch + i
                lr_list = self.lr_schedule.update_lr(iter, max_iters)
                # train for one batch
                images = images.to(self.device)
                labels = labels.to(self.device)
                self.opt.zero_grad()
                if self.world_size > 1:
                    loss = self.net.module.train_step(images, labels, self.local_rank)
                else:
                    loss = self.net.train_step(images, labels)

                if isinstance(loss, dict):
                    loss['total'].backward()
                else:
                    loss.backward()
                self.opt.step()

                # display interval
                if iter % display_iter == 0:
                    new_time = time.time()
                    speed = self.world_size * train_batch * display_iter / (new_time - old_time)
                    old_time = new_time
                    if self.local_rank == 0:
                        if isinstance(loss, dict):
                            loss_info = ""
                            for loss_name in loss.keys():
                                loss_info += " %s:%.5f"%(loss_name, loss[loss_name].item())
                                visual_add_scale('loss_%s'%loss_name, loss[loss_name].item(), iter)
                        else:
                            loss_info = " %.5f"%loss.item()
                            visual_add_scale('loss_total', loss.item(), iter)

                        for lr_seq, lr_info in enumerate(lr_list):
                            visual_add_scale('lr_%s'%lr_seq, lr_list[lr_seq], iter)

                        logging.info('rank %d: epoch[%d/%d] iter[%d/%d/%d] lr %s loss[%s] speed %.2f'
                                     % (self.local_rank, epoch, max_epoch, i, iters_per_epoch, iter, lr_list, loss_info, speed))

            # save the module(only rank 0)
            if epoch % save_epoch == 0 and self.local_rank == 0:
                torch.save(self.net.state_dict(), '%s/model_%d.pkl' % (self.save_dir, epoch))
                logging.info('the epoch is %d, save the snapshot' % epoch)

            # test the module(if only rank 0 test, it will block)
            # if only rank 0 test, the process will be block!
            if eval_enable and epoch % eval_epoch == 0 and epoch >= eval_start:
                self.net.eval()
                result_info = {'preds': [], 'labels': []}
                for i, (images, labels) in enumerate(self.eval_loader):
                    if self.local_rank == 0:
                        logging.info('eval batch: %d/%d' % (i + 1, len(self.eval_loader)))
                    images = images.to(self.device)
                    if self.world_size > 1:
                        preds = self.net.module.eval_step(images)
                    else:
                        preds = self.net.eval_step(images)
                    result_info['preds'].append(preds.detach().cpu())
                    result_info['labels'].append(labels)
                self.net.train()
                if eval_type == 'top1':
                    accu = calculate_top1(result_info)
                elif eval_type == 'map':
                    accu = calculate_map(result_info)
                else:
                    logging.error('unsupport eval metric:%s' % eval_type)
                    exit(0)

                if self.local_rank == 0:
                    logging.info('epoch[ %d/%d ] accu(%s) %.5f'%(epoch, max_epoch, eval_type, accu))

