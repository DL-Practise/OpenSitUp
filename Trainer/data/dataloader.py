from DLEngine.modules.dataloader import dataset
import torch


def create_dataloader(phase, arg_dict, rank, world_size):
    data_name = arg_dict['data_name']
    dataset_ = dataset.__dict__[data_name](phase, arg_dict)
    if phase == 'train':
        sampler = torch.utils.data.distributed.DistributedSampler(dataset_,
                                                              num_replicas=world_size,
                                                              rank=rank)
        dataloader_ = torch.utils.data.DataLoader(dataset_,
                                              batch_size = arg_dict['batch_size'],
                                              num_workers = arg_dict['num_workers'],
                                              collate_fn = dataset_.collate_fn,
                                              sampler=sampler)
    else:
        sampler = None
        dataloader_ = torch.utils.data.DataLoader(dataset_,
                                                  batch_size=arg_dict['batch_size'],
                                                  num_workers=arg_dict['num_workers'])

    return dataloader_, sampler