import torch


def create_train_dataloader(dataset, arg_dict, rank, world_size):
    sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                              num_replicas=world_size,
                                                              rank=rank)
    dataloader_ = torch.utils.data.DataLoader(dataset,
                                              batch_size = arg_dict['batch_size'],
                                              num_workers = arg_dict['num_workers'],
                                              collate_fn = dataset.collate_fn,
                                              sampler=sampler)
    return dataloader_, sampler


def create_val_dataloader(dataset, arg_dict):
    sampler = None
    dataloader_ = torch.utils.data.DataLoader(dataset,
                                              batch_size=arg_dict['batch_size'],
                                              num_workers=arg_dict['num_workers'])
    return dataloader_
