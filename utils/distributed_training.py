import torch.distributed as dist
import torch.multiprocessing as mp

def setup(rank, world_size):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def spawn_multiprocessor(train_fn, world_size, *args):
    mp.spawn(train_fn, 
             args=(world_size, *args),
             nprocs=world_size, 
             join=True)