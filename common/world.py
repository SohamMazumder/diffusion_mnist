""" Contains the following:
    - Information about the world (working environment).
    - A simple abstraction layer for distributed training
      to switch easily between single-GPU and distributed training.
 """
import copy
import os
from pathlib import Path
import torch
import torch.distributed as D
from torch.nn import parallel

size = 1
""" World size. """

rank = 0
""" 
    If DDP is used, the rank of the current process. Set to 'main' for the main process - the one which spawns 
    training processes. 
    If DDP is not used, set to 0.
"""

distributed = False
""" Whether distributed (multi-GPU) training is used or not. """

cache_dir = '../cache'
""" Cache directory. """

data_dir = '../blob/datasets'
""" Data directory with all datasets. See also get_data_dir()"""

training_time = 0
""" Training 'time' showing overall training progress, typically just the number of training examples seen so far. 
    Can be used to implement reproducible training schedules that are invariant to epoch size.
"""


def get_data_path(sub_path: str) -> str:
    """ Get a path to a sub-path of the data directory. If sub_path is absolute, it is returned as is.
        Usage example: to get a path to a gaze training dataset, call concat_path(world.data_dir, 'gaze/256x256/train').
    """
    return concat_paths(data_dir, sub_path)


def concat_paths(base_path: str, sub_path: str) -> str | None:
    """
    Concatenate a path from the world setting with a sub_path. Makes basic checks arguments checks to support
    typical use cases occurring when the paths are specified in a config file.

    To simplify usage, there are specific functions for common cases, such as get_data_path(). Therefore, this function
    is not intended to be used directly, but rather to be called from other functions.

    :param base_path: a path from the world setting, will be ignored if sub_path is absolute.
    :param sub_path: a relative or absolute path.
    :return: resulting path as string
    """
    if base_path is None or sub_path is None:
        return sub_path

    sub_path = Path(sub_path)
    if sub_path.is_absolute():
        return str(sub_path)

    return str(Path(base_path) / sub_path)


def init(devices, rank_):
    """ Init world, shall be called before using anything else.
        The next call shall be to diag.init() to initialize logging, to see the progress of
        subsequent operations.

        Then world.configure() can be called to make more sophisticated configuration.
    """
    global size, rank
    rank = rank_
    if devices is None:
        size = 1
    else:
        size = len(devices)
        set_visible_devices(devices)
    set_device()


def configure(free_port=None, **config):
    """
    Configure the world.
    :param free_port: if master_port is auto, use this port for all processes. It can be auto-detected in
    the main process by common.utils.find_free_port().
    :param config: configuration parameters.
    """
    global cache_dir, data_dir
    cache_dir = config.get('cache_dir', cache_dir)
    data_dir = config.get('data_dir', data_dir)

    if size > 1:
        ddp_config = copy.deepcopy(config['ddp'])
        if ddp_config['master_port'] == 'auto':
            if free_port is None:
                raise ValueError('free_port (one for all processes) shall be specified if master_port is auto')
            ddp_config['master_port'] = free_port
        ddp_setup(**ddp_config)


def set_device(default=None):
    """
    Set default device to create models and tensors on by default.
    :param default 'cpu', 'cuda:N' or None to autodetect.
    """
    if (default is None and torch.cuda.is_available()) or (default is not None and default.startswith('cuda')):
        if default is None:
            device = torch.device('cuda', 0)
        else:
            device = torch.device(default)
        torch.cuda.set_device(device)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = torch.device('cpu')
        torch.set_default_tensor_type('torch.FloatTensor')

    return device


def set_visible_devices(devices):
    # Each process will only see its GPU under id 0, do this setting before any other CUDA call.
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{devices[rank]}'
    print(f"Rank {rank} uses physical device {os.environ['CUDA_VISIBLE_DEVICES']}")


def ddp_setup(master_addr, master_port):
    if not torch.cuda.is_available():
        raise RuntimeError('Only CUDA distributed training is supported')

    # Initialize process group
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(master_port)

    backend = 'nccl' if D.is_nccl_available() else 'gloo'
    print(f'Rank {rank} uses backend {backend}')
    torch.distributed.init_process_group(backend, rank=rank, world_size=size)

    global distributed
    distributed = True


def ddp_cleanup():
    if distributed:
        torch.distributed.destroy_process_group()


class DDPWrapper(torch.nn.Module):
    """ An imitation of DistributedDataParallel for single GPU training. """
    def __init__(self, module, **kwargs):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def make_ddp_model(module, **kwargs):
    """ Create a DDP model or wrapper, depending on whether DDP is used. """
    if distributed:
        return parallel.DistributedDataParallel(module, **kwargs)
    else:
        return DDPWrapper(module, **kwargs)


def barrier():
    """ Synchronizes all processes if DDP is used, otherwise has no effect. """
    if distributed:
        D.barrier()