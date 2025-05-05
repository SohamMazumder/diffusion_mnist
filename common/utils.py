""" Various helpers. """
import pickle

import copy
import subprocess
import time
import glob
import hashlib
import importlib
import os
import shutil
import socket
import contextlib
import csv
import json
import traceback
import types
import warnings

import matplotlib
import matplotlib.pyplot
import yaml

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image
from PIL.PngImagePlugin import PngInfo


CHECKPOINT_FILE_SUFFIXES = {
    'model': '-model.pt',
    'training': '-training.pt',
}

def make_red_green_image(image):
    """
    Convert a 1 channel image of positive and negative numbers into a color image
    for visualization. Red corresponds to negative numbers, green to positive ones.
    :param image: one channel image (H, W) or (H, W, 1).
    :return: a 3-channel image in RGB format.
    """
    if image.ndim == 3:
        image = np.squeeze(image, axis=2)
    pos = (image >= 0) * image
    neg = (image <= 0) * image
    blue = np.zeros(image.shape, dtype=image.dtype)
    rg = np.stack([-neg, pos, blue], axis=-1)
    return rg


@torch.no_grad()
def visualize_sign(t, dim=1, rgb_map=(-1, 0, 1)):
    """
    Convert a tensor of positive and negative numbers into a color image for visualization.
    :param t: tensor (B, 1, H, W)
    :param dim: the data dimension, typically 1.
    :param rgb_map: map positive and negative values to RGB channels.
    :return: tensor (B, 3, H, W) in RGB format.
    """
    if t.shape[dim] != 1:
        raise ValueError(f'Size of dimension {dim} must be 1, actual {t.shape[dim]}')
    channels = [0, 0, 0]
    for i in range(len(rgb_map)):
        if rgb_map[i] < 0:
            channels[i] = -t * (t <= 0)
        elif rgb_map[i] > 0:
            channels[i] = t * (t > 0)
        else:
            channels[i] = torch.zeros_like(t)
    result = torch.cat(channels, dim=dim)
    return result


@torch.no_grad()
def visualize_flow(flow, magnitude_as='v', as_tensor=True, scale=1, clamp=None):
    """
    Create an image visualizing 2D flow using the HSV colorwheel:
    https://vision.middlebury.edu/flow/floweval-ijcv2011.pdf

    :param flow: a tensor (B, H, W, 2). It is up to the caller to normalize it if desired.
    :param magnitude_as: show magnitude as
     * 'v': HSV value, small values are dark
     * 's': HSV saturation, small values are bright
    :param as_tensor: if True, return a tensor (B, 3, H, W). Otherwise, return a numpy array (B, H, W, 3).
    :return: flow image or tensor.
    """

    if magnitude_as not in ('s', 'v'):
        raise ValueError('magnitude_as must be s or v')

    magnitude = torch.sum(flow**2, dim=-1).sqrt()
    angle = (torch.atan2(-flow[..., 1], -flow[..., 0]) + torch.pi) / (2 * torch.pi)

    ones = torch.ones_like(angle)
    if magnitude_as == 's':
        hsv = torch.stack((angle, magnitude, ones), -1)
    else:
        hsv = torch.stack((angle, ones, magnitude), -1)

    rgb = matplotlib.colors.hsv_to_rgb(hsv.cpu().numpy()) * scale

    if clamp is not None:
        rgb = np.clip(rgb, a_min=None, a_max=clamp)

    if as_tensor:
        rgb = torch.tensor(rgb, dtype=flow.dtype, device=flow.device).movedim(-1, -3)

    return rgb


def visualize_by_colormap(value, min_value=None, max_value=None, scale='lin', colormap='viridis'):
    """
    Visualize a value using a colormap.
    :param value: a numpy array or a tensor.
    :param min_value: minimum value.
    :param max_value: maximum value.
    :param scale: log or lin.
    :param colormap: matplotlib colormap name.
    :return: a numpy array of shape (..., 3) in RGB format.
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if min_value is None:
        min_value = 0 if scale == 'lin' else 1e-8
    if max_value is None:
        max_value = 1 if scale == 'lin' else 1
    value = np.clip(value, min_value, max_value)
    if scale == 'log':
        if min_value <= 0 or max_value <= 0:
            raise ValueError('min_value and max_value must be positive for log scale')
        value = np.log(value)
        min_value = np.log(min_value)
        max_value = np.log(max_value)

    value = (value - min_value) / (max_value - min_value)

    colormap = matplotlib.pyplot.get_cmap(colormap)
    mapped = colormap(value)[..., :3].astype(np.float32)
    return mapped


def tensor_to_images(t, vis_sign=None):
    """
    Convert a tensor to images for visualization.
    :param t: a tensor (B, C, H, W) or (C, H, W)
    :param vis_sign: if C == 1, uses visualize_sign(). None: autodetect.
    :return: a list of HWC, 3 channel images.
    """
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().numpy()

    if t.ndim < 3 or t.ndim > 4:
        raise ValueError('Unsupported tensor shape')

    if t.ndim == 3:
        t = np.expand_dims(t, 0)

    t = np.ascontiguousarray(np.moveaxis(t, 1, -1))
    result = []
    for i in range(len(t)):
        image = t[i]
        if image.shape[2] == 1:
            image = np.squeeze(image)
        if image.ndim == 2:
            if vis_sign is None and image.min() < 0:
                vis_sign = True
            if vis_sign:
                image = visualize_sign(torch.from_numpy(image)).numpy()
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        result.append(image)
    return result


def show_tensor(name, t, red_green=None, wait_key_time=1):
    """
    Show  tensor in a window.
    :param name: window name
    :param t: a tensor (B, C, H, W) or (C, H, W)
    :param red_green: if C == 1, show negative values in red and positive in green. None: autodetect.
    :param wait_key_time: a time in ms to wait for a keypress. 0 - wait forever.
    :return:
    """
    images = tensor_to_images(t, red_green)
    for i, image in enumerate(images):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow(name+f'-{i}', image)
    cv2.waitKey(wait_key_time)


def set_device(default=None):
    warnings.warn('utils.set_device() is deprecated, use world.set_device() instead', DeprecationWarning)
    from common import world
    world.set_device(default)

def get_checkpoint_file(checkpoint_path):
    """
    Get the checkpoint file from a file or directory path.
    :param checkpoint_path: path to the checkpoint file. If it is a directory, return the last checkpoint.
    """
    if os.path.isdir(checkpoint_path):
        checkpoints = glob.glob(checkpoint_path + '/*.pt')
        if not checkpoints:
            raise ValueError(f'No checkpoints in {checkpoint_path}')
        checkpoints.sort()
        checkpoint_file = os.path.normpath(checkpoints[-1])
    elif os.path.isfile(checkpoint_path):
        checkpoint_file = checkpoint_path
    else:
        raise ValueError(f'Wrong checkpoint path {checkpoint_path}')

    return checkpoint_file


def load_config(config_file=None, checkpoint_file=None):
    """
    Loads configuration from a config or a checkpoint.
    If only the config_file is specified, no checkpoint will be loaded.
    If only the checkpoint_file is specified, the config is loaded from the checkpoint.
    If both are specified, the config from the config_file is taken.

    :param config_file: path to the config file.
    :param checkpoint_file: path to the checkpoint file. If it is a directory, load the last checkpoint.
    :return config dictionary.
    """

    if config_file:
        print(f'Loading config from {config_file}')
        with open(config_file, encoding='utf-8') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    elif checkpoint_file:
        checkpoint_file = get_checkpoint_file(checkpoint_file)
        print(f'Loading checkpoint {checkpoint_file}')
        checkpoint = torch.load(checkpoint_file)
        print(f'Loading config from checkpoint {checkpoint_file}')
        config = checkpoint['config']
    else:
        raise ValueError('At least one of the config_file and checkpoint_file must be specified')

    return config


def load_checkpoint_old(checkpoint_file, map_location=None):
    """
    Loads a checkpoint.
    :param checkpoint_file: path to the checkpoint file. If it is a directory, load the last checkpoint.
    :param map_location: see torch.load().
    :return A tuple (epoch, training_time, checkpoint).
    """

    print('load_checkpoint_old() is deprecated and will be removed, use load_checkpoint() instead.')

    checkpoint_file = get_checkpoint_file(checkpoint_file)

    print(f'Loading checkpoint {checkpoint_file}')
    checkpoint = torch.load(checkpoint_file, map_location=map_location)
    epoch = checkpoint['epoch']
    training_time = checkpoint['training_time'] if 'training_time' in checkpoint else None

    # TOOD(IA): remove it when the next model (1.9) is trained ('time' is the old name for 'training_time')
    if training_time is None and 'time' in checkpoint:
        training_time = checkpoint['time']

    return epoch, training_time, checkpoint


def load_module(module, state_dict, key_prefix=None, strict=True):
    """
    Loads a module.
    :param module: module.
    :param state_dict: state dictionary.
    :param key_prefix: dictionary key prefix to remove, in case this module is an attribute of a parent module.
    :param strict True or False - see load_state_dict(), 'auto' - try load strict,
    if this does not work, try non-strict.
    """
    if key_prefix is not None:
        state_dict_orig = state_dict
        state_dict = {}
        for k, v in state_dict_orig.items():
            if k.startswith(key_prefix + '.'):
                new_key = k[len(key_prefix + '.'):]
                state_dict[new_key] = v
    strict_value = strict if strict != 'auto' else True
    try:
        module.load_state_dict(state_dict, strict=strict_value)
    except RuntimeError as e:
        if strict != 'auto':
            raise
        traceback.print_exception(e)
        print('Cannot load complete checkpoint (see exception above), try to load what is possible')
        module.load_state_dict(state_dict, strict=False)


def save_checkpoint_old(path, epoch, training_time, config, modules, **kwargs):
    """
    Save a checkpoint at the end of the epoch.
    :param path: path to checkpoint file.
    :param epoch: finished epoch.
    :param training_time: training time (number of training examples).
    :param config: configuration.
    :param modules: an iterable of modules (e.g. models, optimizers) to save.
    :param kwargs: additional data to save in the checkpoint dictionary.
    """
    print('save_checkpoint_old() is deprecated and will be removed, use save_checkpoint() instead.')

    data = {
        'epoch': epoch,
        'training_time': training_time,
        'config': config,
        'modules': [m.state_dict() for m in modules]
    }
    for k, v in kwargs.items():
        data[k] = v
    torch.save(data, path)


def save_checkpoint(file_name, num_samples, config, model_modules, training_modules, **kwargs):
    """
    Saves a checkpoint. Creates 2 files, one with the main model, the other with the other checkpoint data.
    Rationale: the model file is downloaded from the training server, and many of them are stored for longer,
    so the size matters.
    :param file_name: file name.
    :param num_samples: number of training examples seen so far.
    :param config: configuration.
    :param model_modules: modules related to the main model - the model itself and optionally others.
    :param training_modules: modules related to the training - optimizers, discriminators, etc.
    :param kwargs: additional data to save in the checkpoint dictionary.
    """
    base_name, suffix = split_checkpoint_file_name(file_name)
    if suffix not in list(CHECKPOINT_FILE_SUFFIXES.values()):
        raise ValueError(f'Unexpected checkpoint file suffix in {file_name}')

    training_file = base_name + CHECKPOINT_FILE_SUFFIXES['training']
    model_file = base_name + CHECKPOINT_FILE_SUFFIXES['model']
    model_data = {
        'num_samples': num_samples,
        'config': config,
        'model_modules': [m.state_dict() for m in model_modules]
    }
    print(f'Saving model to {model_file}, training data to {training_file} ...', end='')
    torch.save(model_data, model_file)

    training_data = {
        'training_modules': [m.state_dict() for m in training_modules]
    }
    for k, v in kwargs.items():
        training_data[k] = v
    torch.save(training_data, training_file)
    print(' done.')

    return model_file, training_file


def split_checkpoint_file_name(file_name, raise_if_unknown_format=True):
    for s in CHECKPOINT_FILE_SUFFIXES.values():
        if file_name.endswith(s):
            return file_name[:-len(s)], s
    if raise_if_unknown_format:
        raise ValueError(f'Unknown format for checkpoint name: {file_name}')
    return file_name, ''


def load_checkpoint(checkpoint_path, map_location=None):
    """
    Loads a checkpoint. A checkpoint consists of 2 files: model (mandatory) and training (optional). You can
    specify any, the other will be automatically detected.
    See also save_checkpoint.
    :param checkpoint_path: path to the checkpoint file. If it is a directory, load the last checkpoint.
    :param map_location: see torch.load().
    :return checkpoint dictionary.
    """
    file_name = get_checkpoint_file(checkpoint_path)
    base_name, suffix = split_checkpoint_file_name(file_name, False)
    if not suffix:
        model_file = file_name
        training_file = None
    elif suffix == CHECKPOINT_FILE_SUFFIXES['training']:
        training_file = file_name
        model_file = base_name + CHECKPOINT_FILE_SUFFIXES['model']
    elif suffix == CHECKPOINT_FILE_SUFFIXES['model']:
        model_file = file_name
        training_file = base_name + CHECKPOINT_FILE_SUFFIXES['training']

    if model_file is None:
        raise ValueError(f'Cannot find model file in {checkpoint_path}')

    data = {}
    print(f'Loading model from {model_file}')
    data.update(torch.load(model_file, map_location=map_location))

    # Temp code to convert old checkpoints. TODO(ia): delete.
    # data['model_modules'] = data['modules'][:1]
    # del data['modules']
    # torch.save(data, 'new.pt')

    if training_file is not None:
        if os.path.isfile(training_file):
            print(f'Loading training data from {training_file}')
            data.update(torch.load(training_file, map_location=map_location))
        elif file_name == training_file:
            # Training file was specified, but not present:
            raise RuntimeError(f'File {file_name} does not exist.')

    return data


def create_object(*args, **kwargs):
    """
    Create an object or a function specified by class name and arguments.
    :param args: arguments to pass to the object constructor.
    :param kwargs: keyword arguments to pass to the object constructor.
    One of the kwargs must be class_name, containing one of the following:
    - a string like 'package.subpackage.module.Class'
    - a tuple ('package.subpackage.module', 'Class.Subclass').
    :return: object instance.
    """
    if 'class_name' not in kwargs:
        raise ValueError('Missing class_name keyword argument.')
    kwargs = copy.copy(kwargs)
    class_name = kwargs['class_name']
    del kwargs['class_name']

    if isinstance(class_name, str):
        parts = class_name.split('.')
        module_name = '.'.join(parts[:-1])
        class_name = parts[-1]
    else:
        module_name = class_name[0]
        class_name = class_name[1]

    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    if isinstance(cls, (types.FunctionType, types.BuiltinFunctionType)):
        # The function 'class' or type is the object.
        if args or kwargs:
            raise ValueError('Function does not accept arguments.')
        obj = cls
    else:
        # Create the object.
        obj = cls(*args, **kwargs)
    return obj


def get_parameters(obj):
    """
    Returns a generator to iterate over parameters in an optimizer or a model.
    :param obj: optimizer or module.
    :return: generator object to enumerate parameters.
    """
    if isinstance(obj, torch.optim.Optimizer):
        for group in obj.param_groups:
            for p in group['params']:
                yield p
    else:
        for p in obj.parameters():
            yield p


def count_parameters(obj):
    """
    Count parameters in the optimizer or model.
    :param obj: optimizer or module.
    :return: the number of parameters.
    """
    return sum(p.numel() for p in get_parameters(obj))


def compute_parameter_checksum(obj):
    """
    Compute a simple checksum of all parameters in the optimizer or model.
    :param obj: optimizer or module.
    :return: the sum of all parameters.
    """
    return sum(p.abs().sum() for p in get_parameters(obj))


def max_abs_param(module):
    """
    Max of abs of the parameters of a module.
    :param module: module.
    :return: the max abs value.
    """
    m = 0
    for p in module.parameters():
        m = max(m, p.abs().max().item())
    return m


def draw_axes(image, r, tx, ty, length=100, width=1, colors=None):
    """
    Draws axes of the standard right-handed CS: x, y as on the screen, z axis looking away from the observer.
    :param image: image
    :param r: rotation matrix for post-multiplication.
    :param tx: translation x in pixels
    :param ty: translation y in pixels
    :param length: axis length as a number or a tuple
    :param colors: color for axes x, y, z.
    :param width: line width.
    """
    if type(length) in (int, float):
        length = (length,) * 3
    axes = np.array(((1, 0, 0), (0, 1, 0), (0, 0, 1)), dtype=np.float32)
    origin = np.array((tx, ty, 0), dtype=np.float32)
    axes = np.dot(axes, r) * np.array(length).reshape(3, 1) + origin

    if colors is None:
        colors = ((0, 0, 255), (0, 255, 0), (255, 0, 0)) \
            if image.dtype == np.uint8 else ((0, 0, 1), (0, 1, 0), (1, 0, 0))

    o = tuple(origin[:2].astype(int))
    # Draw z-axis last.
    for ai in range(3):
        a = tuple(np.round(axes[ai, :2]).astype(int))
        cv2.line(image, o, a, colors[ai], width)

    return image


def draw_direction_vector(image, vector, tx=None, ty=None, color=(1, 1, 0), length=100, width=1):
    """
    Draws a direction vector on an image.
    :param image: image as numpy array.
    :param tx: translation x in pixels
    :param ty: translation y in pixels
    :param length: scaling factor
    :param color: color
    :param width: line width.
    """
    vector = vector * length
    tx = image.shape[1] // 2 if tx is None else int(tx)
    ty = image.shape[0] // 2 if ty is None else int(ty)
    start = tx, ty
    end = int(start[0] + vector[0]), int(start[1] + vector[1])
    cv2.arrowedLine(image, start, end, color, width)
    # Draw start of the line with a different color, as the direction of the arrow is not clear for short vectors.
    start_color = tuple(c * 0.90 for c in color)
    cv2.line(image, start, start, start_color, width)


class VideoReader:
    """ Read video from a video file, camera, pictures, etc. """
    def __init__(self, video_path, roi=None, loop=False, flip=False):
        """
        Create object
        :param video_path: path to a video. Pass an integer number to open a camera by its ID.
        """
        self._video = None
        self._frames = None
        self._frame_index = 0
        self._loop = loop
        self._roi = roi
        self._video_path = video_path
        self._flip = flip
        if os.path.isdir(video_path):
            self._frames = glob.glob(video_path + '/**/*.png', recursive=True)
        elif os.path.splitext(video_path)[1].lower() in ['.png', '.jpg', '.jpeg']:
            self._frames = [video_path]
        else:
            if self._video_path.isnumeric():
                self._video_path = int(self._video_path)  # Camera id
                self._loop = False
            print(f'Opening video capture {self._video_path} ...')
            self._video = cv2.VideoCapture(self._video_path)
            print(f'Opened video capture {self._video_path}')

    @property
    def frame_index(self):
        """ Returns the index of the next frame returned by read_frame(). In loop mode will jump down skipping 0. """
        return self._frame_index

    def read_frame(self):
        """
        Read a frame.
        :return: an image in BGR format, or None in case of end of file, etc.
        """
        if self._frames is not None:
            if self._frame_index >= len(self._frames):
                if not self._loop:
                    return None
                self._frame_index = 0
            frame_path = self._frames[self._frame_index]
            self._frame_index += 1
            frame = cv2.imread(frame_path)
        else:
            ret, frame = self._video.read()
            self._frame_index += 1
            if not ret and self._loop:
                self._video = cv2.VideoCapture(self._video_path)
                ret, frame = self._video.read()
                self._frame_index = 1
            if not ret:
                frame = None

        if self._flip:
            frame = np.ascontiguousarray(np.flip(frame, axis=1))

        if self._roi is not None:
            frame = frame[self._roi[1]:self._roi[1]+self._roi[3], self._roi[0]:self._roi[0]+self._roi[2]]

        return frame


def rand_uniform(b, e, shape, device):
    """ Generate uniform random numbers in range [b, e). """
    return torch.rand(shape, device=device) * (e - b) + b


def interpolate(inp, size=None, scale_factor=None, mode='bilinear', align_corners=False, antialias=False):
    """
    A wrapper for torch.functional.interpolate() with convenient default parameter settings
    without generating warnings.
    """
    if size == inp.shape[-2:] and scale_factor is None:
        return inp
    if scale_factor == 1 and size is None:
        return inp

    if mode in ('nearest', 'area'):
        align_corners = None
    if size is None:
        recompute_scale_factor = False
    else:
        recompute_scale_factor = None
    return F.interpolate(inp, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners,
                         recompute_scale_factor=recompute_scale_factor, antialias=antialias)


def almost_equal(x, y, eps=1e-5):
    return (x - y).abs().max() < eps


def make_clean_directory(path):
    """
    Creates an empty directory.

    If it exists, delete its content.
    If the directory is opened in Windows Explorer, may throw PermissionError,
    although the directory is usually cleaned. The caller may catch this exception to avoid program termination.
    :param path: path to the directory.
    """
    need_create = True
    if os.path.isdir(path):
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        need_create = False
    elif os.path.isfile(path):
        os.remove(path)
    if need_create:
        os.makedirs(path)


class PerformanceChecker:
    """ Checks the performance of the code.  """
    def __init__(self, name='unnamed', warmups=10, log_interval=100, log_format='time'):
        """
        Create object
        :param name: a name used for printing
        :param warmups: an integer >= 0. if > 0, this number of initial measurements will be discarded.
        :param print_interval: if > 0, log result every print_interval number of measurements.
        :param log_format: one of 'time', 'fps'
        """
        self._log_interval = log_interval
        self.total_run_time = 0
        self.count = -warmups
        self._start_time = None
        self._name = name
        self._log_format = log_format

    def start(self):
        self._start_time = time.perf_counter_ns()

    def stop(self):
        self.count += 1
        if self.count <= 0:
            return
        run_time = time.perf_counter_ns() - self._start_time
        self.total_run_time += run_time
        if self._log_interval > 0 and self.count % self._log_interval == 0:
            self.log()

    def log(self):
        if self._log_format == 'fps':
            print(f'{self._name}: {self.count / self.total_run_time * 1e9:.1f} FPS')
        else:
            rt = self.total_run_time / 1e6
            print(f'{self._name}: run time total: {rt} ms, average: {rt / self.count} ms')


def range_1_2(x):
    """
    Convert from input range [0, 1] to [-1, 1].
    """
    return x * 2 - 1


def range_2_1(x):
    """
    Convert from input range [-1, 1] to [0, 1].
    """
    return x * 0.5 + 0.5


def range_255_2(x):
    """
    Convert from input range [0, 255] to [-1, 1].
    """
    return x * (1 / 127.5) - 1


def range_2_255(x):
    """
    Convert from input range [-1, 1] to [0, 255].
    """
    return (x + 1) * 127.5


def range_255_1(x):
    """
    Convert from input range [0, 255] to [0, 1].
    """
    return x * (1 / 255)


def range_1_255(x):
    """
    Convert from input range [0, 1] to [0, 255].
    """
    return x * 255


def range_mean_std_1_2(mean, std):
    """
    Convert mean and std from input range [0, 1] to [-1, 1].
    They can be used for image normalization as usual n = (x - mean) / std.
    """
    if not isinstance(mean, torch.Tensor):
        mean = torch.tensor(mean)
    if not isinstance(std, torch.Tensor):
        std = torch.tensor(std)
    mean2 = 2 * mean - 1
    std2 = 2 * std
    return mean2, std2


def get_sha256(filename):
    sha256 = hashlib.sha256()
    with open(filename, 'rb') as f:
        while True:
            data = f.read(4096)
            if not data:
                break
            sha256.update(data)
    return sha256


def read_images(paths, convert_fn=range_255_2, dtype=np.float32,
                as_tensor=True, rgb=True, as_batch=True):
    """
    Read image or images from files with some conversion operations.
    This function encapsulates the typical operations, repeatedly used in experiments.
    It is for convenience, not for speed.

    :param paths: a path, a glob patter or a list of paths.
    :param convert_fn: a function to convert the image.
    :param as_tensor: convert to torch.tensor if True.
    :param rgb: convert to RGB if True.
    :param dtype: datatype to convert to.
    :param as_batch: stack all images to a batch if True, otherwise return a list of images.
    :return: images, paths
    """
    if not isinstance(paths, (list, tuple)):
        paths = glob.glob(paths, recursive=True)
        paths.sort()  # Ensure reproducibility
    images = []
    for image_path in paths:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f'Cannot read image {image_path}')
        if dtype is not None:
            image = image.astype(dtype)
        if convert_fn is not None:
            image = convert_fn(image)
        if rgb:
            image = np.ascontiguousarray(image[..., ::-1])
        if as_tensor:
            image = torch.tensor(rearrange(image, 'h w c -> c h w')).contiguous()
        images.append(image)

    if as_batch:
        if as_tensor:
            images = torch.stack(images)
        else:
            images = np.stack(images)

    return images, paths


def find_free_port():
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def is_port_free(port):
    # See https://stackoverflow.com/questions/2470971/fast-way-to-test-if-a-port-is-in-use-using-python
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) != 0


def check_shape(t, s):
    """ Checks that a tensor has a given shape.
    :param t: tensor
    :param s: shape. None means this dimension can vary.
    """
    if t.ndim != len(s):
        raise ValueError(f'Wrong number of dimensions: got {t.ndim}, expected {len(s)}')

    for i, (t_size, s_size) in enumerate(zip(t.shape, s)):
        if s_size is None:
            continue
        if t_size != s_size:
            raise ValueError(f'Wrong size for dimension {i}: got {t_size}, expected {s_size}')


def find_paths(root_dir, filter_fn=None, sort_fn=None):
    """
    Find paths in the given root directory, filtered and sorted by criteria.
    For example:

    find_paths('mydir', filter_fn=os.path.isdir, sort_fn=os.path.getmtime)

    returns a list of directories, sorted by modification time.

    :param root_dir: root dir to search.
    :param filter_fn: filter criterion.
    :param sort_fn: sort criterion.
    :return: a list of paths.
    """
    filtered = []
    for p in os.listdir(root_dir):
        p = os.path.join(root_dir, p)
        if not filter_fn or filter_fn(p):
            filtered.append(p)
    if sort_fn:
        filtered.sort(key=sort_fn)
    return filtered


@torch.no_grad()
def visualize_tensor(t):
    if t.ndim != 4:
        raise ValueError('A 4D tensor is expected')
    ch = t.shape[1] // 3
    r = t[:, :ch]
    g = t[:, ch:2 * ch]
    b = t[:, 2 * ch:]
    rgb = []
    for c in (r, g, b):
        c = torch.linalg.norm(c, dim=1, keepdim=True)
        rgb.append(c)
    rgb = torch.concat(rgb, dim=1)
    m = rgb.amax(dim=(1, 2, 3), keepdim=True)
    rgb = rgb / (m + 1e-8)
    return rgb


@torch.no_grad()
def visualize_pyramid(pyramid, alignment=0):
    """
    Create an image showing all images in a pyramid.
    :param pyramid: a list of images [B, C, S, S] with pyramidal sizes, e.g. 256, 128, 64, 32.
    :param alignment: 0 or 1, controls the side the images stick to.
    :return: an image with a pyramid.
    """
    if not pyramid:
        return None

    b, c = pyramid[0].shape[:2]
    for img in pyramid:
        bi, ci, h, w = img.shape
        if bi != b:
            raise ValueError('All images must have the same batch size')
        if ci != c:
            raise ValueError('All images must have the same number of channels')

    pyramid = list(pyramid)  # Prevent original list from modification
    pyramid.sort(key=lambda x: -x.shape[-1])

    result = pyramid[0]
    if len(pyramid) == 1:
        return result

    height = sum(p.shape[2] for p in pyramid[1:])
    rest = torch.zeros(*result.shape[:2], height, result.shape[3])
    p = 0
    for i in range(1, len(pyramid)):
        img = pyramid[i]
        if alignment == 0:
            start = 0
            end = img.shape[3]
        else:
            start = -img.shape[3]
            end = rest.shape[3]
        rest[:, :, p:p+img.shape[2], start:end] = img
        p += img.shape[2]

    result = torch.cat((result, rest), dim=2)
    return result


@torch.no_grad()
def visualize_features(features, image=None):
    """
    Visualize features.
    :param features: feature tensor, list or dict.
    :param image: image or a list of images.
    :return: image containing visualized features.
    """

    text_params = {
        'fontFace': cv2.FONT_HERSHEY_SIMPLEX,
        'fontScale': 0.75,
        'thickness': 1,
        'color': (0, 1, 0)
    }
    # Convert any feature input to a dictionary.
    if isinstance(features, torch.Tensor):
        features = { f'{features.shape[-1]}': features }
    elif isinstance(features, list):
        features = { f'{i}-{f.shape[-1]}': f for i, f in enumerate(features) }

    if isinstance(image, torch.Tensor):
        image = (image,)

    if image is not None:
        shape = image[0].shape[-2:]
    else:
        shape = (
            max(f.shape[-2] for f in features),
            max(f.shape[-1] for f in features),
        )

    vis = []
    for key, f in features.items():
        fv = visualize_tensor(f)
        fv = fv.clamp(0, 1)
        fv = interpolate(fv, size=shape, mode='nearest')
        fv = fv.movedim(1, -1).cpu().numpy()
        fv = np.ascontiguousarray(fv)
        for i in range(len(fv)):
            cv2.putText(fv[i], f'{key}', (2, 30), **text_params)
        vis.append(fv)

    if image is not None:
        image = [im.expand(-1, 3, -1, -1) if im.shape[1] == 1 else im for im in image]
        image = torch.cat(image, 2)
        vis.insert(0, (range_2_1(image).clamp(0, 1).movedim(1, -1).cpu().numpy()))
    vis = np.concatenate(vis, 1)
    return vis


def read_csv_annotations(file_name, delimeter='|'):
    with open(file_name, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=delimeter)
        for ri, row in enumerate(reader):
            if ri == 0:
                anno = {k: [] for k in row}
            else:
                for i, k in enumerate(anno):
                    anno[k].append(row[i])

    for k in anno:
        anno[k] = np.array(json.loads('[' + ','.join(anno[k]) + ']'), dtype=np.float32)

    return anno


class Eval:
    """ Switch model to eval, preserving original training state after exit. """
    def __init__(self, model):
        self._model = model
        self._training = None

    def __enter__(self):
        self._training = self._model.training
        if self._training:
            self._model.train(False)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        assert self._training is not None
        assert not self._model.training
        if self._training:
            self._model.train(True)


class AttrDict(dict):
    """ A dictionary that allows access by attributes: d['key] == d.key.
        See also: https://github.com/makinacorpus/easydict
    """

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        del self[key]


def make_valid_filename(name):
    """ Replaces characters that cannot appear in a file name. """
    invalid_chars = '\\/:*?<>|,'
    t = str.maketrans(invalid_chars, '_' * len(invalid_chars))
    return name.translate(t)


def print_git_commit_id():
    """ Prints the current git commit id. The code can be browsed under <REPO_URL>/tree/<COMMIT_ID>
    or <REPO_URL>/commit/<COMMIT_ID>.
    """
    command = ['git', 'rev-parse', 'HEAD']
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)

    if result.returncode == 0:
        commit_id = result.stdout.strip()
        print('Git commit id:', commit_id)
    else:
        error_message = result.stderr.strip()
        print('Error:', error_message)


def apply_to_collection(coll, func):
    """
    Apply a function to all elements of a collection. For example, to transfer all tensors in a dictionary to the GPU,
    call `apply_to_collection(some_dict, lambda x: x.cuda())`.
    :param coll: collection of objects. A collection can contain nested collections (e.g. lists of tuples),
    the function will be applied to all elements.
    :param func: a function to apply, it shall take one parameter and return a transformed object.
    :return: the updated collection.
    """
    if isinstance(coll, list):
        return [apply_to_collection(v, func) for v in coll]
    elif isinstance(coll, tuple):
        return tuple(apply_to_collection(v, func) for v in coll)
    elif isinstance(coll, dict):
        return {k: apply_to_collection(v, func) for k, v in coll.items()}
    return func(coll)


def inverse_softplus(s, beta=1):
    """ Computes inverse of softplus:
        s = softplus(x, beta)
        x = inverse_softplus(s, beta)
    :param s: a tensor with positive values.
    :return result tensor.
    """
    return torch.log(torch.exp(s * beta) - 1) / beta


class PinnedBuffer:
    """ Speed up data exchange between CPU and GPU by using pinned memory.

        Inspired by StyleGAN: https://github.com/NVlabs/stylegan3/blob/c233a919a6faee6e36a316ddd4eddababad1adf9/viz/renderer.py#L195
    """
    def __init__(self, device=None):
        if device is None:
            device = torch.tensor([0]).device
        self._device = device
        self._pinned_bufs = {}

    def _get_pinned_buf(self, ref):
        key = (tuple(ref.shape), ref.dtype)
        buf = self._pinned_bufs.get(key, None)
        if buf is None:
            buf = torch.empty(ref.shape, dtype=ref.dtype, device='cpu').pin_memory()
            self._pinned_bufs[key] = buf
        return buf

    def to_device(self, buf):
        return self._get_pinned_buf(buf).copy_(buf).to(self._device)

    def to_cpu(self, buf):
        return self._get_pinned_buf(buf).copy_(buf).clone()


def save_png_with_metadata(filename, image, metadata, compress_level=6):
    """ Save a numpy RGB image with metadata. """
    if metadata is None:
        metadata_chunk = None
    else:
        metadata_bytes = pickle.dumps(metadata)
        metadata_chunk = PngInfo()
        metadata_chunk.add(b'meta', metadata_bytes)
    pil_image = Image.fromarray(image)
    pil_image.save(filename, pnginfo=metadata_chunk, compress_level=compress_level)


def load_png_with_metadata(filename):
    """ Load a numpy RGB image with metadata. """
    image = Image.open(filename)
    image_np = np.array(image)
    metadata_chunk = None
    for chunk in image.private_chunks:
        if chunk[0] == b'meta':
            metadata_chunk = chunk
            break
    if metadata_chunk is None:
        metadata = None
    else:
        metadata = pickle.loads(metadata_chunk[1])
    return image_np, metadata


def zoom_crop(image, tx=0, ty=0, scale=1, output_size=1024,
              interpolation=cv2.INTER_CUBIC, border_mode=cv2.BORDER_REPLICATE):
    """ Crop a picture in a similar way as in the training with data augmentation.
        This is helpful to feed the model with the same kind of images as during training,
        if a model was optimized for a specific scale and position of the face on the picture.

        For the origin of the code see data/augmentation.py.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError('image should be a numpy array')
    if image.ndim != 3:
        raise ValueError('image should be HxWxC')

    def make_transform2(scale=1.0, tx=0.0, ty=0.0):
        t = np.array([scale, 0, tx, 0, scale, ty, 0, 0, 1], dtype=np.float64).reshape(3, 3)
        return t

    # Convert normalized coordinates from [-1, 1] to pixels
    n2pix = output_size / 2
    tx *= n2pix
    ty *= n2pix
    t_transl = make_transform2(tx=tx, ty=ty)

    t_scale = make_transform2(scale=scale)

    image_center = ((image.shape[1] - 1) / 2, (image.shape[0] - 1) / 2)
    # Move center of the image to (0, 0)
    t_image_center_to_origin = make_transform2(tx=-image_center[0], ty=-image_center[1])

    input_size = min(*image.shape[:2])
    t_scale_to_intermediate_resolution = make_transform2(scale=output_size / input_size)
    t_normalize = t_scale_to_intermediate_resolution @ t_image_center_to_origin

    t_origin_to_output_center = make_transform2(tx=(output_size - 1) / 2, ty=(output_size - 1) / 2)
    # The sequence of warps in the logical order from 1st to last
    t_warp = [
        t_normalize,
        t_scale,
        t_transl,
        t_origin_to_output_center  # Move (0, 0) to the center of the output image
    ]
    t_warp = np.linalg.multi_dot(t_warp[::-1])

    image = cv2.warpAffine(image, t_warp[:2, :3], (output_size,) * 2,
                           flags=interpolation, borderMode=border_mode)

    return image


def cli_callable(method):
    """Decorator to mark a method as CLI callable."""
    method._is_cli_callable_ = True
    return method

def collect_cli_methods(cls):
    """Class decorator to collect CLI callable methods."""
    cli_methods = []
    for name, method in cls.__dict__.items():
        if callable(method) and hasattr(method, '_is_cli_callable_'):
            cli_methods.append(name)
    return cli_methods