# [file name]: vutils.py
import torch
import math
irange = range
import os
import random
import shutil

from concurrent.futures import ProcessPoolExecutor

def make_grid(tensor, nrow=8, padding=2,
              normalize=False, range=None, scale_each=False, pad_value=0):
    """Make a grid of images.

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The Final grid size is (B / nrow, nrow). Default is 8.
        padding (int, optional): amount of padding. Default is 2.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by subtracting the minimum and dividing by the maximum pixel value.
        range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If True, scale each image in the batch of
            images separately rather than the (min, max) over all images.
        pad_value (float, optional): Value for the padded pixels.
    """
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.view(1, tensor.size(0), tensor.size(1))
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.view(1, tensor.size(0), tensor.size(1), tensor.size(2))

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if range is not None:
            assert isinstance(range, tuple), \
                "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min + 1e-5)

        def norm_range(t, range):
            if range is not None:
                norm_ip(t, range[0], range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, range)
        else:
            norm_range(tensor, range)

    if tensor.size(0) == 1:
        return tensor.squeeze()

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    grid = tensor.new(3, height * ymaps + padding, width * xmaps + padding).fill_(pad_value)
    k = 0
    for y in irange(ymaps):
        for x in irange(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * height + padding, height - padding)\
                .narrow(2, x * width + padding, width - padding)\
                .copy_(tensor[k])
            k = k + 1
    return grid


def save_image(tensor, filename, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0):
    """Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    from PIL import Image
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    im = Image.fromarray(ndarr)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    im.save(filename)


def make_grid_forFID(tensor, nrow=8, padding=2, normalize=False,
                    range=None, scale_each=False, pad_value=0, rand=0):
    """Make a grid of images for FID calculation."""
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:
        tensor = tensor.view(1, tensor.size(0), tensor.size(1))
    if tensor.dim() == 3:
        if tensor.size(0) == 1:
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.view(1, tensor.size(0), tensor.size(1), tensor.size(2))

    if tensor.dim() == 4 and tensor.size(1) == 1:
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()
        if range is not None:
            assert isinstance(range, tuple), \
                "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min + 1e-5)

        def norm_range(t, range):
            if range is not None:
                norm_ip(t, range[0], range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:
                norm_range(t, range)
        else:
            norm_range(tensor, range)

    if tensor.size(0) == 1:
        return tensor.squeeze()

    return tensor[rand]


def save_image_forFID(tensor, nrow=8, padding=2,
                    normalize=False, range=None, scale_each=False,
                    pad_value=0, batchSize=16, counter=0, output='./'):
    """Save images for FID calculation."""
    from PIL import Image
    tensor = tensor.cpu()

    for rand in irange(batchSize):
        grid = make_grid_forFID(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                             normalize=normalize, range=range, scale_each=scale_each, rand=rand)
        ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
        im = Image.fromarray(ndarr.astype('uint8'))
        if counter < 500:
            im.save(os.path.join(output, '{}.png'.format(counter)))
        counter += 1
    return counter


def extract_nimages(read_path, write_path, num=500):
    """Extract a subset of images from a directory."""
    read_path = os.path.join(read_path, os.listdir(read_path)[0])

    if not os.path.exists(write_path):
        os.makedirs(write_path)

        filenames = os.listdir(read_path)
        random.shuffle(filenames)

        for i in range(num):
            filename = filenames[i]
            shutil.copy(os.path.join(read_path, filename), os.path.join(write_path, filename))


def init_net(net, gpu_ids=[]):
    """Initialize network with optional parallelization."""
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    return net


def save_networks(net, save_filename, gpu_ids):
    """Save network state dict."""
    if len(gpu_ids) > 0 and torch.cuda.is_available():
        torch.save(net.module.cpu().state_dict(), save_filename)
        net.cuda(gpu_ids[0])
    else:
        torch.save(net.cpu().state_dict(), save_filename)


def __patch_instance_norm_state_dict(state_dict, module, keys, i=0):
    """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)."""
    key = keys[i]
    if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
        if module.__class__.__name__.startswith('InstanceNorm'):
            if key == 'running_mean':
                if getattr(module, 'num_features', None) is None:
                    num_features = module.running_mean.size(0)
                    module.num_features = num_features
                    module.affine = True
                    module.weight = torch.ones(num_features)
                    module.bias = torch.zeros(num_features)
            elif key == 'running_var':
                if getattr(module, 'num_features', None) is None:
                    module.num_features = module.running_var.size(0)
                    module.affine = True
                    module.weight = torch.ones(module.num_features)
                    module.bias = torch.zeros(module.num_features)
            elif key == 'num_batches_tracked':
                module.num_batches_tracked = torch.tensor(0, dtype=torch.long)
    else:
        __patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)


def load_networks(net, load_filename, device):
    """Load network state dict with compatibility fixes."""
    if isinstance(net, torch.nn.DataParallel):
        net = net.module
    print('loading the model from %s' % load_filename)
    state_dict = torch.load(load_filename, map_location=str(device))
    if hasattr(state_dict, '_metadata'):
        del state_dict._metadata
    
    # patch InstanceNorm checkpoints prior to 0.4
    for key in list(state_dict.keys()):
        __patch_instance_norm_state_dict(state_dict, net, key.split('.'))
    
    net.load_state_dict(state_dict)
    return net