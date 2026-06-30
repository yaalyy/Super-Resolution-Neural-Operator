# modified from: https://github.com/yinboc/liif

import os
import time
import shutil
import math

import torch
import numpy as np
from torch.optim import SGD
from tensorboardX import SummaryWriter
from Adam import Adam
import cv2
import matplotlib.pyplot as plt

from torchvision import transforms
from torchvision.transforms import InterpolationMode
import random
import math

def show_feature_map(feature_map,layer,name='rgb',rgb=False):
    feature_map = feature_map.squeeze(0)
    #if rgb: feature_map = feature_map.permute(1,2,0)*0.5+0.5
    feature_map = feature_map.cpu().numpy()
    feature_map_num = feature_map.shape[0]
    row_num = math.ceil(np.sqrt(feature_map_num))
    if rgb:
        #plt.figure()
        #plt.imshow(feature_map)
        #plt.axis('off')
        feature_map = cv2.cvtColor(feature_map,cv2.COLOR_BGR2RGB)
        cv2.imwrite('data/'+layer+'/'+name+".png",feature_map*255)
        #plt.show()
    else:
        plt.figure()
        for index in range(1, feature_map_num+1):
            t = (feature_map[index-1]*255).astype(np.uint8)
            t = cv2.applyColorMap(t, cv2.COLORMAP_TWILIGHT)
            plt.subplot(row_num, row_num, index)
            plt.imshow(t, cmap='gray')
            plt.axis('off')
            #ensure_path('data/'+layer)
            cv2.imwrite('data/'+layer+'/'+str(name)+'_'+str(index)+".png",t)
        #plt.show()
        plt.savefig('data/'+layer+'/'+str(name)+".png")

def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, InterpolationMode.BICUBIC)(
            transforms.ToPILImage()(img)))

def downsample(img, scale_min=1, scale_max=4, inp_size=None, augment=False, epoch=None):
    if epoch<20: s = random.randint(scale_min, scale_max)
    s = random.uniform(scale_min, scale_max)
    #print(s)
    
    if inp_size is None:
        h_lr = math.floor(img.shape[-2] / s + 1e-9)
        w_lr = math.floor(img.shape[-1] / s + 1e-9)
        h_hr = round(h_lr * s)
        w_hr = round(w_lr * s)
        img = img[:, :, :h_hr, :w_hr]
        img_down = torch.stack([resize_fn(x, (h_lr, w_lr)) for x in img], dim=0) 
        crop_lr, crop_hr = img_down, img
    else:
        h_lr = inp_size
        w_lr = inp_size
        h_hr = round(h_lr * s)
        w_hr = round(w_lr * s)
        x0 = random.randint(0, img.shape[-2] - w_hr)
        y0 = random.randint(0, img.shape[-1] - w_hr)
        crop_hr = img[:, :, x0: x0 + w_hr, y0: y0 + w_hr]
        crop_lr = torch.stack([resize_fn(x, w_lr) for x in crop_hr], dim=0)
    
    if augment == True:
        hflip = random.random() < 0.5
        vflip = random.random() < 0.5
        dflip = random.random() < 0.5

        def augment(x):
            if hflip: x = x.flip(-2)
            if vflip: x = x.flip(-1)
            if dflip: x = x.transpose(-2, -1)
            return x
        crop_lr = augment(crop_lr)
        crop_hr = augment(crop_hr)

    coord = make_coord([h_hr, w_hr], flatten=False)
    coord = coord.unsqueeze(0).expand(img.shape[0], *coord.shape[:2], 2)

    cell = torch.tensor([2 / crop_hr.shape[-2], 2 / crop_hr.shape[-1]], dtype=torch.float32).unsqueeze(0).expand(img.shape[0], 2)
    return {
        'inp': crop_lr.contiguous(),
        'coord': coord.contiguous(),
        'cell': cell.contiguous(),
        'gt': crop_hr.contiguous()
    }    

class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


class Timer():

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v


def time_text(t):
    if t >= 3600:
        return '{:.3f}h'.format(t / 3600)
    elif t >= 60:
        return '{:.3f}m'.format(t / 60)
    else:
        return '{:.3f}s'.format(t)


_log_path = None


def set_log_path(path):
    global _log_path
    _log_path = path


def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)


def ensure_path(path, remove=True):
    basename = os.path.basename(path.rstrip('/'))
    if os.path.exists(path):
        if remove and (basename.startswith('_')
                or input('{} exists, remove? (y/[n]): '.format(path)) == 'y'):
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


def set_save_path(save_path, remove=True):
    ensure_path(save_path, remove=remove)
    set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    return log, writer


def compute_num_params(model, text=False):
    tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
    if text:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


def make_optimizer(param_list, optimizer_spec, load_sd=False):
    Optimizer = {
        'sgd': SGD,
        'adam': Adam
    }[optimizer_spec['name']]
    optimizer = Optimizer(param_list, **optimizer_spec['args'])
    if load_sd:
        optimizer.load_state_dict(optimizer_spec['sd'])
    return optimizer


def numpy_collate(batch):
    elem = batch[0]
    if isinstance(elem, dict):
        return {key: numpy_collate([sample[key] for sample in batch]) for key in elem}
    if torch.is_tensor(elem):
        return np.stack([sample.detach().cpu().numpy() for sample in batch], axis=0)
    if isinstance(elem, np.ndarray):
        return np.stack(batch, axis=0)
    return np.asarray(batch)


def to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=(device.type == 'cuda'))
    if isinstance(value, np.ndarray):
        return torch.as_tensor(value, device=device)
    return value


def batch_to_device(batch, device):
    return {key: to_device(value, device) for key, value in batch.items()}


def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    #ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    ret = torch.stack(torch.meshgrid(*coord_seqs,indexing='ij'), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


def to_pixel_samples(img):
    """ Convert the image to coord-RGB pairs.
        img: Tensor, (3, H, W)
    """
    coord = make_coord(img.shape[-2:])
    rgb = img.view(3, -1).permute(1, 0)
    return coord, rgb


def calc_psnr(sr, hr, dataset=None, scale=1, rgb_range=1):
    diff = (sr - hr) / rgb_range
    if dataset is not None:
        if dataset == 'benchmark':
            shave = scale
            if diff.size(1) > 1:
                gray_coeffs = [65.738, 129.057, 25.064]
                convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
                diff = diff.mul(convert).sum(dim=1)
        elif dataset == 'div2k':
            shave = scale + 6
        else:
            raise NotImplementedError
        valid = diff[..., shave:-shave, shave:-shave]
    else:
        valid = diff
    mse = valid.pow(2).mean()
    return -10 * torch.log10(mse)


def calc_mse(pred, gt):
    return (pred - gt).pow(2).mean()


def calc_rel_l2(pred, gt, eps=1e-12):
    diff = (pred - gt).reshape(pred.shape[0], -1)
    target = gt.reshape(gt.shape[0], -1)
    rel = torch.linalg.norm(diff, dim=1) / (torch.linalg.norm(target, dim=1) + eps)
    return rel.mean()
