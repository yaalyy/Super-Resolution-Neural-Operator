import random
import math
from torchvision.transforms import InterpolationMode

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import register
from utils import make_coord

def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, InterpolationMode.BICUBIC)(
            transforms.ToPILImage()(img)))

@register('sr-implicit-downsampled-fast')
class SRImplicitDownsampledFast(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        s = random.uniform(self.scale_min, self.scale_max)

        if self.inp_size is None:
            h_lr = math.floor(img.shape[-2] / s + 1e-9)
            w_lr = math.floor(img.shape[-1] / s + 1e-9)
            h_hr = round(h_lr * s)
            w_hr = round(w_lr * s)
            img = img[:, :h_hr, :w_hr]
            img_down = resize_fn(img, (h_lr, w_lr))
            crop_lr, crop_hr = img_down, img
        else:
            h_lr = self.inp_size
            w_lr = self.inp_size
            h_hr = round(h_lr * s)
            w_hr = round(w_lr * s)
            x0 = random.randint(0, img.shape[-2] - w_hr)
            y0 = random.randint(0, img.shape[-1] - w_hr)
            crop_hr = img[:, x0: x0 + w_hr, y0: y0 + w_hr]
            crop_lr = resize_fn(crop_hr, w_lr)

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord = make_coord([h_hr, w_hr], flatten=False)
        hr_rgb = crop_hr

        if self.inp_size is not None:
            
            idx = torch.tensor(np.random.choice(h_hr*w_hr, h_lr*w_lr, replace=False))
            #idx,_ = torch.sort(idx)
            hr_coord = hr_coord.view(-1, hr_coord.shape[-1])
            hr_coord = hr_coord[idx, :]
            hr_coord = hr_coord.view(h_lr, w_lr, hr_coord.shape[-1])

            hr_rgb = crop_hr.contiguous().view(crop_hr.shape[0], -1)
            hr_rgb = hr_rgb[:, idx]
            hr_rgb = hr_rgb.view(crop_hr.shape[0], h_lr, w_lr)
        
        cell = torch.tensor([2 / crop_hr.shape[-2], 2 / crop_hr.shape[-1]], dtype=torch.float32)
        
        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }


def resize_tensor(img, size, mode='bicubic'):
    align_corners = False if mode in ('bilinear', 'bicubic') else None
    kwargs = {'mode': mode}
    if align_corners is not None:
        kwargs['align_corners'] = align_corners
    return F.interpolate(img.unsqueeze(0), size=size, **kwargs).squeeze(0)


@register('pde-implicit-downsampled')
class PDEImplicitDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=2, scale_max=None,
                 augment=False, interpolation='bicubic'):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.interpolation = interpolation

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        field = self.dataset[idx]
        s = random.uniform(self.scale_min, self.scale_max)

        if self.inp_size is None:
            h_lr = math.floor(field.shape[-2] / s + 1e-9)
            w_lr = math.floor(field.shape[-1] / s + 1e-9)
            h_hr = round(h_lr * s)
            w_hr = round(w_lr * s)
            crop_hr = field[:, :h_hr, :w_hr]
        else:
            h_lr = self.inp_size
            w_lr = self.inp_size
            h_hr = round(h_lr * s)
            w_hr = round(w_lr * s)
            if h_hr > field.shape[-2] or w_hr > field.shape[-1]:
                raise ValueError('HR crop {}x{} is larger than field {}x{}'.format(
                    h_hr, w_hr, field.shape[-2], field.shape[-1]))
            x0 = random.randint(0, field.shape[-2] - h_hr)
            y0 = random.randint(0, field.shape[-1] - w_hr)
            crop_hr = field[:, x0: x0 + h_hr, y0: y0 + w_hr]

        crop_lr = resize_tensor(crop_hr, (h_lr, w_lr), mode=self.interpolation)

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord = make_coord([h_hr, w_hr], flatten=False)
        cell = torch.tensor([2 / h_hr, 2 / w_hr], dtype=torch.float32)

        return {
            'inp': crop_lr.contiguous(),
            'coord': hr_coord.contiguous(),
            'cell': cell,
            'gt': crop_hr.contiguous()
        }
