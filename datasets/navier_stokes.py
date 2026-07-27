import numpy as np
import torch
from torch.utils.data import Dataset

from datasets import register


@register('navier-stokes-mat')
class NavierStokesMat(Dataset):

    def __init__(self, path, field='u', sample_start=0, sample_end=None,
                 time_start=0, time_end=None, sample_step=1, time_step=1,
                 first_k=None, repeat=1, one_frame_per_sample=False,
                 random_time=False):
        self.path = path
        self.field = field
        self.sample_step = sample_step
        self.time_step = time_step
        self.first_k = first_k
        self.repeat = repeat
        self.one_frame_per_sample = one_frame_per_sample
        self.random_time = random_time
        self._file = None

        import h5py
        with h5py.File(self.path, 'r') as f:
            if self.field not in f:
                raise KeyError('{} not found in {}'.format(self.field, self.path))

            shape = f[self.field].shape
            if self.field == 'u':
                self.has_time = True
                self.time_count, self.height, self.width, self.sample_count = shape
            elif len(shape) == 3:
                self.has_time = False
                self.height, self.width, self.sample_count = shape
                self.time_count = 1
            else:
                raise ValueError('unsupported field shape for {}: {}'.format(
                    self.field, shape))

        if sample_end is None:
            sample_end = self.sample_count
        if time_end is None:
            time_end = self.time_count

        self.sample_indices = list(range(sample_start, sample_end, sample_step))
        self.time_indices = list(range(time_start, time_end, time_step))
        if not self.has_time:
            self.time_indices = [0]
            self.one_frame_per_sample = True

        if self.one_frame_per_sample:
            total = len(self.sample_indices)
        else:
            total = len(self.sample_indices) * len(self.time_indices)
        if first_k is not None:
            total = min(total, first_k)
        self.total = total

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_file'] = None
        return state

    def _get_file(self):
        if self._file is None:
            import h5py
            self._file = h5py.File(self.path, 'r')
        return self._file

    def __len__(self):
        return self.total * self.repeat

    def __getitem__(self, idx):
        idx = idx % self.total
        n_times = len(self.time_indices)
        if self.one_frame_per_sample:
            sample_idx = self.sample_indices[idx]
            if self.random_time:
                time_idx = self.time_indices[np.random.randint(n_times)]
            else:
                time_idx = self.time_indices[idx % n_times]
        else:
            sample_idx = self.sample_indices[idx // n_times]
            time_idx = self.time_indices[idx % n_times]

        dataset = self._get_file()[self.field]
        if self.has_time:
            field = np.asarray(dataset[time_idx, :, :, sample_idx], dtype=np.float32)
        else:
            field = np.asarray(dataset[:, :, sample_idx], dtype=np.float32)

        return torch.from_numpy(field).unsqueeze(0)
