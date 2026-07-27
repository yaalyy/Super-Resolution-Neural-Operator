import torch
from torch.utils.data import Dataset

from datasets import register


@register('darcy-flow-pt')
class DarcyFlowPT(Dataset):

    def __init__(self, path, field='y', sample_start=0, sample_end=None,
                 sample_step=1, first_k=None, repeat=1):
        self.path = path
        self.field = field
        self.sample_step = sample_step
        self.first_k = first_k
        self.repeat = repeat

        data = torch.load(self.path, map_location='cpu')
        if not isinstance(data, dict):
            raise TypeError('{} should contain a dict, got {}'.format(
                self.path, type(data)))
        if self.field not in data:
            raise KeyError('{} not found in {}; available fields: {}'.format(
                self.field, self.path, sorted(data.keys())))

        field_tensor = data[self.field].float().contiguous()
        if field_tensor.dim() not in (3, 4):
            raise ValueError('unsupported field shape for {}: {}'.format(
                self.field, tuple(field_tensor.shape)))

        self.data = field_tensor
        self.sample_count = self.data.shape[0]
        if sample_end is None:
            sample_end = self.sample_count
        self.sample_indices = list(range(sample_start, sample_end, sample_step))

        total = len(self.sample_indices)
        if first_k is not None:
            total = min(total, first_k)
        self.total = total

    def __len__(self):
        return self.total * self.repeat

    def __getitem__(self, idx):
        idx = idx % self.total
        sample_idx = self.sample_indices[idx]
        field = self.data[sample_idx]
        if field.dim() == 2:
            field = field.unsqueeze(0)
        return field
