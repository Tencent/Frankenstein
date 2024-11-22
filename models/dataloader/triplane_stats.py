
'''
Normalize the triplane dataset, scaling each channel individually to [-1, 1]

Different channels have vastly different ranges of values, unlike RGB images.
'''

import numpy as np
import torch


def compute_stat(sample, stats_dir, middle, _range):
    # RESCALE IMAGE -- Needs to be aligned with input normalization!
    need_reshape = False
    shape_origin = None
    if len(sample.shape) == 5:
        need_reshape = True
        shape_origin = sample.shape
        sample = sample.reshape(shape_origin[0], shape_origin[1]*shape_origin[2], shape_origin[3], shape_origin[4])
    if middle is None or _range is None:
        min_values = np.load(f'{stats_dir}/lower_bound.npy').astype(np.float32).reshape(1, -1, 1, 1)  # should be (1, 96, 1, 1)
        max_values = np.load(f'{stats_dir}/upper_bound.npy').astype(np.float32).reshape(1, -1, 1, 1)
        _range = max_values - min_values
        middle = (min_values + max_values) / 2

    # print(sample.shape)  # eg: [4, 96, 128, 128]
    middle_tensor = torch.tensor(middle, dtype=torch.float32, device=sample.device)
    _range_tensor = torch.tensor(_range, dtype=torch.float32, device=sample.device)

    return sample, middle_tensor, _range_tensor, need_reshape, shape_origin



def unnormalize(sample, stats_dir, middle=None, _range=None):

    sample, middle_tensor, _range_tensor, need_reshape, shape_origin = compute_stat(sample, stats_dir, middle, _range)

    sample = sample * (_range_tensor / 2) + middle_tensor

    if need_reshape:
        sample = sample.view(*shape_origin)

    return sample

def normalize(sample, stats_dir, middle=None, _range=None):
   
    sample, middle_tensor, _range_tensor, need_reshape, shape_origin = compute_stat(sample, stats_dir, middle, _range) 

    sample = (sample - middle_tensor) / (_range_tensor / 2)

    if need_reshape:
        sample = sample.view(*shape_origin)

    return sample
