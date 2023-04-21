# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Blendable dataset."""

import time

import numpy as np
import torch

from megatron import print_rank_0
from megatron import mpu


class BlendableDataset(torch.utils.data.Dataset):


    def __init__(self, datasets, weights):

        self.datasets = datasets
        num_datasets = len(datasets)
        assert num_datasets == len(weights)

        self.size = 0
        for dataset in self.datasets:
            self.size += len(dataset)

        # Normalize weights.
        weights = np.array(weights, dtype=np.float64)
        sum_weights = np.sum(weights)
        assert sum_weights > 0.0
        weights /= sum_weights

        # Build indecies.
        start_time = time.time()
        assert num_datasets < 255
        # self.dataset_index = np.zeros(self.size, dtype=np.uint8)
        # self.dataset_sample_index = np.zeros(self.size, dtype=np.int64)

        # FIXME bugs in build_blending_indices, causing index out of boundary 
        # from megatron.data import helpers
        # helpers.build_blending_indices(self.dataset_index,
        #                                self.dataset_sample_index,
        #                                weights, num_datasets, self.size,
        #                                torch.distributed.get_rank() == 0)

        # start = 0
        # for i, dataset in enumerate(self.datasets):
        #     end = start + len(dataset)
        #     self.dataset_index[start:end] = i
        #     self.dataset_sample_index[start:end] = np.arange(len(dataset), dtype=np.int64)
        #     start = end
        self.dataset_sizes = np.array([len(dataset) for dataset in self.datasets])
        self.dataset_cum_sizes = np.cumsum(self.dataset_sizes)
        assert self.dataset_cum_sizes[-1] == self.size
        self.num_datasets = num_datasets

        self.collate_fn = getattr(datasets[0], 'collate_fn', None)

        print_rank_0('> elapsed time for building blendable dataset indices: '
                     '{:.2f} (sec)'.format(time.time() - start_time))


    def __len__(self):
        return self.size


    def __getitem__(self, idx):
        dataset_idx = idx % self.num_datasets
        sample_idx = idx // self.num_datasets
        return self.datasets[dataset_idx][sample_idx]

        # # dataset_idx = self.dataset_index[idx]
        # # sample_idx = self.dataset_sample_index[idx]
        # dataset_idx = np.searchsorted(self.dataset_cum_sizes, idx, 'right')
        # if dataset_idx > 0:
        #     sample_idx = idx - self.dataset_cum_sizes[dataset_idx-1]
        # else:
        #     sample_idx = idx
        # return self.datasets[dataset_idx][sample_idx]
