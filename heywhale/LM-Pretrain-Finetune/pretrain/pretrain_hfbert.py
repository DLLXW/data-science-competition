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

"""Pretrain BART"""

from functools import partial

import torch

from megatron import (
    get_args,
    get_timers,
    mpu,
    print_rank_0,
    get_tokenizer
)
from megatron.data.dataset_utils import build_train_valid_test_datasets
from megatron.model.bart_model import BartModel
from megatron.training import pretrain
from megatron.utils import average_losses_across_data_parallel_group


def model_provider(pre_process=True, post_process=True):
    """Build the model."""
    assert pre_process and post_process, "BART doesn't yet support pipelining"

    print_rank_0('building BART model ...')
    model = BartModel()
    print_rank_0(model)
    return model


def get_batch(data_iterator):
    """Build the batch."""

    keys = ['source', 'target', 'prev_output_tokens', 'attn_mask', 'loss_mask', 'use_decoder']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    source = data_b['source'].long()
    target = data_b['target'].long()
    prev_output_tokens = data_b['prev_output_tokens'].long()
    attn_mask = data_b['attn_mask'].long()
    loss_mask = data_b['loss_mask'].float()
    use_decoder = data_b['use_decoder'].long()
    # print('source', source[0])
    # print('target', target[0])
    # tokenizer = get_tokenizer()
    # print('source', tokenizer.detokenize(source[0]))
    # print('target', tokenizer.detokenize(target[0]))
    return source, target, prev_output_tokens, attn_mask, loss_mask, use_decoder


def loss_func(loss_mask, output_tensor):
    lm_loss_, _ = output_tensor

    lm_loss_ = lm_loss_.float()
    lm_loss = torch.sum(
        lm_loss_.view(-1) * loss_mask.reshape(-1)) / loss_mask.sum()

    loss = lm_loss
    timers = get_timers()
    timers('reducing-losses').start()
    averaged_losses = average_losses_across_data_parallel_group([lm_loss])
    timers('reducing-losses').stop()

    return loss, {'lm loss': averaged_losses[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    timers = get_timers()

    # Get the batch.
    timers('batch-generator').start()
    source, target, prev_output_tokens, attn_mask, loss_mask, use_decoder = get_batch(data_iterator)
    timers('batch-generator').stop()

    # Forward model lm_labels
    output_tensor = model(source, attn_mask, prev_output_tokens, target, use_decoder)

    return output_tensor, partial(loss_func, loss_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for BART ...')
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        max_seq_length=args.encoder_seq_length,
        max_seq_length_dec=args.decoder_seq_length,
        masked_lm_prob=args.mask_prob,
        short_seq_prob=args.short_seq_prob,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup),
        dataset_type='bart')
    print_rank_0("> finished creating BART datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    pretrain(train_valid_test_datasets_provider, model_provider, forward_step,
             args_defaults={'tokenizer_type': 'Huggingface'})
