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

from apex.optimizers import FusedAdam as Adam
from apex.optimizers import FusedSGD as SGD
from apex.optimizers import FusedLAMB as LAMB

from megatron import get_args
from megatron.model import LayerNorm

from .grad_scaler import ConstantGradScaler, DynamicGradScaler
from .optimizer import Float16OptimizerWithFloat16Params, FP32Optimizer
import torch.nn as nn

def _get_params_for_weight_decay_optimization_old(modules):
    """Divide params into with-weight-decay and without-weight-decay groups.
    Layernorms and baises will have no weight decay but the rest will.
    """

    weight_decay_params = {'params': []}
    no_weight_decay_params = {'params': [], 'weight_decay': 0.0}
    for module in modules:
        for module_ in module.modules():
            if isinstance(module_, (LayerNorm, nn.LayerNorm)):
                no_weight_decay_params['params'].extend(
                    [p for p in list(module_._parameters.values())
                     if p is not None])
            else:
                weight_decay_params['params'].extend(
                    [p for n, p in list(module_._parameters.items())
                     if p is not None and n != 'bias'])
                no_weight_decay_params['params'].extend(
                    [p for n, p in list(module_._parameters.items())
                     if p is not None and n == 'bias'])

    return weight_decay_params, no_weight_decay_params

def _get_params_for_weight_decay_optimization(modules):
    """Divide params into with-weight-decay and without-weight-decay groups.
    Layernorms and baises will have no weight decay but the rest will.
    """
    wd_params = []
    no_wd_params = []
    for module in modules:
        for name, p in module.named_parameters():
            n = name.lower()
            if 'layer_norm.' in n or 'layernorm.' in n or '.bias' in n:
                no_wd_params.append(p)
                # print(name, 'no_wd')

            else:
                wd_params.append(p)
                # print(name, 'wd')
    
    param_groups = [
        {'params': wd_params},
        {'params': no_wd_params, 'weight_decay': 0.0},
    ]
    return param_groups

def _get_params_for_weight_decay_optimization_encoder_decoder(modules):
    """Divide params into with-weight-decay and without-weight-decay groups.
    Layernorms and baises will have no weight decay but the rest will.
    """
    encoder_wd_params = []
    encoder_no_wd_params = []
    wd_params = []
    no_wd_params = []
    for module in modules:
        for name, p in module.named_parameters():
            n = name.lower()
            if 'encoder.' in name:
                if 'layer_norm.' in n or 'layernorm.' in n or '.bias' in n:
                    encoder_no_wd_params.append(p)
                    # print(name, 'encoder_no_wd')
                else:
                    encoder_wd_params.append(p)
                    # print(name, 'encoder_wd')

            else:
                if 'layer_norm.' in n or 'layernorm.' in n or '.bias' in n:
                    no_wd_params.append(p)
                    # print(name, 'no_wd')

                else:
                    wd_params.append(p)
                    # print(name, 'wd')
    
    args = get_args()
    param_groups = [
        {'params': encoder_no_wd_params, 'weight_decay': 0.0, 'lr': args.lr_encoder},
        {'params': encoder_wd_params, 'lr': args.lr_encoder},
        {'params': no_wd_params, 'weight_decay': 0.0},
        {'params': wd_params},
    ]
    return param_groups

def get_megatron_optimizer(model):
    args = get_args()

    # Base optimizer.
    if args.lr_encoder is not None:
        param_groups = _get_params_for_weight_decay_optimization_encoder_decoder(model)
    else:
        param_groups = _get_params_for_weight_decay_optimization(model)
    # for name, param in model[0].named_parameters():
    #     print(name, param.size())
    if args.optimizer == 'adam':
        optimizer = Adam(param_groups,
                         lr=args.lr,
                         weight_decay=args.weight_decay,
                         betas=(args.adam_beta1, args.adam_beta2),
                         eps=args.adam_eps)
    elif args.optimizer == 'sgd':
        optimizer = SGD(param_groups,
                        lr=args.lr,
                        weight_decay=args.weight_decay,
                        momentum=args.sgd_momentum)
    elif args.optimizer == 'lamb':
        optimizer = LAMB(param_groups,
                         lr=args.lr,
                         weight_decay=args.weight_decay,
                         betas=(args.adam_beta1, args.adam_beta2),
                         eps=args.adam_eps)
    else:
        raise Exception('{} optimizer is not supported.'.format(
            args.optimizer))

    if args.deepspeed:
        # fp16 wrapper is not required for DeepSpeed.
        return optimizer

    # Determine whether the params have main-grad field.
    params_have_main_grad = False
    if args.DDP_impl == 'local':
        params_have_main_grad = True

    if args.fp16 or args.bf16:

        # Grad scaler:
        #    if loss-scale is provided, instantiate the constant scaler.
        #    if we are using fp16 and loss-scale is not present, use a
        #       dynamic scaler.
        #    otherwise we are running in bf16 with no loss-scale so
        #       leave it as None.
        grad_scaler = None
        # Constant loss scale.
        if args.loss_scale:
            grad_scaler = ConstantGradScaler(args.loss_scale)
        # Dynamic loss scale.
        else:
            if args.fp16:
                grad_scaler = DynamicGradScaler(
                    initial_scale=args.initial_loss_scale,
                    min_scale=args.min_loss_scale,
                    growth_factor=2.0,
                    backoff_factor=0.5,
                    growth_interval=args.loss_scale_window,
                    hysteresis=args.hysteresis)

        # Megatron optimizer.
        return Float16OptimizerWithFloat16Params(optimizer,
                                                 args.clip_grad,
                                                 args.log_num_zeros_in_grad,
                                                 params_have_main_grad,
                                                 args.bf16,
                                                 grad_scaler)

    # FP32.
    return FP32Optimizer(optimizer, args.clip_grad,
                         args.log_num_zeros_in_grad,
                         params_have_main_grad)
