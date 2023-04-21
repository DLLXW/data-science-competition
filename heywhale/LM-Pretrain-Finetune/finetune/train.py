
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import time
import math
import inspect
import pickle
from contextlib import nullcontext
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from transformers import BartForConditionalGeneration
#from evaluate import CiderD
#from torch.nn.parallel import DistributedDataParallel as DDP
#from torch.distributed import init_process_group, destroy_process_group

from dataset_cn import DiagDataset
from torch.optim.lr_scheduler import _LRScheduler
import logging
from lion import Lion

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler

    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self,  optimizer, total_iters, start_lr=1e-6,last_epoch=-1):
        self.total_iters = total_iters
        self.start_lr=start_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [self.start_lr+base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger

# -----------------------------------------------------------------------------

def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

def train_epoch(epoch):
    start_time=time.time()
    for step, (desc_id, desc_mask,diag_id,diag_label) in enumerate(train_loader):
        desc_id = desc_id.to(device)
        desc_mask = desc_mask.to(device)
        diag_id = diag_id.to(device)
        diag_label = diag_label.to(device)
        #学习率调节
        # if epoch>=warmup_epoch:
        #     scheduler.step(epoch-warmup_epoch + step / iter_per_epoch)
        # else:
        #     warmup_scheduler.step()
        lr = get_lr(epoch*iter_per_epoch+step) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # and using the GradScaler if data type is float16
        for micro_step in range(gradient_accumulation_steps):
            with ctx:
                output = model(
                    input_ids=desc_id,
                    attention_mask=desc_mask,
                    decoder_input_ids=diag_id,
                    labels=diag_label,
                )
            loss = output[0]
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
        # clip the gradient
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)
        #打印日志
        if step % log_interval == 0:
            spend_time=time.time()-start_time
            logger.info(
                    'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.7f} epoch_Time:{}min:'.format(
                        epoch,
                        max_epoch, 
                        step, 
                        iter_per_epoch,
                        loss.item(), 
                        optimizer.param_groups[-1]['lr'],
                        spend_time / (step+1) * iter_per_epoch // 60 - spend_time // 60))

@torch.no_grad()
def valid_epoch(epoch):
    global best_val_loss
    losses = []
    model.eval()
    for _, (desc_id, desc_mask,diag_id,diag_label) in enumerate(val_loader):
        desc_id = desc_id.to(device)
        desc_mask = desc_mask.to(device)
        diag_id = diag_id.to(device)
        diag_label = diag_label.to(device)
        with ctx:
            output = model(
                input_ids=desc_id,
                attention_mask=desc_mask,
                decoder_input_ids=diag_id,
                labels=diag_label,
            )
        loss = output[0]
        losses.append(loss.item())
    #
    model.train()
    val_loss=np.mean(losses)
    #
    logger.info('valid loss = {:.4f}'.format(val_loss))
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        logger.info('best val_loss: {} best_epoch: {} '.format(best_val_loss,epoch))
        torch.save(model.state_dict(),'{}/best.pth'.format(save_dir))
    #
    return val_loss

def get_group_parameters(model):
    params = list(model.named_parameters())
    no_decay = ['bias,','layer_norm']
    #embed = ['shared.weight','embed_positions.weight']
    #no_main = no_decay + other
    param_group = [
        {'params':[p for n,p in params if not any(nd in n for nd in no_decay)],'weight_decay':1e-2},#需要wd的层
        {'params':[p for n,p in params if any(nd in n for nd in no_decay)],'weight_decay':0}#
    ]
    return param_group

# I/O
if __name__=="__main__":
    out_dir = 'output/'
    max_epoch=6
    eval_interval = 1
    log_interval = 50
    #
    gradient_accumulation_steps = 1 # used to simulate larger batch sizes
    batch_size = 16 # if gradient_accumulation_steps > 1, this is the micro-batch size
    # adamw optimizer
    learning_rate = 4e-5 # max learning rate
    max_iters = 6000 # total number of training iterations
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
    # learning rate decay settings
    decay_lr = True # whether to decay the learning rate
    warmup_iters = 1000 # how many steps to warm up for
    lr_decay_iters = 6000 # should be ~= max_iters per Chinchilla
    min_lr = 1e-6 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
    # DDP settings
    backend = 'nccl' # 'nccl', 'gloo', etc.
    # system
    device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype = 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    device_type = 'cuda' if 'cuda' in device else 'cpu' 
    ctx = nullcontext() if device_type == 'cpu' else torch.autocast(device_type=device_type, dtype=ptdtype)
    # -----------------------------------------------------------------------------
    save_dir =os.path.join(out_dir , '20230419/')
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    logger = get_logger(os.path.join(save_dir,'log.log'))
    # various inits, derived attributes, I/O setup
   
    best_val_loss = 1e9
    # attempt to derive vocab_size from the dataset
    #-----init dataloader------
    df=pd.read_csv('./data/diagnosis/train.csv',)
    df.columns=['report_ID','description','diagnosis']
    train_X, val_X = train_test_split(df, test_size=0.2, random_state=42)
    print(len(train_X),len(val_X))
    # train_X=train_X[:128]
    # val_X=val_X[:128]
    train_ds = DiagDataset(train_X, max_length=256,finetune=True)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        pin_memory=False,
        drop_last=False,
        shuffle=False,        
        num_workers=0,
    )
    val_ds = DiagDataset(val_X, max_length=256,finetune=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        pin_memory=False,
        drop_last=False,
        shuffle=False,        
        num_workers=0,
    )
    #init model
    model = BartForConditionalGeneration.from_pretrained("custom_pretrain_bart/")#fnlp/cpt-large or fnlp/bart-large-chinese
    #model = CPTForConditionalGeneration.from_pretrained("./custom_pretrain_cpt/")
    #model.resize_token_embeddings(1401)
    #
    # ckpt_path = './output/pesudo_pretrain/best.pth'
    # state_dict = torch.load(ckpt_path, map_location='cpu')
    # model.load_state_dict(state_dict)
    # print("load_ckpt from .....")
    model.to(device)
    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
    # optimizer
    # new PyTorch nightly has a new 'fused' option for AdamW that is much faster
    use_fused = (device_type == 'cuda') and ('fused' in inspect.signature(torch.optim.AdamW).parameters)
    print(f"using fused AdamW: {use_fused}")
    extra_args = dict(fused=True) if use_fused else dict()
    #optimizer = torch.optim.AdamW(params=model.parameters(),lr=learning_rate, betas=(beta1, beta2), **extra_args)
    param_group=get_group_parameters(model)
    optimizer = torch.optim.AdamW(param_group,lr=learning_rate, betas=(beta1, beta2), **extra_args)
    #optimizer = Lion(model.parameters(),lr=learning_rate,weight_decay=1e-2)
    #print("user lion .....")
    #
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=max_epoch, T_mult=1, eta_min=1e-6, last_epoch=-1)
    iter_per_epoch=len(train_loader)
    #warmup_epoch=1
    #warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * warmup_epoch,start_lr=1e-6)
    # training loop
    for epoch in range(max_epoch):
        train_epoch(epoch)
        val_loss=valid_epoch(epoch)
        torch.save(model.state_dict(),'{}/epoch_last.pth'.format(save_dir))