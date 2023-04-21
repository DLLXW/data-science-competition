
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
import math
import pickle
from contextlib import nullcontext
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
#from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
from dataset import DiagDataset
from torch.optim.lr_scheduler import _LRScheduler
import logging

        
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
    for step, (X, Y) in enumerate(train_loader):
        X=X.to(device)
        Y=Y.to(device)
        lr = get_lr(epoch*iter_per_epoch+step) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # and using the GradScaler if data type is float16
        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
            with ctx:
                logits, loss = model(X, Y)
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
    for _, (X, Y) in enumerate(val_loader):
        X=X.to(device)
        Y=Y.to(device)
        with ctx:
            logits, loss = model(X, Y)
        losses.append(loss.item())
    model.train()
    val_loss=np.mean(losses)
    #
    logger.info('valid loss = {:.4f}'.format(val_loss))
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        logger.info('best val_loss: {} best_epoch: {} '.format(best_val_loss,epoch))
        torch.save(raw_model.state_dict(),'{}/best.pth'.format(save_dir))
    #
    return val_loss

def init_model():
    # model init
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                    bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    meta_vocab_size=1301
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    #print(gptconf)
    model = GPT(gptconf)
    
    # crop down the model block size if desired, using model surgery
    if block_size < model.config.block_size:
        model.crop_block_size(block_size)
        model_args['block_size'] = block_size # so that the checkpoint will have the right value
    return model
# I/O
if __name__=="__main__":
    out_dir = 'out'
    max_epoch=10
    eval_interval = 1
    log_interval = 10
    eval_iters = 200
    eval_only = False # if True, script exits right after the first eval
    always_save_checkpoint = True # if True, always save a checkpoint after each eval
    init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
    #
    gradient_accumulation_steps = 5 # used to simulate larger batch sizes
    batch_size = 64 # if gradient_accumulation_steps > 1, this is the micro-batch size
    block_size = 256
    # model
    n_layer = 12
    n_head = 12
    n_embd = 768
    dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
    bias = False # do we use bias inside LayerNorm and Linear layers?
    # adamw optimizer
    learning_rate = 3e-4 # max learning rate
    max_iters = 2500 # total number of training iterations
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
    # learning rate decay settings
    decay_lr = True # whether to decay the learning rate
    warmup_iters = 200 # how many steps to warm up for
    lr_decay_iters = 2500 # should be ~= max_iters per Chinchilla
    min_lr = 1e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
    # DDP settings
    backend = 'nccl' # 'nccl', 'gloo', etc.
    # system
    device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype = 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    compile = False # use PyTorch 2.0 to compile the model to be faster
    # -----------------------------------------------------------------------------
    config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
    exec(open('configurator.py').read()) # overrides from command line or config file
    config = {k: globals()[k] for k in config_keys} # will be useful for logging
    # -----------------------------------------------------------------------------
    save_dir =os.path.join(out_dir , '20230408_demo')
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    logger = get_logger(os.path.join(save_dir,'log.log'))
    # various inits, derived attributes, I/O setup
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    logger.info("ddp:{}".format(ddp))
    if ddp:
        init_process_group(backend=backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank # each process gets a different seed
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0
        gradient_accumulation_steps *= 8 # simulate 8 gpus
    if master_process:
        os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.autocast(device_type=device_type, dtype=ptdtype)
    #
    best_val_loss = 1e9
    # attempt to derive vocab_size from the dataset
    meta_path = os.path.join('./meta.pkl')
    meta_vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        meta_vocab_size = meta['vocab_size']
        print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")
    #-----init dataloader------
    df=pd.read_csv('./data/diagnosis/train.csv')
    df.columns=['report_ID','description','diagnosis']
    train_X, val_X = train_test_split(df, test_size=0.2, random_state=42)
    print(len(train_X),len(val_X))
    # train_X=train_X[:128]
    # val_X=val_X[:128]
    train_ds = DiagDataset(train_X, max_length=256)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        pin_memory=False,
        drop_last=False,
        shuffle=False,        
        num_workers=0,
    )
    val_ds = DiagDataset(val_X, max_length=256)
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        pin_memory=False,
        drop_last=False,
        shuffle=False,        
        num_workers=0,
    )
    #init model
    model=init_model()
    model.to(device)
    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
    # optimizer
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    #
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=max_epoch, T_mult=1, eta_min=1e-6, last_epoch=-1)
    iter_per_epoch=len(train_loader)
    warmup_epoch=1
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * warmup_epoch,start_lr=1e-6)
    # compile the model
    if compile:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model) # requires PyTorch 2.0
    # wrap model into DDP container
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    #
    raw_model = model.module if ddp else model # unwrap DDP container if needed
    # training loop
    for epoch in range(max_epoch):
        train_epoch(epoch)
        val_loss=valid_epoch(epoch)
        torch.save(raw_model.state_dict(),'{}/epoch_{}.pth'.format(save_dir,epoch))
    if ddp:
        destroy_process_group()