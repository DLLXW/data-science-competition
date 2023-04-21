#!/bin/bash

GPUS_PER_NODE=2 #几张卡
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1 #几台机器
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH="dataset/"#数据集路径
CHECKPOINT_PATH=checkpoints/bart-large #预训练模型输出保存路径
VOCAB_FILE=vocab-bart-large/ #词表和配置

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
#num-layers num-decoder-layers hidden-size num-attention-heads 这几个参数这里默认的是base模型的
#large模型的本来需要相应进行修改，但是我已经写到vocab-bart-large/config.json里面了，这里传入的不会生效，代码里面已经注释了。
python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_bart.py \
       --num-layers 24 \
       --num-decoder-layers 12 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --micro-batch-size 16 \
       --global-batch-size 256 \
       --seq-length 256 \
       --max-position-embeddings 256 \
       --mask-prob 0.15 \
       --train-iters 4000 \
       --lr-decay-iters 4000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file $VOCAB_FILE \
       --data-impl mmap \
       --split 90,9,1 \
       --distributed-backend nccl \
       --lr 6e-5 \
       --lr-decay-style cosine \
       --min-lr 1e-6 \
       --initial-loss-scale 65536 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .1 \
       --log-interval 50 \
       --save-interval 1000 \
       --eval-interval 500 \
       --eval-iters 10 \
       --fp16 \
       --optimizer adam \
       --num-workers 2 \
       # --checkpoint-activations
