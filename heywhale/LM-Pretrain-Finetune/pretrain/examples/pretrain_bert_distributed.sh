#!/bin/bash

# --node_ips="$NODE_IPS"  --num_nodes=2  --train_out=/workspace/out --train_log=/workspace/log --train_visualized_log=/workspace/visualizedlog --data_url=/dataset --gpu_num_per_node=8  

# parse args like "--option=argument"
for i in "$@"
do
case $i in
    --num_nodes=*)
    NNODES="${i#*=}"
    shift # past argument=value
    ;;
    --node_ips=*)
    NODE_IPS="${i#*=}"
    shift # past argument=value
    ;;
    --train_out=*)
    OUTPUT_DIR="${i#*=}"
    shift # past argument=value
    ;;
    --gpu_num_per_node=*)
    GPUS_PER_NODE="${i#*=}"
    shift # past argument=value
    ;;
    *)
          # unknown option
    ;;
esac
done


echo "num_nodes=" $NNODES
echo "node_ips=" $NODE_IPS
echo "train_out=" $OUTPUT_DIR
echo "gpu_num_per_node=" $GPUS_PER_NODE

# cd apex
# pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
# cd ..

# NODE_RANK=`python tools/get_node_rank.py --num_nodes=$NNODES --node_ips=$NODE_IPS`
# IFS=',' read -ra ADDR <<< "$NODE_IPS"

# GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=${ADDR[0]}
MASTER_PORT=6000
# NNODES=1
# NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH=wiki-en-v1_text_sentence
CHECKPOINT_PATH=$OUTPUT_DIR/checkpoints/bert_110m

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
echo $DISTRIBUTED_ARGS

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_bert.py \
       --num-layers 12 \
       --hidden-size 768 \
       --num-attention-heads 12 \
       --micro-batch-size 32 \
       --global-batch-size 2048 \
       --seq-length 512 \
       --max-position-embeddings 512 \
       --train-iters 1000000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file bert-large-cased-vocab.txt \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.0001 \
       --lr-decay-style linear \
       --min-lr 1.0e-8 \
       --lr-decay-iters 990000 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --log-interval 100 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --fp16
