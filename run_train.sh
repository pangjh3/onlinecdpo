src=en
tgt=zh
# size=$3
prenum=65399



# Multi-nodes are also supported
export CUDA_VISIBLE_DEVICES=0,1,2,3

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_NET_GDR_READ=1

export MASTER_ADDR="${CHIEF_IP:=localhost}"
export MASTER_PORT="${MASTER_PORT:=29500}"

export HF_HOME=/apdcephfs/share_733425/vinnylywang/jianhuipang/hf_cache
export TRANSFORMERS_CACHE=/apdcephfs/share_733425/vinnylywang/jianhuipang/hf_cache



train_path=train.py
# model_path=/apdcephfs/share_733425/vinnylywang/zefengdu/llama2-chinese/model/llama2-7B-HF

# HOST_NUM will be 1
HOST_NUM=1
torchrun --nnodes $HOST_NUM --node_rank $INDEX --nproc_per_node 4 \
    --master_addr $MASTER_ADDR --master_port $MASTER_PORT  \
    ${train_path} \
