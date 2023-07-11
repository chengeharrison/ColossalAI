set_n_least_used_CUDA_VISIBLE_DEVICES() {
    local n=${1:-"9999"}
    echo "GPU Memory Usage:"
    local FIRST_N_GPU_IDS=$(nvidia-smi --query-gpu=memory.used --format=csv \
        | tail -n +2 \
        | nl -v 0 \
        | tee /dev/tty \
        | sort -g -k 2 \
        | awk '{print $1}' \
        | head -n $n)
    export CUDA_VISIBLE_DEVICES=$(echo $FIRST_N_GPU_IDS | sed 's/ /,/g')
    echo "Now CUDA_VISIBLE_DEVICES is set to:"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
}

set_n_least_used_CUDA_VISIBLE_DEVICES 1
# export CUDA_VISIBLE_DEVICES="4,5,6,7"

torchrun --standalone --nproc_per_node=1 train_sft_mem.py \
    --pretrain "/data3/data/SFT_improvement/LLaMa-7B/" \
    --model 'llama' \
    --strategy colossalai_zero2_cpu \
    --log_interval 10 \
    --save_path  "/data3/data/SFT_improvement/Coati-SFT-v2-433k" \
    --dataset "/data/users/lcxyc/improve_sft/final/final_dataset.json" \
    --batch_size 4 \
    --accumulation_steps 8 \
    --lr "5e-6" \
    --grad_checkpoint \
    --max_len 2048 \
    --use_wandb \
    --max_epochs 3 \
