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
# "/data3/data/Coati-last6/"
# "/data3/data/SFT_improvement/Coati-SFT-v2-433k-epoch2-3-steps-6.77k/"
# "/data3/data/model_eval_for_commerical_use/phoenix-inst-chat-7b/"
set_n_least_used_CUDA_VISIBLE_DEVICES 1
# export CUDA_VISIBLE_DEVICES="1"
python chat.py \
    --model_path "/data3/data/model_eval_for_commerical_use/phoenix-inst-chat-7b/" \
    --model_max_length 2048 \
    --max_new_tokens 1024 \