# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.
import argparse
# Need to call this before importing transformers.
from llama_flash_attn_monkey_patch import (
    replace_llama_attn_with_flash_attn,
)

replace_llama_attn_with_flash_attn()

from train_sft import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy',
                        choices=['naive', 'ddp', 'colossalai_gemini', 'colossalai_zero2', 'colossalai_zero2_cpu'],
                        default='colossalai_zero2')
    parser.add_argument('--model', choices=['gpt2', 'bloom', 'opt', 'llama'], default='bloom')
    parser.add_argument('--pretrain', type=str, default=None)
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--eval_dataset', type=str, default=None)
    parser.add_argument('--max_datasets_size', type=int, default=None)
    parser.add_argument('--save_path', type=str, default='output')
    parser.add_argument('--need_optim_ckpt', type=bool, default=False)
    parser.add_argument('--max_epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--lora_rank', type=int, default=0, help="low-rank adaptation matrices rank")
    parser.add_argument('--log_interval', type=int, default=100, help="how many steps to log")
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--accumulation_steps', type=int, default=8)
    parser.add_argument('--use_wandb', default=False, action='store_true')
    parser.add_argument('--grad_checkpoint', default=False, action='store_true')
    args = parser.parse_args()
    
    train(args)