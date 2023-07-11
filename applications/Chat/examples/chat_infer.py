import os
import dataclasses
import argparse
import torch
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

PROMPT_DICT = {
    "prompt_input":
        ("Below is an instruction that describes a task, paired with an input that provides further context. "
         "Write a response that appropriately completes the request.\n\n"
         "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"),
    "prompt_no_input": ("Below is an instruction that describes a task. "
                        "Write a response that appropriately completes the request.\n\n"
                        "### Instruction:\n{instruction}\n\n### Response:"),
}

def generate(
    model, tokenizer, prompt, device, generation_kwargs, round, context_len=2048, 
):
    len_prompt = len(prompt)

    input_ids = tokenizer(prompt).input_ids
    input_echo_len = len(input_ids)
    output_ids = list(input_ids)
    
    output = model.generate(torch.as_tensor([input_ids], device=device), **generation_kwargs)
    
    answer = tokenizer.decode(output[0], skip_special_tokens=True).split("### Response:")[1]
    
    return answer

    
    
def main(args):
    max_new_tokens = args.max_new_tokens
    model_max_length = args.model_max_length
    
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16).to(torch.cuda.current_device())
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        model_max_length=model_max_length,
        padding_side="left",
        truncation_side='left',
        padding=True,
        truncation=True,
        use_fast=False
    )
    
    assert max_new_tokens <= model_max_length
    
    # if not tokenizer.bos_token_id:
    #     tokenizer.bos_token_id = "<s>"
    if not tokenizer.eos_token_id:
        tokenizer.eos_token_id = "</s>"
    # if not tokenizer.pad_token_id:
    #     tokenizer.pad_token_id = "</s>"

    model_kwargs = {
        'max_new_tokens': max_new_tokens,
        # 'early_stopping': True,
        # 'top_k': -1,
        # 'top_p': 1.0,
        # 'temperature': 1.0,
    }
    
    round = 1
    
    while True:
        raw_text = input(">>> Human: ")
        if not raw_text:
            print('prompt should not be empty!')
            continue
        
        if raw_text.strip() == "clear":
            conv.clear()
            continue
        
        if raw_text.strip() == "exit":
            print('End of chat.')
            break
    
        query_text = raw_text.strip()
        
        # inputs = tokenizer(conv.get_prompt(), return_tensors='pt', truncation=True, max_length=args.model_max_length-args.max_new_tokens).to(torch.cuda.current_device())
        # input_ids = tokenizer(conv.get_prompt())["input_ids"]
        # attention_mask = tokenizer(conv.get_prompt())["attention_mask"]
        # output_ids = list(input_ids)
        
        # l_prompt = len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False))

        # max_src_len = model_max_length - max_new_tokens - 15
        
        # inputs = {k: v[:][-max_src_len:] for k, v in inputs.items()}
        
        # output = model.generate(**inputs, **generation_kwargs)
        
        # answer = ""
        # for tok_id in output[0][inputs['input_ids'].shape[1]:]:
        #     if tok_id != tokenizer.eos_token_id:
        #         answer += tokenizer.decode(tok_id)
        
        

        # sess_text += answer + tokenizer.eos_token_id
        
        prompt = PROMPT_DICT["prompt_no_input"].format(instruction=query_text)
        
        answer = generate(model, tokenizer, prompt, round=round, context_len=model_max_length, generation_kwargs=model_kwargs,device=torch.cuda.current_device())
        
        print(f">>> Assistant: {answer}")
        
        round+=1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--model_max_length', type=int, default=2048)
    parser.add_argument('--max_new_tokens', type=int, default=512)
    args = parser.parse_args()
    main(args)