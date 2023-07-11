import os
import dataclasses
import argparse
from enum import auto, Enum
import torch
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from chatio import simple_io, rich_io
from auto_gptq import AutoGPTQForCausalLM

from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

def get_gpu_memory(max_gpus=None):
    gpu_memory = []
    num_gpus = (
        torch.cuda.device_count()
        if max_gpus is None
        else min(max_gpus, torch.cuda.device_count())
    )

    for gpu_id in range(num_gpus):
        with torch.cuda.device(gpu_id):
            device = torch.cuda.current_device()
            gpu_properties = torch.cuda.get_device_properties(device)
            total_memory = gpu_properties.total_memory / (1024 ** 3)
            allocated_memory = torch.cuda.memory_allocated() / (1024 ** 3)
            available_memory = total_memory - allocated_memory
            gpu_memory.append(available_memory)
    return gpu_memory


def load_model(
        model_path, device="cuda", num_gpus=1, max_gpu_memory=None, load_8bit=False, load_4bit=False, debug=False
):
    if device == "cpu":
        kwargs = {}
    elif device == "cuda":
        kwargs = {"torch_dtype": torch.float16}
        if num_gpus == "auto":
            kwargs["device_map"] = "auto"
        else:
            num_gpus = int(num_gpus)
            if num_gpus != 1:
                kwargs["device_map"] = "auto"
                if max_gpu_memory is None:
                    kwargs[
                        "device_map"
                    ] = "sequential"  # This is important for not the same VRAM sizes
                    available_gpu_memory = get_gpu_memory(num_gpus)
                    kwargs["max_memory"] = {
                        i: str(int(available_gpu_memory[i] * 0.85)) + "GiB"
                        for i in range(num_gpus)
                    }
                else:
                    kwargs["max_memory"] = {i: max_gpu_memory for i in range(num_gpus)}
        # print("init_kwargs", kwargs)
    elif device == "mps":
        kwargs = {"torch_dtype": torch.float16}
        # Avoid bugs in mps backend by not using in-place operations.
        replace_llama_attn_with_non_inplace_operations()
    else:
        raise ValueError(f"Invalid device: {device}")

    if "google/flan-t5" in model_path:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    elif "dolly" in model_path:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
        tokenizer.eos_token_id = 50277  # 50277 means "### End"
    elif "pythia" in model_path or "stablelm" in model_path:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    elif "phoenix" in model_path:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        if load_4bit:
            model = AutoGPTQForCausalLM.from_quantized(model_path, device, use_triton=True)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    if load_8bit:
        compress_module(model, device)

    if (device == "cuda" and num_gpus == 1) or device == "mps":
        model.to(device)

    if debug:
        print(model)

    return model, tokenizer

class SeparatorStyle(Enum):
    ADD_BOS_EOS_TOKEN = auto()
    TWO = auto()


@dataclasses.dataclass
class Conversation:
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.ADD_BOS_EOS_TOKEN
    sep: str = "</s>"

    skip_next: bool = False
    
    def clear(self):
        self.messages=[]

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.ADD_BOS_EOS_TOKEN:
            ret = self.system
            for role, message in self.messages:
                if message:
                    ret += role + ": " + "<s>" + message + "</s>"
                else:
                    ret += role + ": " + "<s>"
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")
    
    def save_prompt(self):
        if self.sep_style == SeparatorStyle.ADD_BOS_EOS_TOKEN:
            ret = self.system
            for role, message in self.messages:
                if message:
                    ret += role + ": " + "<s>" + message + "</s>\n"
                else:
                    ret += role + ": " + "<s>"
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep)

    def dict(self):
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep
        }

@dataclasses.dataclass
class Conversation_last6:
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.ADD_BOS_EOS_TOKEN
    sep: str = "</s>"

    skip_next: bool = False
    
    def clear(self):
        self.messages=[]

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.ADD_BOS_EOS_TOKEN:
            ret = self.system
            for role, message in self.messages:
                if message:
                    ret += role + ": " + "<s>" + message + "</s>"
                else:
                    ret += role + ": " + "<s>"
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")
    
    def save_prompt(self):
        if self.sep_style == SeparatorStyle.ADD_BOS_EOS_TOKEN:
            ret = self.system
            for role, message in self.messages:
                if message:
                    ret += role + ":\n" + message + "\n\n"
                else:
                    ret += role + ":\n"
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep)

    def dict(self):
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep
        }
        
conv = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n",
    roles=("Human", "Assistant"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.ADD_BOS_EOS_TOKEN,
    sep="</s>",
)

conv_last6 = Conversation_last6(
    system="Below is an instruction that describes a task. Write a response that appropriately completes the request. Do not generate new instructions.\n\n",
    roles=("Instruction", "Response"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.ADD_BOS_EOS_TOKEN,
    sep="</s>",
)

def get_default_conv_template(model_name=None):
    if model_name is None:
        return default_conversation
    model_name = model_name.lower()
    if "phoenix" in model_name or "chimera" in model_name:
        return default_conversation
    else:
        raise NotImplementedError

default_conversation = conv
conv_templates = {"default": conv}


@torch.inference_mode()
def generate(
    model, tokenizer, prompt, device, generation_kwargs, round, context_len=2048, stream_interval=2
):
    temperature = float(generation_kwargs.get("temperature", 0.0))
    max_new_tokens = int(generation_kwargs.get("max_new_tokens", 256))
    stop_str = generation_kwargs.get("stop", "</s>")
    stop_token_ids = generation_kwargs.get("stop_ids", [tokenizer.eos_token_id])
    

    input_ids = tokenizer(prompt).input_ids
    output_ids = list(input_ids)
    
    l_prompt = len(tokenizer.decode(input_ids, skip_special_tokens=False))

    if model.config.is_encoder_decoder:
        max_src_len = context_len
    else:
        max_src_len = context_len - generation_kwargs.get("max_new_tokens", 256) - 15

    input_ids = input_ids[-max_src_len:]
    
    with open("input.txt", mode='a', encoding="utf-8") as f:
        f.write("\n\n" + "=" * 10 + "\n")
        f.write(f"Input for round {round}:\n{tokenizer.decode(input_ids)}\n\n")
        f.write("=" * 10 + "\n")
    
    # output = model.generate(torch.as_tensor([input_ids], device=device), **generation_kwargs)
    
    for i in range(max_new_tokens):
        if i == 0:
            if model.config.is_encoder_decoder:
                encoder_outputs = model.encoder(
                    input_ids=torch.as_tensor([input_ids], device=device)
                )
                out = model(
                    torch.as_tensor([input_ids], device=device),
                    decoder_input_ids=torch.as_tensor(
                        [[model.generation_config.decoder_start_token_id]],
                        device=device,
                    ),
                    encoder_outputs=encoder_outputs,
                    use_cache=True,
                )
                logits = out.logits
                past_key_values = out.past_key_values
            else:
                out = model(torch.as_tensor([input_ids], device=device), use_cache=True)
                logits = out.logits
                past_key_values = out.past_key_values
        else:
            if model.config.is_encoder_decoder:
                out = model(
                    input_ids=torch.as_tensor([input_ids], device=device),
                    use_cache=True,
                    encoder_outputs=encoder_outputs,
                    decoder_input_ids=torch.as_tensor([[token]], device=device),
                    past_key_values=past_key_values,
                )
                logits = out.logits
                past_key_values = out.past_key_values
            else:
                out = model(
                    input_ids=torch.as_tensor([[token]], device=device),
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                logits = out.logits
                past_key_values = out.past_key_values

        last_token_logits = logits[0][-1]
        
        if device == "mps":
            # Switch to CPU by avoiding some bugs in mps backend.
            last_token_logits = last_token_logits.float().to("cpu")

        if temperature < 1e-4:
            token = int(torch.argmax(last_token_logits))
        else:
            probs = torch.softmax(last_token_logits / temperature, dim=-1)
            token = int(torch.multinomial(probs, num_samples=1))

        output_ids.append(token)

        if token in stop_token_ids:
            stopped = True
        else:
            stopped = False

        if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
            output = tokenizer.decode(output_ids, skip_special_tokens=False)
            if stop_str:
                pos = output.rfind(stop_str, l_prompt)
                if pos != -1:
                    output = output[l_prompt:pos]
                    stopped = True
                else:
                    output = output[l_prompt:]
                yield output
            else:
                raise NotImplementedError

        if stopped:
            break

    del past_key_values
    
    # answer = ""
    # # for tok_id in output[0][len(input_ids):]:
    # #     if tok_id != tokenizer.eos_token_id:
    # #         answer += tokenizer.decode(tok_id)
    # answer = tokenizer.decode(output[0][len(input_ids):], skip_special_tokens=True)
    
    # return answer

    
    
def main(args):
    max_new_tokens = args.max_new_tokens
    model_max_length = args.model_max_length
    
    # model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16).to(torch.cuda.current_device())
    # model.eval()

    # tokenizer = AutoTokenizer.from_pretrained(
    #     args.model_path,
    #     model_max_length=model_max_length,
    #     padding_side="left",
    #     truncation_side='left',
    #     padding=True,
    #     truncation=True,
    #     use_fast=False
    # )
    model, tokenizer = load_model(
        args.model_path
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
        # 'temperature':0.1,
    }
    
    if "last6" in args.model_path:
        conv = conv_last6
    else:
        conv = default_conversation
        
    roles = conv.roles
    round = 1
    
    while True:
        chat_io = simple_io if args.io=="simple" else rich_io
        # raw_text = print(">>> Human:", end=" ")
        inp = chat_io.prompt_for_input(conv.roles[0])
        
        if not inp:
            print('prompt should not be empty!')
            continue
        
        if inp.strip() == "clear":
            conv.clear()
            os.system("clear")
            continue
        
        if inp.strip() == "exit":
            print('End of chat.')
            break
    
        query_text = inp.strip()
        
        conv.append_message(roles[0], query_text)
        conv.append_message(roles[1], None)
        
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
        generate_stream_func = generate
        chat_io.prompt_for_output(conv.roles[1])
        output_stream = generate_stream_func(model, tokenizer, conv.get_prompt(), round=round, context_len=model_max_length, generation_kwargs=model_kwargs, device=torch.cuda.current_device(), stream_interval=2)
        # exit()
        
        # print(f">>> Assistant:", end=" ")
        outputs = chat_io.stream_output(output_stream)
        
        conv.messages[-1][-1]=outputs.strip()
        
        with open("round.txt", mode='a', encoding="utf-8") as f:
            f.write("\n\n" + "=" * 10 + "\n")
            f.write(f"round {round}:\n{conv.save_prompt()}\n\n")
            f.write("=" * 10 + "\n")
        
        # print(f">>> Assistant:", end=" ")
        
        round+=1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--model_max_length', type=int, default=2048)
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--io', type=str, default="rich")
    args = parser.parse_args()
    main(args)