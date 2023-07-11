import argparse
from concurrent.futures import ProcessPoolExecutor
from utils import jload, jdump
from typing import Dict, Sequence, Optional

import transformers
from tqdm import tqdm


def make_sample(sample, start_idx, end_idx):
    assert (end_idx - start_idx) % 2 == 0
    
    return {
        "type":sample["type"], "language":sample["language"], "dataset":sample["dataset"], "conversations":sample["conversations"][start_idx:end_idx], "id":sample["id"]
    }


tokenizer = max_length = None


def split_one_sample(sample):
    tokenized_lens = []
    conversations = sample["conversations"]
    # conversations = conversations[: len(conversations) // 2 * 2]
    for c in conversations:
        length = len(tokenizer(c["value"]).input_ids) + 15
        tokenized_lens.append(length)

    start_idx = 0
    cur_len = 0

    # if len(conversations) % 2 != 0 or len(conversations) < 2:
    #     return []
    if len(conversations) < 2:
        return []

    new_samples = []
    for i in range(0, len(conversations), 2):
        tmp_len = tokenized_lens[i] + tokenized_lens[i + 1] if i+1<len(conversations) else tokenized_lens[i]
        if cur_len + tmp_len > max_length:
            new_samples.append(make_sample(sample, start_idx, i))
            break
            start_idx = i
            cur_len = 0
        elif i == len(conversations) - 2:
            new_samples.append(make_sample(sample, start_idx, i + 2))
            break

        cur_len += tmp_len
        
    # print(len(new_samples))
    
    assert len(new_samples) == 1 or len(new_samples) == 0

    return new_samples


def split_all(content, tokenizer_, max_length_):
    """
    Keep the maximum round of conversations within the max token length constraint
    """
    global tokenizer, max_length
    tokenizer = tokenizer_
    max_length = max_length_

    new_content = []

    with ProcessPoolExecutor() as executor:
        for result in tqdm(executor.map(split_one_sample, content), total=len(content)):
            new_content.extend(result)

    return new_content


def filter_invalid_roles(content):
    new_content = []
    for i, c in enumerate(content):
        roles = ["human", "gpt"]
        if len(c["conversations"]) <= 0:
            continue

        valid = True
        # roles = ["human", "gpt"] if c["conversations"][0]['from']=='human' else ["gpt", "human"]
        # if c["conversations"][0]['from']=='gpt':
        #     print(True)
        for j, s in enumerate(c["conversations"]):
            if s["from"] != roles[j % 2]:
                valid = False
                break

        if valid:
            new_content.append(c)

    return new_content


def split_long_conversation(in_file, out_file, model_name_or_path="/data/scratch/LLaMa-7B/", max_length=2048):
    content = jload(in_file)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        model_max_length=max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.bos_token="<s>"
    tokenizer.eos_token="</s>"
    new_content = split_all(content, tokenizer, max_length)
    new_content = filter_invalid_roles(new_content)

    print(f"total: {len(content)}, new: {len(new_content)}")
    jdump(new_content, out_file)


# tokenizer = transformers.AutoTokenizer.from_pretrained(
#         "/data/scratch/LLaMa-7B/",
#         model_max_length=2048,
#         padding_side="right",
#         use_fast=False,
#     )
# tokenizer.bos_token="<s>"
# tokenizer.eos_token="</s>"
# print(len(tokenizer("Human: <s> </s> \n").input_ids))
# print(len(tokenizer("Assistant: <s> </s> \n").input_ids))