import transformers
import torch
from utils import jload, jdump
from transformers import pipeline
import os
import tqdm
import fasttext
os.environ['CUDA_VISIBLE_DEVICES'] = "7" 

def content(d, tokenizer):
    string=""
    
    for sentence in d['conversations']:
        string += sentence['from'] + ": " + sentence['value'] + "\n"
        
    # print(string)
    # string=string.replace("\n", " ")
    # print(tokenizer.decode(tokenizer.encode(string)[0:512]))
    # exit()
    
    return tokenizer.decode(tokenizer.encode(string)[0:128])

# model = fasttext.load_model("/data/users/lcxyc/improve_sft/fasttext-language-identification/model.bin")
# print(model.predict("Hello, world!")[0][0])
# exit()

# [{'label': 'English', 'score': 0.9802148342132568}]

def save(less_or_more, p):
    data=jload(f"coati_conversation_splitted_{less_or_more}.json")

    # labels = {}

    batch_size = 1

    data_batched = [data[i:i*batch_size] for i in range(0, len(data), batch_size)]

    res = {"Simplified Chinese":[],"Traditional Chinese":[], "English":[]}
    for idx, d in tqdm.tqdm(enumerate(data), total=len(data)):
        # batch_ = [content(d, tokenizer) for d in batch]
    
        label = p(content(d, tokenizer))
        # print(label)
        for l in label:
            if l['label'] == 'Mandarin Chinese':
                d["language"] = "Simplified Chinese"
                res["Simplified Chinese"].append(d)
            elif l['label'] == 'Cantonese Chinese':
                d["language"] = "Traditional Chinese"
                res["Traditional Chinese"].append(d)
            elif l['label'] == 'English':
                d["language"] = "English"
                res["English"].append(d)
    
        # if idx == 1000:
        #     break

    print(f"{less_or_more} Simplified Chinese: {len(res['Simplified Chinese'])}")
    print(f"{less_or_more} Traditional Chinese: {len(res['Traditional Chinese'])}")
    
    tmp = []
    tmp.extend(res["Simplified Chinese"])
    tmp.extend(res["Traditional Chinese"])
    
    jdump(tmp, f"preprocess/conversation/correct_split/{less_or_more}/coati_conversation_splitted_{less_or_more}_Chinese.json")

    print(f"{less_or_more} English: {len(res['English'])}")
    jdump(res["English"], f"preprocess/conversation/correct_split/{less_or_more}/coati_conversation_splitted_{less_or_more}_English.json")

model_id = "/data/users/lcxyc/improve_sft/xlm-v-base-language-id/"
tokenizer = transformers.AutoTokenizer.from_pretrained(
        "facebook/xlm-v-base",
        model_max_length=512,
        use_fast=False,
        max_length=512,
        truncation=True,
    )

p = pipeline("text-classification", model=model_id, tokenizer=tokenizer, device=torch.cuda.current_device())
# p("Hello world")
# print(p("你好，我来自中国"))


save("less", p)
save("more", p)


# print(labels)