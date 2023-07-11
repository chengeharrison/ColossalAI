import random
import os
import json
from utils import jload, jdump
from split_long_conversation import split_long_conversation
from filter_long_instruction import filter_long_instruction

def parse_id(id):
    id = id.split("] [")
    
    number = id[0][1:]
    type = id[1][6:]
    lang = id[2][6:]
    dataset_type = id[3][9:-1]
    
    return int(number), type.lower(), lang, dataset_type

def fill_in_dict(number, type, lang, dataset_type, conversations):
    return {"type":type, "language":lang, "dataset":dataset_type, "conversations":conversations, "id":number}

def save_data_per_type():
    data = jload("data.json")

    dataset_per_type={}
    types=set()
    for d in data:
        number, type, lang, dataset_type = parse_id(d['id'])
        
        if lang == 'zh':
            lang = 'Chinese'
        elif lang == 'en':
            lang = 'English'
            
        types.add(type)
        if type not in dataset_per_type:
            dataset_per_type[type]=[fill_in_dict(number, type, lang, dataset_type, d['conversations'])]
        else:
            dataset_per_type[type].append(fill_in_dict(number, type, lang, dataset_type, d['conversations']))

    for type in dataset_per_type:
        jdump(dataset_per_type[type], f"phoenix_{type}.json")
    
    print(types)

def check_instruction():
    data = jload("phoenix_instruction.json")
    
    question_person = set()
    answer_person = set()
    
    for d in data:
        assert len(d['conversations']) == 2, f"id:{d['id']}\n\n{d['conversations']}"
        question_person.add(d['conversations'][0]['from'])
        answer_person.add(d['conversations'][1]['from'])
        
    print(question_person)
    print(answer_person)
    
def save_instruction():
    data = jload("phoenix_instruction.json")
    
    def fill_in_dict(d):
        return {"type":d["type"], "language":d["language"], "dataset":d["dataset"], "conversations":d["conversations"], "id":d["id"]}
    
    datasets = {}
    res = []
    for d in data:
        tmp = fill_in_dict(d)
        res.append(tmp)
        
        if d["dataset"] not in datasets:
            datasets[d["dataset"]]=[tmp]
        else:
            datasets[d["dataset"]].append(tmp)
    
    language_per_dataset={dataset:{} for dataset in datasets}
    
    for dataset in datasets:
        for d in datasets[dataset]:
            if d["language"] not in language_per_dataset[dataset]:
                language_per_dataset[dataset][d["language"]]=1
            else:
                language_per_dataset[dataset][d["language"]]+=1
                
    # for d in datasets["User-centered-instructions"]:
    #     assert d["language"]=="zh"
    # print(datasets["User-centered-instructions"][20000])
    # exit()
                   
    jdump(res, "coati_instruction.json")
    
    # for dataset in datasets:
    #     print(f"{dataset}: {len(datasets[dataset])}")
        
    # for dataset in language_per_dataset:
    #     for language in language_per_dataset[dataset]:
    #         print(f"{dataset} {language}: {language_per_dataset[dataset][language]}")
    
    # User-centered-instructions: 65289
    # Alpaca-gpt4: 100559
    # Alpaca-gpt4-post-translation: 51398
    # Alpaca-gpt3.5-post-output: 49371
    
    # User-centered-instructions zh: 65289
    # Alpaca-gpt4 en: 51880
    # Alpaca-gpt4 zh: 48679
    # Alpaca-gpt4-post-translation multi-lingual: 51398
    # Alpaca-gpt3.5-post-output multi-lingual: 49371
    
    
def check_conversation():
    data=jload("phoenix_conversation.json")
    
    max_instructions = 0
    rounds=set()
    for d in data:
        length = len(d['conversations'])
        rounds.add(length)
        
        max_instructions+=length-1
        
        # if length %2 ==1 and length == 3:
        #     print(d)
        #     exit()
        
        # if d['instruction'][-2]['from'] == 'human':
        #     print(d['instruction'])
        #     exit()
    
    rounds=list(rounds)
    rounds.sort()
    print(rounds)
    
    print(f"total {len(data)} conversations")
    print(f"max {max_instructions} instructions")
    

def save_conversation():
    data = jload("phoenix_conversation.json")
    
    def fill_in_dict(d):
        return {"type":d["type"], "language":d["language"], "dataset":d["dataset"], "conversations":d["conversations"], "id":d["id"]}
    
    datasets = {}
    res = []
    for d in data:
        tmp = fill_in_dict(d)
        res.append(tmp)
        
        if d["dataset"] not in datasets:
            datasets[d["dataset"]]=[tmp]
        else:
            datasets[d["dataset"]].append(tmp)
    
    language_per_dataset={dataset:{} for dataset in datasets}
    
    for dataset in datasets:
        for d in datasets[dataset]:
            if d["language"] not in language_per_dataset[dataset]:
                language_per_dataset[dataset][d["language"]]=1
            else:
                language_per_dataset[dataset][d["language"]]+=1
                
    # for d in datasets["User-centered-instructions"]:
    #     assert d["language"]=="zh"
    # print(datasets["User-centered-instructions"][20000])
    # exit()
                   
    jdump(res, "coati_conversation.json")
    
    for dataset in datasets:
        print(f"{dataset}: {len(datasets[dataset])}")
        
    for dataset in language_per_dataset:
        for language in language_per_dataset[dataset]:
            print(f"{dataset} {language}: {language_per_dataset[dataset][language]}")
    
    # ShareGPT/Discord: 197893
    
    # ShareGPT/Discord multi-lingual: 197893
    

def save_instruction_per_lang():
    data=jload("coati_instruction_filtered.json")
    
    dataset_per_lang={}
    for d in data:
        lang=d['language']
        
        if lang not in ["Chinese", "English"]:
            continue
            
        if lang not in dataset_per_lang:
            dataset_per_lang[lang]=[d]
        else:
            dataset_per_lang[lang].append(d)

    for lang in dataset_per_lang:
        print(f"{lang}: {len(dataset_per_lang[lang])}")
        jdump(dataset_per_lang[lang], f"preprocess/coati_instruction_filtered_{lang}.json")
    
        
# save_data_per_type()
# check_instruction()
# save_instruction()
# check_conversation()
# save_conversation()
# split_long_conversation("coati_conversation.json", "coati_conversation_splitted_more.json")
# filter_long_instruction("coati_instruction.json", "coati_instruction_filtered.json")

# save_instruction_per_lang()
# split_long_conversation("sharegpt_cn.json","preprocess/conversation/new_cn/sharegpt_cn_more.json")


def analyze_new_conversation():
    file_path = os.path.expanduser("sharegpt_zh_38K.json")
    with open(file_path, "r") as f:
        data = []
        for line in f:
            data.append(json.loads(line))
    
    results = []
    
    for d in data:
        item={"type": "conversation",
        "language": "Chinese",
        "dataset": "ShareGPT-post-translation",
        "conversations": d["conversations"],
        "id": d['id']}
        
        results.append(item)
        
    jdump(results, "sharegpt_cn.json")

# analyze_new_conversation()
# data = jload("sharegpt_cn.json")
# print(len(data))
# split_long_conversation("sharegpt_cn.json","preprocess/conversation/correct_split/new_cn/sharegpt_cn_more.json")


# for i in range(14, 29):
#     split_long_conversation(f"/data/users/lcxyc/improve_sft/ultra-chat/ultra-chat-{i}.json",f"/data/users/lcxyc/improve_sft/ultra-chat-split/ultra-chat-split-{i}.json")
# split_long_conversation("/data/users/lcxyc/improve_sft/belle-chat.json","/data/users/lcxyc/improve_sft/belle-chat-split.json")
split_long_conversation("/data/users/lcxyc/improve_sft/final/final_dataset_needed_split.json","/data/users/lcxyc/improve_sft/final/final_dataset.json")

