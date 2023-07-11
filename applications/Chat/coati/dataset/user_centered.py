import os
from utils import jdump, jload
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
        
        if "User" not in dataset_type:
            continue
        
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
        jdump(dataset_per_type[type][0:10000], f"phoenix_{type}_10000.json")
    
    print(types)
    
save_data_per_type()