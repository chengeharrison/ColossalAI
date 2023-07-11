import json

def save_per_category(file_path):
    with open(file_path, 'r') as f:
        json_list = []
        for line in f:
            data = json.loads(line)
            print(data.keys())
            json_list.append(data)
        return json_list

save_per_category("moss-003-sft-no-tools.jsonl")