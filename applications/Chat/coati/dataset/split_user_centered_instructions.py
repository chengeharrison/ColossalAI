from utils import jload, jdump

def save_data_per_dataset():
    data = jload("preprocess/instruction/coati_instruction_filtered_Chinese.json")

    dataset_per_dataset={}
    
    for d in data:
        dataset_type = d['dataset']
        if dataset_type not in dataset_per_dataset:
            dataset_per_dataset[dataset_type]=[d]
        else:
            dataset_per_dataset[dataset_type].append(d)

    for dataset_type in dataset_per_dataset:
        print(f"{dataset_type}: {len(dataset_per_dataset[dataset_type])}")
        jdump(dataset_per_dataset[dataset_type], f"preprocess/instruction/coati_instruction_filtered_Chinese_{dataset_type}.json")
    
    # print(types)

save_data_per_dataset()