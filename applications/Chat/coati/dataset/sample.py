import random
import copy
from utils import jload, jdump
random.seed(42)

def sample_200k():
    res = []
    
    res.extend(random.sample(instruction_chinese_user_data, 51880))
    res.extend(instruction_chinese_alpaca_data)
    res.extend(instruction_english_data)
    
    res_more = copy.deepcopy(res)
    
    res.extend(conversation_chinese_less_data)
    res.extend(random.sample(conversation_english_less_data, 51880))
    
    res_more.extend(conversation_chinese_more_data)
    res_more.extend(random.sample(conversation_english_more_data, 51880))
    
    
    assert len(res) == 51880+48679+51880+8344+51880
    assert len(res_more) == 51880+48679+51880+14437+51880
    
    for idx, data in enumerate(res):
        data["id"] = idx+1
    
    for idx, data in enumerate(res_more):
        data["id"] = idx+1
    
    jdump(res, "final/train/200k_less.json")
    jdump(res_more, "final/train/200k_more.json")


def sample_40k():
    res = []
    
    res.extend(random.sample(instruction_chinese_user_data, 10240))
    res.extend(random.sample(instruction_chinese_alpaca_data, 10240))
    res.extend(random.sample(instruction_english_data, 10240))
    
    res_more = copy.deepcopy(res)
    
    res.extend(random.sample(conversation_chinese_less_data, 5120))
    res.extend(random.sample(conversation_english_less_data, 5120))
    
    res_more.extend(random.sample(conversation_chinese_less_data, 5120))
    res_more.extend(random.sample(conversation_english_more_data, 5120))
    
    
    assert len(res) == 40*1024
    assert len(res_more) == 40*1024
    
    for idx, data in enumerate(res):
        data["id"] = idx+1
    
    for idx, data in enumerate(res_more):
        data["id"] = idx+1
    
    jdump(res, "final/train/40k_less.json")
    jdump(res_more, "final/train/40k_more.json")
    

def sample_282k():
    res = []
    
    res.extend(random.sample(instruction_chinese_user_data, 51880))
    res.extend(instruction_chinese_alpaca_data)
    res.extend(instruction_english_data)
    res.extend(conversation_trans_cn_data)
    
    res_more = copy.deepcopy(res)
    
    res.extend(conversation_chinese_less_data)
    res.extend(random.sample(conversation_english_less_data, 51880))
    
    res_more.extend(conversation_chinese_more_data)
    res_more.extend(random.sample(conversation_english_more_data, 51880))
    
    
    assert len(res) == 272416
    assert len(res_more) == 282456
    
    for idx, data in enumerate(res):
        data["id"] = idx+1
    
    for idx, data in enumerate(res_more):
        data["id"] = idx+1
    
    jdump(res, "final/train/282k_less.json")
    jdump(res_more, "final/train/282k_more.json")


# print(len(jload("final/train/272k_less.json")))


instruction_chinese_alpaca = "preprocess/instruction/coati_instruction_filtered_Chinese_Alpaca-gpt4.json"
instruction_chinese_user = "preprocess/instruction/coati_instruction_filtered_Chinese_User-centered-instructions.json"
instruction_english = "preprocess/instruction/coati_instruction_filtered_English.json"
conversation_chinese_less = "preprocess/conversation/correct_split/less/coati_conversation_splitted_less_Chinese.json"
conversation_english_less = "preprocess/conversation/correct_split/less/coati_conversation_splitted_less_English.json"
# conversation_chinese_more = "preprocess/conversation/all_chinese/more/coati_conversation_splitted_more_Chinese.json"
# conversation_english_more = "preprocess/conversation/more/coati_conversation_splitted_more_English.json"

# same for less and more
conversation_trans_cn = "preprocess/conversation/correct_split/new_cn/sharegpt_cn_less.json"

hardcoded_mix = "/data/users/lcxyc/improve_sft/final/hardcoded.json"

instruction_chinese_alpaca_data = jload(instruction_chinese_alpaca)
instruction_chinese_user_data = jload(instruction_chinese_user)
instruction_english_data = jload(instruction_english)
conversation_chinese_less_data = jload(conversation_chinese_less)
conversation_english_less_data = jload(conversation_english_less)
# conversation_chinese_more_data = jload(conversation_chinese_more)
# conversation_english_more_data = jload(conversation_english_more)
conversation_trans_cn_data = jload(conversation_trans_cn)
# hardcoded_mix_data = jload(hardcoded_mix)

assert len(instruction_chinese_alpaca_data) == 48679
assert len(instruction_chinese_user_data) == 65287
assert len(instruction_english_data) == 51880
assert len(conversation_chinese_less_data) == 8104 + 3517
assert len(conversation_english_less_data) == 71720
# assert len(conversation_chinese_more_data) == 14437 + 7559
# assert len(conversation_english_more_data) == 118192
assert len(conversation_trans_cn_data) == 38358
# assert len(hardcoded_mix_data) == 1573

results = []
results.extend(instruction_chinese_alpaca_data)
results.extend(instruction_chinese_user_data)
results.extend(instruction_english_data)
results.extend(conversation_chinese_less_data)
results.extend(conversation_english_less_data)
results.extend(conversation_trans_cn_data)
# results.extend(hardcoded_mix_data)

assert len(results) == 287545

jdump(results, "/data/users/lcxyc/improve_sft/final/pheonix_data.json")

# sample_200k()
# sample_40k()
# sample_282k()

# instructwild_en = jload("instructwild_en.json")
# instructwild_cn = jload("instructwild_cn.json")

# res = []
# res.extend(random.sample(instructwild_en, 2048))
# res.extend(random.sample(instructwild_cn, 2048))

# results = []
# for idx, data in enumerate(res):
#     human_value = data["instruction"]+"\n"+data["input"] if data["input"] else data["instruction"]
#     conversation = [{'from':"human",'value':human_value},{'from':"gpt",'value':data["output"]}]
    
#     results.append({"conversations":conversation, "id": idx+1})
    
    
        
# jdump(results,"final/eval/eval.json")