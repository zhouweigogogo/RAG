import json

with open('./NER_lora_train.json','r') as f:
    data = json.load(f)

print(len(data))