import json
import sys
import os
import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer

prompt = """你是一个问题分类器。对于每个提供给你的问题，你需要猜测出该问题是属于文本理解任务还是在SQL查询任务,如果是文本理解任务，则需要返回判断类型结果以及问题中涉及的公司名称以及该问题的语义关键词；如果是SQL查询任务，则只需要返回判断类型结果即可，无需返回公司名称以及关键词。以下是一些例子：\n
问题：“在2019年的中期报告里，XX基金管理有限公司管理的基金中，有多少比例的基金是个人投资者持有的份额超过机构投资者？希望得到一个精确到两位小数的百分比。回答：《SQL查询任务》\n
问题：“XXXX股份有限公司变更设立时作为发起人的法人有哪些？回答：《文本理解任务，公司名称：XXXX股份有限公司，关键词：变更设立时作为发起人的法人有哪些》\n”
问题：“我想知道XXXXXX债券A基金在20200930的季报中，其可转债持仓占比最大的是哪个行业？用申万一级行业来统计。”回答：《SQL查询任务》\n
问题：“XXXXXX股份有限公司2020年增资后的投后估值是多少？”回答：《文本理解任务，公司名称：XXXXXX股份有限公司，关键词：2020年增资后的投后估值是多少》\n
问题：根据XXXXXX股份有限公司招股意向书，全球率先整体用LED路灯替换传统路灯的案例是？”回答：《文本理解任务，公司名称：XXXXXX股份有限公司，关键词：全球率先整体用LED路灯替换传统路灯的案例》\n
问题：什么公司、在何时与XXXXXX股份有限公司发生了产品争议事项？产品争议事项是否已经解决？回答：《文本理解任务，公司名称：XXXXXX股份有限公司，关键词：什么公司、在何时与之发生了产品争议事项，是否已经解决》\n
问题：请问XXXX年一季度有多少家基金是净申购?它们的净申购份额加起来是多少?请四舍五入保留小数点两位。回答：《SQL查询任务》\n
问题：我想知道股票XXXXXX在申万行业分类下的二级行业是什么？用最新的数据。回答：《SQL查询任务》\n
问题：截止2005年12月31日，南岭化工厂的总资产和净资产分别是多少？回答：《文本理解任务，公司名称：南岭化工厂，关键词：截止2005年12月31日的总资产和净资产》\n
问题：XXXXXX股份有限公司的中标里程覆盖率为多少？回答：《文本理解任务，公司名称：XXXXXX股份有限公司，关键词：中标里程覆盖率》\n
问题：各报告期末，XXXXX股份有限公司存货分别为多少万元？占XX资产的比例分别多少?回答：《文本理解任务，公司名称：XXXXX股份有限公司，关键词：各报告期末的存货以及占XX资产的比例》\n
请你根据上面提供的例子，对当前用户问题类型进行分类，输出标准请严格按照上面提示要求，\n
当判断问题为文本理解类型时，你的严格输出格式为:《任务类型结果，公司名称：，关键词：》；当判断问题为SQL查询任务类型时，你的严格输出格式为:《任务类型结果》\n
请直接输出结果，不可以输出其他的文字。一般情况下，如果问题中涉及到基金以及股票时都是SQL查询任务。"""



if __name__ == '__main__':

    data_path = '/home/shared/class/zhouw/RAG/example2/data/question.json'
    device = "cuda:0"

    with open(data_path, 'r') as f:
        question = [eval(js) for js in f.readlines()]
    # print(question)

    model_base = '/home/shared/class/zhouw/mymodel/qwen2.5-3B-instruct'
    tokenizer = AutoTokenizer.from_pretrained(model_base)
    model = AutoModelForCausalLM.from_pretrained(model_base, device_map="auto", trust_remote_code=True)

    results = []

    for q in question:
        q_text = q['question']
        messages = [
            {"role":"system", "content":prompt},
            {'role':"user", "content":q_text}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = tokenizer([text], return_tensors="pt").to(device)
        generated_ids = model.generate(**inputs, max_new_tokens=512)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        print(response)

        results.append([q_text, response])

    with open('class_q_text.txt','w',encoding='utf-8') as f:
        for line in results:
            f.write(f"{line[0]} {line[1]}\n")

    





    