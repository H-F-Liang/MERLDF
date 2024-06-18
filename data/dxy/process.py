import json
import pandas as pd

dataset = open('test_set.json', 'r', encoding='utf-8')
dataset = dataset.read()

res_train = []
dic = json.loads(dataset)
for item in dic:
    sample = {
        'label': item['label'],
        'explicit_symptom': '',
        'none_symptom': '',
        'implicit_symptom': '',
        'true_symptom': ''
    }
    for k in item.keys():
        if k in ['exp_sxs', 'imp_sxs']:
            # 遍历症状：key就是一个个症状
            for key in item[k].keys():
                if item[k][key] == '1' and k == 'exp_sxs':
                    sample['explicit_symptom'] += ('，' if len(sample['explicit_symptom']) else '') + key
                elif item[k][key] == '1' and k != 'exp_sxs' and key not in item['exp_sxs'].keys():
                    sample['true_symptom'] += ('，' if len(sample['true_symptom']) else '') + key
                elif item[k][key] != '1':
                    if item[k][key] == '0' and key not in item['exp_sxs'].keys():
                        sample['none_symptom'] += ('，' if len(sample['none_symptom']) else '') + key
                    elif key not in item['exp_sxs'].keys():
                        sample['implicit_symptom'] += ('，' if len(sample['implicit_symptom']) else '') + key
    # print(item)
    # print(sample)
    res_train.append(sample)

df = pd.DataFrame(res_train)
df.to_csv('test_set.csv', sep='|', index=False)
