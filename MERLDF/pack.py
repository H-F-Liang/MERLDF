import random
import pandas as pd


def ReadData(path):
    df = pd.read_csv(path, sep='|')
    # 动作空间
    res_data = []
    # 动作空间
    action_space = set()
    labels = set()
    symptom_dic = {}
    for data in df.iterrows():
        labels.add(data[1]['label'])
        res_data.append({
            'label': data[1]['label'],
            'explicit_symptom': str(data[1]['explicit_symptom']).split('，'),
            'none_symptom': str(data[1]['none_symptom']).split('，'),
            'implicit_symptom': str(data[1]['implicit_symptom']).split('，'),
            'true_symptom': str(data[1]['true_symptom']).split('，'),
        })
        t = str(data[1]['explicit_symptom']).split('，') + str(data[1]['none_symptom']).split('，') + str(
            data[1]['implicit_symptom']).split('，') + str(data[1]['true_symptom']).split('，')
        for i in t:
            if i not in symptom_dic.keys():
                symptom_dic[i] = 1
            else:
                symptom_dic[i] += 1
        action_space = action_space | (set(str(data[1]['explicit_symptom']).split('，')) | set(
            str(data[1]['none_symptom']).split('，')) | set(str(data[1]['implicit_symptom']).split('，')) | set(
            str(data[1]['true_symptom']).split('，')))

    for key in symptom_dic.keys():
        if symptom_dic[key] < 1 or key == 'nan':
            if key in list(action_space):
                action_space.remove(key)
            for index, item in enumerate(res_data):
                if len(res_data[index]['explicit_symptom']) != 0 and key in res_data[index]['explicit_symptom']:
                    res_data[index]['explicit_symptom'].remove(key)
                if len(res_data[index]['none_symptom']) != 0 and key in res_data[index]['none_symptom']:
                    res_data[index]['none_symptom'].remove(key)
                if len(res_data[index]['implicit_symptom']) != 0 and key in res_data[index]['implicit_symptom']:
                    res_data[index]['implicit_symptom'].remove(key)
                if len(res_data[index]['true_symptom']) != 0 and key in res_data[index]['true_symptom']:
                    res_data[index]['true_symptom'].remove(key)

    return res_data, list(action_space), list(labels)


# 里面包含了对脏数据的处理，仅处理mz10数据集有作用
def ProcessTrainData(train_path, test_path):
    train_data, train_action_space, train_labels = ReadData(train_path)
    test_data, test_action_space, test_labels = ReadData(test_path)
    # print(len(train_action_space), len(test_action_space), len(list(set(train_action_space) | set(test_action_space))))
    # 去掉测试集里有但是训练集中没出现过的症状的样本(脏数据)
    t_data = []
    s_data = []
    for data in test_data:
        symptoms = set(data['explicit_symptom']) | set(data['none_symptom']) | set(
            data['implicit_symptom']) | set(data['true_symptom'])
        f = True
        for s in symptoms:
            if s not in train_action_space:
                f = False
                # 把所有训练集未出现过的症状从动作空间删除
                s_data.append(s)
        if f:
            t_data.append(data)
    test_data = t_data
    action_space = list(set(train_action_space) | set(test_action_space))
    for s in s_data:
        if s in action_space:
            action_space.remove(s)
    action_space.sort()
    labels = list(set(train_labels) | set(test_labels))
    labels.sort()

    random.shuffle(train_data)
    random.shuffle(test_data)
    return train_data, test_data, action_space, labels

