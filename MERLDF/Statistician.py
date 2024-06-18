import torch
from scipy import stats, integrate


class Statistician(object):
    def __init__(self, dataset, action_space, label_space):
        self.tau = 0.1
        # 窗口大小
        self.window_size = len(dataset)
        self.action_space = action_space
        self.label_space = label_space
        # 统计数据集中症状-疾病的真实分布
        # 为了实时进行修改需要对当前实际的选择进行记录，后续同一个样本出现时再进行替换
        # 疾病对应的症状
        self.label_symptoms = {}
        # 数据集中疾病下症状数量的统计
        self.label_table = {}
        self.action_table = {}
        # 记录agent再过完一遍数据集后所有做出的动作
        self.memery = []
        self.__build__(dataset)
        # self.sta(dataset)

    def __build__(self, dataset):
        for data in dataset:
            label = data['label']
            if label not in self.label_symptoms.keys():
                self.label_symptoms[label] = {
                    'true_symptom': [],
                    'implicit_symptom': [],
                    'none_symptom': []
                }
            else:
                self.label_symptoms[label]['true_symptom'] = list(set(self.label_symptoms[label]['true_symptom']) | set(data['explicit_symptom']) | set(data['true_symptom']))
                self.label_symptoms[label]['implicit_symptom'] = list(set(self.label_symptoms[label]['implicit_symptom']) | set(data['implicit_symptom']))
                self.label_symptoms[label]['none_symptom'] = list(set(self.label_symptoms[label]['none_symptom']) | set(data['none_symptom']))
            symptoms = list(set(data['explicit_symptom']) | set(
                data['implicit_symptom']) | set(data['true_symptom']) | set(data['none_symptom']))
            # symptoms = list(set(data['explicit_symptom']) | set(data['true_symptom']) | set(data['none_symptom']))
            # symptoms = list(set(data['explicit_symptom']) | set(data['true_symptom']))

            if label not in self.label_table.keys():
                self.label_table[label] = {
                    'num': 1,
                    'symptoms': {}
                }
            else:
                self.label_table[label]['num'] += 1

            for symptom in symptoms:
                # 初始化label_table中的
                if symptom not in self.label_table[label]['symptoms'].keys():
                    self.label_table[label]['symptoms'][symptom] = 0
                self.label_table[label]['symptoms'][symptom] += 1  # if symptom not in data['explicit_symptom'] else 0

    # def sta(self, dataset):
    #     # 统计各个标签所占的比例
    #     label_dict = {}
    #     action_dict = {}
    #     for label in self.label_space:
    #         label_dict[label] = 0
    #     for action in self.action_space:
    #         action_dict[action] = 0
    #
    #     for data in dataset:
    #         label = data['label']
    #         label_dict[label] += 1
    #         symptoms = list(set(data['explicit_symptom']) | set(data['implicit_symptom']) | set(data['true_symptom']) | set(data['none_symptom']))
    #         for symptom in symptoms:
    #             action_dict[symptom] += 1
    #     data = sorted(label_dict.items(), key=lambda x: x[1], reverse=True)
    #     print(data)
    #     data = sorted(action_dict.items(), key=lambda x: x[1], reverse=True)
    #     s = sum(action_dict.values())
    #     print(s)
    #     t = 0
    #     for i, d in enumerate(data):
    #         t += d[1]
    #         if (i + 1) % 5 == 0:
    #             print('Top-{}:'.format(i + 1) + str(t / s * 100) + '%')
    #     # # pprint.pprint()
    #     # # plt.figure(figsize=(12, 8))
    #     # # 将数据转换为两个列表，一个用于类别，一个用于值
    #     # categories, values = zip(*data)
    #     #
    #     # # 创建Seaborn柱状图
    #     # sns.relplot(y=list(values), x=list(categories), color=(243 / 255, 162 / 255, 97 / 255), kind="line", ci=None).figure.set_size_inches(16,6)
    #     # # 添加标题和标签
    #     # plt.title('疾病统计')
    #     # plt.xlabel('疾病')
    #     # plt.ylabel('数量')
    #     # plt.xticks([])
    #     #
    #     # # 显示图形
    #     # plt.show()
    #     data = sorted(action_dict.items(), key=lambda x: x[1], reverse=True)
    #     # plt.figure(figsize=(12, 8))
    #     # 将数据转换为两个列表，一个用于类别，一个用于值
    #     categories, values = zip(*data)
    #
    #     # 创建Seaborn柱状图
    #     sns.relplot(y=list(values), x=list(categories), color=(21 / 255, 151 / 255, 165 / 255), kind="line", ci=None).figure.set_size_inches(12, 8)
    #     # 添加标题和标签
    #     plt.title('症状统计')
    #     plt.xlabel('症状')
    #     plt.ylabel('数量')
    #     plt.xticks([])
    #
    #     # 显示图形
    #     plt.show()

    # 传入一个询问完毕的状态矩阵
    def select(self, index, state, label):
        # ele是memery中的一个个元素
        ele = {
            'label': label,
            'symptoms': []
        }
        for i, val in enumerate(state.sum(dim=1)):
            if val == 1:
                ele['symptoms'].append(self.action_space[i])
        if len(self.memery) != self.window_size:
            self.memery.append(ele)
        else:
            self.memery[index] = ele

    def action_proportion(self, action, label):
        total_num = 0
        # 标签下没有该症状就返回0
        if action not in self.label_table[label]['symptoms'].keys():
            return 0.
        for symptom_num in self.label_table[label]['symptoms'].values():
            total_num += symptom_num
        return self.label_table[label]['symptoms'][action] / (total_num + 1)

    def selected_probability(self, action, label):
        # 实时从当前memery中计算得到疾病下动作被选择的概率
        select_num = 0
        total_num = 0
        for i, e in enumerate(self.memery):
            if e['label'] == label:
                total_num += len(e['symptoms'])
                select_num += int(action in e['symptoms'])
        return select_num / (total_num + 1)

    # 获得动作的偏好误差
    def select_error(self, action, label):
        action_proportion = self.action_proportion(action, label) * 100
        selected_probability = self.selected_probability(action, label) * 100
        sigma = action_proportion - selected_probability
        # 返回的如果是负的说明偏好偏高 反之偏低
        err = sigma / (action_proportion + 1e-8)
        return err

    # 返回标签包含的动作 返回的是症状名称 具体还需要转换为动作空间的下标
    def label2action(self, label):
        return self.label_table[label]['symptoms'].keys()

    # 计算两个标签下症状的kl散度
    def kl_divergence(self, label):
        # 数据集中的分布
        action_proportions = []
        # 实际采样的分布
        selected_probabilities = []
        for action in self.label2action(label):
            action_proportions.append(self.action_proportion(action, label) + 1e-10)
            selected_probabilities.append(self.selected_probability(action, label) + 1e-10)
        kl = stats.entropy(action_proportions, selected_probabilities)
        return kl

    def ready(self):
        return len(self.memery) == self.window_size

    def label_weight(self):
        weight = [0 for i in self.label_space]
        total_num = 0
        for label in self.label_table.keys():
            total_num += self.label_table[label]['num']
            weight[self.label_space.index(label)] = self.label_table[label]['num']
        weight = torch.exp(-torch.FloatTensor(weight) / total_num).cuda()
        return weight
