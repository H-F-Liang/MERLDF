class User(object):

    def __init__(self, goal_set):

        self.implicit_symptoms = None
        self.explicit_symptoms = None
        self.false_symptoms = None
        self.true_symptoms = None
        self.symptoms = set()
        self.goal_set = goal_set  # (self.data_filter(goal_set[0]), self.data_filter(goal_set[1]))
        self.goal = None
        self.cnt = 0
        self.test_index = 0
        self.train_index = 0
        self.dqn_index = 0
        self.clf_train_index = 0
        self.clf_test_index = 0
        self.mode = 'train'
        self.label = None

    def reset(self, mode, greedy_strategy):
        # 从数据集中随机选择样本
        if mode == 'train':
            self.mode = 'train'
            if greedy_strategy == 'dqn':
                self.goal = self.goal_set[0][self.dqn_index]
                self.dqn_index = (self.dqn_index + 1) % len(self.goal_set[0])
            else:
                self.goal = self.goal_set[0][self.train_index]
                self.train_index = (self.train_index + 1) % len(self.goal_set[0])
        elif mode == 'clf_train':
            self.mode = 'clf_train'
            self.goal = self.goal_set[0][self.clf_train_index]
            self.clf_train_index = (self.clf_train_index + 1) % len(self.goal_set[0])
        elif mode == 'clf_test':
            self.mode = 'clf_test'
            self.goal = self.goal_set[1][self.clf_test_index]
            self.clf_test_index = (self.clf_test_index + 1) % len(self.goal_set[1])
        else:
            self.mode = 'test'
            self.goal = self.goal_set[1][self.test_index]
            self.test_index = (self.test_index + 1) % len(self.goal_set[1])

        self.label = self.goal['label']
        self.symptoms = set(self.goal['explicit_symptom']) | set(self.goal['none_symptom']) | set(
            self.goal['implicit_symptom']) | set(self.goal['true_symptom'])
        self.true_symptoms = set(self.goal['true_symptom'])
        self.false_symptoms = set(self.goal['none_symptom'])
        self.explicit_symptoms = set(self.goal['explicit_symptom'])
        self.implicit_symptoms = set(self.goal['implicit_symptom'])

    def data_filter(self, dataset):
        resset = []
        for data in dataset:
            symptoms = set(data['explicit_symptom']) | set(data['none_symptom']) | set(
                data['implicit_symptom']) | set(data['true_symptom'])
            if len(symptoms) > 0:
                resset.append(data)
        return resset

