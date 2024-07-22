import copy
import random
import sys
import os
from collections import deque

from Statistician import Statistician
from disease_classifier import DiseaseClassifier
from pack import *
import torch
import tqdm
from tqdm import tqdm

sys.path.append(os.getcwd().replace("dialogue_system/run", ""))
sys.path.append("../..")
from Agent import AgentDQN
from User import User
from dialogue_manager import DialogueManager


def create_new_directory(base_dir, prefix="exp_"):
    existing_directories = [d for d in os.listdir(base_dir) if d.startswith(prefix)]
    if not existing_directories:
        new_directory_name = prefix + "1"
    else:
        max_number = max([int(d.split("_")[1]) for d in existing_directories])
        new_directory_name = prefix + str(max_number + 1)
    new_directory_path = os.path.join(base_dir, new_directory_name)
    os.makedirs(new_directory_path)
    os.makedirs(os.path.join(new_directory_path, 'model'))
    print(f"本次实验结果保存在：" + new_directory_path)
    return new_directory_path


class RunningSteward(object):
    def __init__(self, parameter):
        self.fdir = create_new_directory("res")
        self.train_performance = open(self.fdir + r"/train_performance.txt", "w", encoding="utf-8")
        self.test_performance = open(self.fdir + r"/test_performance.txt", "w", encoding="utf-8")

        self.parameter = parameter
        self.simulation_epoch = self.parameter['simulation_epoch']
        self.train_epoch = self.parameter['train_epoch']
        self.clf_learning_rate = self.parameter['clf_learning_rate']
        self.clf_train_epoch = self.parameter['clf_train_epoch']
        self.clf_batch_size = self.parameter['clf_batch_size']
        self._lambda = self.parameter['lambda']
        self.hide_size = self.parameter['hide_size']
        self.load_model = self.parameter['load_model']
        self.model_load_path = self.parameter['model_load_path']
        train_set, test_set, self.action_space, self.labels = ProcessTrainData(self.parameter['train_dataset_path'],
                                                                               self.parameter['test_dataset_path'])
        self.train_size = len(train_set)
        self.test_size = len(test_set)
        statistician = Statistician(train_set, action_space=self.action_space, label_space=self.labels)

        self.disease_classifier_cnn = DiseaseClassifier(device=self.parameter['device'], input_size=(len(self.action_space), 3), hide_size=self.hide_size,
                                                        output_size=len(self.labels), weight=statistician.label_weight(), lr=self.clf_learning_rate, _lambda=self._lambda)
        self.disease_classifier_pool = deque(maxlen=self.train_size)
        self.disease_classifier_train_pool = deque(maxlen=self.train_size)
        self.disease_classifier_test_pool = deque(maxlen=self.test_size)
        self.disease_classifier_test_pool_tmp = deque(maxlen=self.test_size)

        agent = AgentDQN(action_space=self.action_space, label_space=self.labels, device=torch.device(self.parameter['device']),
                         learning_rate=parameter['learning_rate'], gamma=self.parameter['gamma'], reduce=self.parameter['reduce'],
                         reward_limit=self.parameter['reward_limit'])
        # 加载现有模型
        if self.load_model:
            agent.load_model(self.model_load_path)

        self.user = User(goal_set=(train_set, test_set))
        self.dialogue_manager = DialogueManager(user=self.user, agent=agent, statistician=statistician, batch_size=self.parameter['batch_size'],
                                                replay_pool_size=self.parameter['replay_pool_size'],
                                                max_turn=self.parameter['max_turn'],
                                                test_max_turn=self.parameter['test_max_turn'],
                                                reward_limit=self.parameter['reward_limit'],
                                                tau=self.parameter['tau'])

        self.acc_hist = 0
        self.recall_hist = 0
        self.disease_counter = 0
        self.symptom_counter = 0

    def train(self):
        if not self.load_model:
            with tqdm(total=self.train_epoch, desc='Train Agent', leave=True, position=0, ncols=150, colour='green') as progress_bar:
                for index in range(self.train_epoch):
                    self.simulation(mode='train', save_record=True, greedy_strategy='epsilon-dqn')
                    # 抽样少量训练集和测试集样本测验
                    # 评估模式 关闭Dropout/BN
                    self.dialogue_manager.agent.dqn.eval()
                    res = self.simulation(mode='train', save_record=False, greedy_strategy='dqn')
                    res = self.simulation(mode='test', save_record=False, greedy_strategy='dqn', sample=True)
                    res = self.simulation(mode='test', save_record=False, greedy_strategy='dqn')
                    # 回复训练模式
                    self.dialogue_manager.agent.dqn.train()
                    # 训练
                    loss = self.dialogue_manager.train()
                    progress_bar.set_postfix(epoch=index+1, loss=loss, turn=res['avg_turn'], recall=res['recall'], match_rate=res['mr2'], recall_best=self.recall_hist)
                    progress_bar.update(1)
            progress_bar.close()
        else:
            self.dialogue_manager.agent.dqn.eval()
            self.simulation(mode='train', save_record=False, greedy_strategy='dqn')
            self.simulation(mode='test', save_record=False, greedy_strategy='dqn')

        with tqdm(total=self.clf_train_epoch, desc='Train Classifier', leave=True, position=0, ncols=150, colour='green') as progress_bar:
            self.init_clf()
            for index in range(self.clf_train_epoch):
                acc = self.train_clf()
                progress_bar.set_postfix(epoch=index+1, acc=acc, acc_best=self.acc_hist)
                progress_bar.update(1)
        progress_bar.close()
        print({'recall_best': self.recall_hist, 'acc_best': self.acc_hist})

    def warm_up(self):
        if not self.load_model:
            with tqdm(total=5, desc='Warm up', leave=True, position=0, ncols=100, colour='green') as progress_bar:
                for i in range(5):
                    self.simulation(mode='train', save_record=True, greedy_strategy="epsilon-dqn")
                    progress_bar.update(1)
            progress_bar.close()

    def simulation(self, mode='train', save_record=True, greedy_strategy='random', sample=False):
        total_reward = 0
        total_turns = 0
        total_symptoms_accuracy = 0
        total_ask_accuracy = 0
        total_std_turn = 0
        total_expect_reward = 0
        loop = self.simulation_epoch
        if mode == 'test' and greedy_strategy == 'dqn' and not sample:
            loop = self.test_size
        if self.load_model and mode == 'test' and greedy_strategy == 'dqn':
            loop = self.test_size
        if self.load_model and mode == 'train' and greedy_strategy == 'dqn':
            loop = self.train_size

        with tqdm(total=loop, desc=mode + ' sample', leave=False, position=1, ncols=100, colour='red') as progress_bar:
            for epoch_index in range(loop):
                self.dialogue_manager.reset(mode=mode, greedy_strategy=greedy_strategy)
                episode_over = False
                dialog_reward = 0
                while not episode_over:
                    reward, episode_over = self.dialogue_manager.next(save_record=save_record,
                                                                      greedy_strategy=greedy_strategy, mode=mode)
                    dialog_reward += reward

                if mode == 'train' and greedy_strategy == 'dqn':
                    self.store_train_state()
                    self.dialogue_manager.statistician.select(self.user.train_index, self.dialogue_manager.state, self.user.label)
                if mode == 'test' and greedy_strategy == 'dqn':
                    self.store_test_state()

                if greedy_strategy == 'dqn':
                    total_reward += dialog_reward
                    total_turns += self.dialogue_manager.turn - self.dialogue_manager.init_state
                    total_std_turn += len(self.dialogue_manager.user.symptoms) - self.dialogue_manager.init_state
                    total_ask_accuracy += (self.dialogue_manager.turn - self.dialogue_manager.wrong - self.dialogue_manager.init_state) / (
                            self.dialogue_manager.turn - self.dialogue_manager.init_state)
                    total_symptoms_accuracy += (self.dialogue_manager.turn - self.dialogue_manager.wrong - self.dialogue_manager.init_state) / (
                            len(self.dialogue_manager.user.symptoms) - self.dialogue_manager.init_state + int(len(self.dialogue_manager.user.symptoms) == self.dialogue_manager.init_state))
                    total_expect_reward += self.parameter['reward_limit'] + (len(self.dialogue_manager.user.symptoms) - self.dialogue_manager.init_state) * self.parameter['reward_limit'] // 8

                progress_bar.update(1)
            progress_bar.close()
        res = None
        if greedy_strategy == 'dqn':
            average_reward = float("%.3f" % (float(total_reward) / loop))
            average_turn = float("%.3f" % (float(total_turns) / loop))
            average_std_turn = float("%.3f" % (float(total_std_turn) / loop))
            average_symptoms_accuracy = float("%.3f" % (float(total_symptoms_accuracy) / loop))
            average_accuracy = float("%.3f" % (float(total_ask_accuracy) / loop))
            average_expect_reward = float("%.3f" % (float(total_expect_reward) / loop))

            if mode == 'test' and not sample:
                if self.recall_hist < average_symptoms_accuracy:
                    self.recall_hist = average_symptoms_accuracy
                    self.dialogue_manager.agent.save_model(self.fdir + '/model/model.pth')
                    self.dialogue_manager.agent.rollback_model()
                    self.disease_classifier_test_pool = copy.deepcopy(self.disease_classifier_test_pool_tmp)
                    self.symptom_counter = 0
                    self.acc_hist = 0

            res = {'mode': mode,
                   "strategy": greedy_strategy,
                   "avg_turn": average_turn,
                   "recall": average_symptoms_accuracy,
                   "mr2": average_accuracy
                   }
            if mode == 'train':
                self.train_performance.write("%.3f %.3f %.3f %.3f\n" % (average_turn, average_reward, average_symptoms_accuracy, average_accuracy))
                self.train_performance.flush()
            elif mode == 'test' and sample:
                self.test_performance.write("%.3f %.3f %.3f %.3f\n" % (average_turn, average_reward, average_symptoms_accuracy, average_accuracy))
                self.test_performance.flush()
        return res

    def init_clf(self):
        # 采集分类器初始样本
        for i in range(self.train_size):
            self.user.reset('clf_train', 'dqn')
            state = torch.zeros(len(self.action_space), 3)
            # 转为向量
            y = torch.zeros(len(self.labels))
            y[self.labels.index(self.user.label)] = 1

            for symptom in self.user.symptoms:
                if symptom in self.user.explicit_symptoms or symptom in self.user.true_symptoms:
                    state[self.action_space.index(symptom), 0] = 1.0
                elif symptom in self.user.false_symptoms:
                    state[self.action_space.index(symptom), 1] = 1.0
                elif symptom in self.user.implicit_symptoms:
                    state[self.action_space.index(symptom), 2] = 1.0
            self.disease_classifier_pool.append((state, y))

    # 每次都存储当前询问后的患者状态，用于后续的微调
    def store_train_state(self):
        y = torch.zeros(len(self.labels))
        y[self.labels.index(self.user.label)] = 1
        state = torch.zeros(len(self.action_space), 3)
        for i, val in enumerate(self.dialogue_manager.state):
            if val.sum() == 1:
                if val[0] == 1:
                    state[i, 0] = 1
                if val[1] == 1 and random.random() < 0.1:
                    state[i, 1] = 1
                if val[2] == 1:
                    state[i, 2] = 1

        self.disease_classifier_train_pool.append((state, y))

    def store_test_state(self):
        y = torch.zeros(len(self.labels))
        y[self.labels.index(self.user.label)] = 1
        state = torch.zeros(len(self.action_space), 3)
        for i, val in enumerate(self.dialogue_manager.state):
            if val.sum() == 1:
                if val[0] == 1:
                    state[i, 0] = 1
                if val[1] == 1 and random.random() < 0.0:
                    state[i, 1] = 1
                if val[2] == 1:
                    state[i, 2] = 1
        self.disease_classifier_test_pool_tmp.append((state, y))

    # 后续微调上界模型
    def train_clf(self):
        n_epoch = len(self.disease_classifier_train_pool) // self.clf_batch_size + 1
        classifier_loss = 0
        classifier_acc = 0
        disease_y = []
        disease_pre = []

        self.disease_classifier_cnn.train_mode()
        for i in range(n_epoch):
            batch_1 = random.sample(self.disease_classifier_train_pool, min(self.clf_batch_size, len(self.disease_classifier_train_pool)))
            batch_2 = random.sample(self.disease_classifier_pool, min(self.clf_batch_size, len(self.disease_classifier_pool)))
            batch = batch_1 + batch_2
            random.shuffle(batch)
            res = self.disease_classifier_cnn.train(batch=batch)
            classifier_loss += res["loss"]
            classifier_acc += res['acc']

        self.disease_classifier_cnn.eval_mode()
        for e in self.disease_classifier_test_pool:
            state, y = e
            y_pre = self.disease_classifier_cnn.predict(x=state)
            y = torch.argmax(y)
            disease_y.append(y)
            disease_pre.append(y_pre)
        acc = float("%.3f" % (((torch.tensor(disease_pre) == torch.tensor(disease_y)).sum() / len(disease_pre)).item()))
        if acc > self.acc_hist:
            self.disease_classifier_cnn.save_model(self.fdir + '/model/clf_model.pth')
            self.disease_classifier_cnn.rollback_model()
            self.acc_hist = acc
            self.disease_counter = 0
        else:
            self.disease_counter += 1
            if self.disease_counter > 10:
                self.disease_classifier_cnn.rollback_model()
                self.disease_counter = 0
        return acc
