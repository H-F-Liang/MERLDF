# -*- coding:utf-8 -*-

import copy
import os
import random
import sys
from collections import deque

from tqdm import tqdm

sys.path.append(os.getcwd().replace("src/dialogue_system/dialogue_manager", ""))
import torch
from Network.Memory import Memory


class DialogueManager(object):
    """
    Dialogue manager of this dialogue system.
    """

    def __init__(self, user, agent, statistician, batch_size, replay_pool_size, max_turn, test_max_turn, reward_limit, tau):
        self.agent = agent
        self.user = user
        self.statistician = statistician
        self.maxlen = replay_pool_size
        self.batch_size = batch_size
        self.test_max_turn = test_max_turn
        self.train_max_turn = max_turn
        self.max_turn = self.train_max_turn if user.mode == 'train' else self.test_max_turn
        self.reward_limit = reward_limit
        self.tau = tau
        self.state = None
        self.init_state = 0
        self.turn = 0
        self.wrong = 0
        self.agent_sample = None
        self.human_sample = None
        # 0，1，2(错误、正确、重复)
        self.action_status = 0
        self.greedy_strategy = ''
        self.experience_replay_pool = Memory(memory_size=self.maxlen)

    # 只更新真实状态
    def state_update(self, action_index):
        if (self.agent.action_space[action_index] in self.user.true_symptoms) or (
                self.agent.action_space[action_index] in self.user.explicit_symptoms):
            self.state[action_index, 0] = 1.0
        elif self.agent.action_space[action_index] in self.user.false_symptoms:
            self.state[action_index, 1] = 1.0
        elif self.agent.action_space[action_index] in self.user.implicit_symptoms:
            self.state[action_index, 2] = 1.0
        else:
            self.state[action_index, 1] = 1.0

    # 只更新临时状态
    def tmp_state_update(self, state, action_index):
        if (self.agent.action_space[action_index] in self.user.true_symptoms) or (
                self.agent.action_space[action_index] in self.user.explicit_symptoms):
            state[action_index, 0] = 1.0
        elif self.agent.action_space[action_index] in self.user.false_symptoms:
            state[action_index, 1] = 1.0
        elif self.agent.action_space[action_index] in self.user.implicit_symptoms:
            state[action_index, 2] = 1.0
        else:
            state[action_index, 1] = 1.0

        return state

    def next(self, save_record, greedy_strategy, mode):
        self.greedy_strategy = greedy_strategy
        tmp_state_1 = copy.deepcopy(self.state)
        tmp_state_2 = copy.deepcopy(self.state)
        action_index, res_state, reward, episode_over = self.agent_next(tmp_state_1, save_record, greedy_strategy)
        action_fixed = False
        reward_fixed = False
        if greedy_strategy != 'dqn' and greedy_strategy != 'random' and not episode_over:
            action_index, res_state, reward, episode_over, action_fixed, reward_fixed = self.human_next(tmp_state_2, reward, action_index, save_record)
        self.state = res_state
        if save_record:
            if action_fixed or reward_fixed:
                self.experience_replay_pool.add(self.human_sample)
            else:
                self.experience_replay_pool.add(self.agent_sample)
        return reward, episode_over

    def agent_next(self, tmp_state, save_record, greedy_strategy):
        state = copy.deepcopy(tmp_state)
        action_index = self.agent.next(state=state, greedy_strategy=greedy_strategy, user=self.user)
        tmp_state = self.tmp_state_update(tmp_state, action_index)
        next_state = copy.deepcopy(tmp_state)
        self.turn += 1
        # 询问到的症状集合
        guess_correct_symptom = {self.agent.action_space[i] for i, val in enumerate(tmp_state) if val.sum() != 0}
        # 结束
        if self.user.symptoms.issubset(guess_correct_symptom):
            episode_over = True
            reward = self.reward_limit
            self.action_status = 1
        # 重复
        elif self.state.sum(dim=1)[action_index] != 0:
            self.wrong += 1
            episode_over = False
            reward = -self.reward_limit // 4
            self.action_status = 2
        # 错误
        elif self.agent.action_space[action_index] not in self.user.symptoms:
            self.wrong += 1
            episode_over = False
            reward = -self.reward_limit // 20
            self.action_status = 0
        # 正确
        else:
            episode_over = False
            reward = self.reward_limit // 20
            self.action_status = 1

        if self.turn - self.init_state >= self.max_turn:
            episode_over = True

        if save_record:
            self.agent_sample = (state, action_index, reward, next_state, episode_over)

        return action_index, tmp_state, reward, episode_over

    # 传入状态和待修正动作
    def human_next(self, tmp_state, reward, action_index, save_record):
        state = copy.deepcopy(tmp_state)
        # 可选
        action_fixed = False
        action_index, action_fixed = self.human_action_fix(action_index, state)

        tmp_state = self.tmp_state_update(tmp_state, action_index)
        next_state = copy.deepcopy(tmp_state)
        # 询问到的症状集合
        guess_correct_symptom = {self.agent.action_space[i] for i, val in enumerate(tmp_state) if val.sum() != 0}
        # 患者所有症状询问完毕对话结束
        if self.user.symptoms.issubset(guess_correct_symptom):
            episode_over = True
            reward = self.reward_limit
            self.action_status = 1
        # 对话尚未结束 当前动作重复
        elif self.state.sum(dim=1)[action_index] != 0:
            episode_over = False
            reward = -self.reward_limit // 4
            self.action_status = 2
        # 对话尚未结束 当前动作错误
        elif self.agent.action_space[action_index] not in self.user.symptoms:
            episode_over = False
            reward = -self.reward_limit // 20
            self.action_status = 0
        # 对话尚未结束 当前动作正确
        else:
            episode_over = False
            reward = self.reward_limit // 20
            self.action_status = 1

        if self.turn - self.init_state >= self.max_turn:
            episode_over = True
        # 对上面的动作做出反馈后进行奖励修正 对话结束时不再修正
        reward_fixed = False
        if not episode_over:
            # 可选
            reward, reward_fixed = self.human_reward_fix(action_index, reward, action_fixed)

        if save_record and (action_fixed or reward_fixed):
            self.human_sample = (state, action_index, reward, next_state, episode_over)

        return action_index, tmp_state, reward, episode_over, action_fixed, reward_fixed

    def reset(self, mode, greedy_strategy):
        self.user.reset(mode, greedy_strategy)
        self.state = torch.zeros((len(self.agent.action_space), 3))

        for symptom in self.user.explicit_symptoms:
            self.state[self.agent.action_space.index(symptom), 0] = 1.0

        self.turn = self.state.sum()
        self.init_state = self.state.sum()
        self.wrong = 0
        self.max_turn = self.train_max_turn if self.user.mode == 'train' else self.test_max_turn
        self.human_sample = None
        self.agent_sample = None

    # 返回当前样本应该被询问但还没有被询问的症状
    def symptom_not_selected(self, state):
        index = [i for i, val in enumerate(state.sum(dim=1)) if
                 val.item() == 0 and self.agent.action_space[i] in self.user.symptoms]
        return index

    def human_action_fix(self, old_action_index, state):
        action_index = old_action_index
        fixed = False
        if self.statistician.ready():
            # 随机一个概率
            if random.random() < max(self.tau, 0.2)\
                    and action_index not in self.symptom_not_selected(state) \
                    and self.agent.action_space[action_index] in self.statistician.label2action(self.user.label) \
                    and self.statistician.select_error(self.agent.action_space[action_index], self.user.label) < -self.statistician.tau:

                not_been_selected = set(self.symptom_not_selected(state))
                label_actions = self.statistician.label2action(self.user.label)
                label_actions_index = set([self.agent.action_space.index(i) for i in label_actions])
                # 找到属于当前标签的且还没有被选择的动作
                index = list(label_actions_index & not_been_selected)
                # 找到这些动作中的低偏好动作 当然也可能没有
                # 优先选择倾向最差的
                ii = [(i, self.statistician.select_error(self.agent.action_space[i], self.user.label))
                      for i in index if self.statistician.select_error(self.agent.action_space[i], self.user.label) > self.statistician.tau]
                if len(ii) > 0:
                    mx = max(ii, key=lambda x: x[1])
                    # mx = random.sample(ii, k=1)[0]
                    action_index = mx[0]
                    fixed = True
        return action_index, fixed

    # 奖励修正应该分为在动作受到修改后修改奖励和动作没有修改但是需要修改奖励
    # 奖励修正发生的可以更频繁 但是动作修正要微调 不能太频繁会影响分布
    def human_reward_fix(self, old_action_index, old_reward, action_fixed):
        fixed = True
        if self.statistician.ready():
            '''
             如果当前的动作属于当前标签
             如果当前的动作在标签中属于高概率动作 那么说明该动作应该被鼓励询问故即使该动作不在当前样本内也不应该给予惩罚
                如果该动作选择正确 且 偏好过高则适当降低奖励
                如果该动作选择错误 且 偏好过高则适当提高的惩罚
                如果动作选择正确 且 偏好过低适当提高奖励
                如果动作选择错误 且 偏好过低适当降低惩罚
                偏好合适则不做处理
            '''
            if self.agent.action_space[old_action_index] in self.statistician.label2action(self.user.label):
                # 动作正确根据分布情况修正奖励
                # 影响agent对动作的偏好程度
                flag = self.statistician.select_error(self.agent.action_space[old_action_index], self.user.label)
                if flag < -self.statistician.tau and self.action_status == 1:
                    reward = self.reward_limit // 40
                elif flag < -self.statistician.tau and self.action_status == 0:
                    reward = -self.reward_limit // 8
                elif flag > self.statistician.tau and self.action_status == 1:
                    reward = self.reward_limit // 5
                elif flag > self.statistician.tau and self.action_status == 0:
                    reward = -self.reward_limit // 40
                else:
                    reward = old_reward
                    fixed = False
            else:
                reward = old_reward
                fixed = False
        else:
            reward = old_reward
            fixed = False

        return reward, fixed

    def decrease_fix(self):
        self.tau = max(0.01, self.tau - 0.0001)

    def train(self):
        return self.__train_dqn()

    def __train_dqn(self):

        if self.experience_replay_pool.size() < self.batch_size:
            # print("经验池未满 ...")
            return 0.
        cur_bellman_loss = 0.0
        n_epoch = self.experience_replay_pool.size() // self.batch_size + 1
        n_epoch = min(400, n_epoch)
        for i in tqdm(range(n_epoch), desc='train policy', position=1, leave=False, ncols=100, colour='red'):
            batch = self.experience_replay_pool.sample(self.batch_size, False)
            loss = self.agent.train(batch=batch)
            cur_bellman_loss += loss["loss"]
        # print("Train Loss %.8f" % (float(cur_bellman_loss) / n_epoch))
        self.agent.update_target_network()
        # self.experience_replay_pool.clear()
        return float(cur_bellman_loss) / n_epoch
