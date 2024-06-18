import torch.nn as nn
import torch
from collections import namedtuple

from Network.Dueling_DQN import Dueling_DQN


class D3QN(object):
    def __init__(self, input_size, output_size, action_space, label_space, device, learning_rate, gamma, reduce, reward_limit):
        self.input_size = input_size
        self.output_size = output_size
        self.action_space = action_space
        self.label_space = label_space
        self.lr = learning_rate
        self.gamma = gamma
        self.reduce = reduce
        self.device = device
        self.reward_limit = reward_limit
        self.update_step = 100
        self.Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'batch_done'))
        self.__build_model()

    def __build_model(self):
        self.target_network = Dueling_DQN(self.input_size, self.output_size).to(self.device)
        self.best_network = Dueling_DQN(self.input_size, self.output_size).to(self.device)
        self.network = Dueling_DQN(self.input_size, self.output_size).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        self.best_network.load_state_dict(self.network.state_dict())
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=400, gamma=self.reduce)
        self.criterion = nn.MSELoss()

    # 单批次训练
    def singleBatch(self, batch):
        batch = self.Transition(*zip(*batch))
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = batch.state, batch.action, batch.reward, batch.next_state, batch.batch_done
        # 返回的是一个个tuple需要转换为tensor
        batch_state = torch.stack(batch_state).to(self.device).unsqueeze(1)
        batch_next_state = torch.stack(batch_next_state).to(self.device).unsqueeze(1)
        batch_action = torch.tensor(batch_action).view(-1, 1).to(self.device)
        batch_reward = torch.tensor(batch_reward).view(-1, 1).to(self.device)
        batch_done = torch.tensor(batch_done).view(-1, 1).to(self.device)
        # 归一化奖励
        min_value = -self.reward_limit
        max_value = self.reward_limit
        batch_reward = 2 * (batch_reward - min_value) / (max_value - min_value) - 1

        # 结束状态需mask掉
        with torch.no_grad():
            target_Q_next = self.target_network(batch_next_state).detach()
            Q_next = self.network(batch_next_state).detach()
            Q_max_action = torch.argmax(Q_next, dim=1, keepdim=True)
            y = 1 / self.gamma * batch_reward + self.gamma * target_Q_next.gather(1, Q_max_action) * (1 - batch_done.int())

        loss = self.criterion(self.network(batch_state).gather(1, batch_action), y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return {"loss": loss.item()}

    def predict(self, Xs, **kwargs):
        Xs = Xs.to(self.device)
        return self.target_network.predict(Xs)

    def load_model(self, model_path):
        self.target_network.load_state_dict(torch.load(model_path)['model'])
        self.target_network.eval()
        self.network.load_state_dict(self.target_network.state_dict())
        self.network.eval()
        self.best_network.load_state_dict(self.target_network.state_dict())
        self.best_network.eval()

    # 保存时直接保存最佳target网络
    def save_model(self, model_save_path='./model.pth'):
        self.best_network.load_state_dict(self.network.state_dict())
        check_point = {
            'action_space': self.action_space,
            'label_space': self.label_space,
            'model': self.best_network.state_dict()
        }
        torch.save(check_point, model_save_path)

    def rollback_model(self):
        self.target_network.load_state_dict(self.best_network.state_dict())
        self.network.load_state_dict(self.best_network.state_dict())

    # 每一轮训练完后更新target网络
    def update_target_network(self):
        self.target_network.load_state_dict(self.network.state_dict())

    def train(self):
        self.network.train(mode=True)
        self.target_network.train(mode=True)
        self.best_network.train(mode=True)

    def eval(self):
        self.network.eval()
        self.target_network.eval()
        self.best_network.eval()

    def training(self):
        return self.network.training and self.target_network.training and self.best_network.training
