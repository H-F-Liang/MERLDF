import copy
import math

import torchvision
from collections import namedtuple
from Loss.FocalLoss import FocalLoss
import torch.nn as nn
import torch.optim
import torch
import torch.nn.functional as F

'''
dataset dxy
seed 666
mask .1 .0
lr 1e-3
epoch 2000
batch_size 16
hide_size 1000
rollback 10
step 5 0.999 / 10 0.999
loss lambda 1.0
res 0.885 0.865
'''

'''
dataset muzhi
seed 666
mask .1 .0
lr 1e-3
epoch 2000
batch_size 16
hide_size 5000
rollback 10
step 20 0.999
loss lambda 1.0
res 0.761 0.754
'''

'''
dataset mz10
seed 666
mask .0 .0
lr 1e-3
epoch 4000
batch_size 1024
hide_size 1000
rollback 15
step 10 0.999
loss lambda 0.2
res 0.661 0.654
'''


# 0.761
class Net(nn.Module):
    def __init__(self, input_size, hide_size, output_size):
        super(Net, self).__init__()
        self.share_layer = nn.Sequential(
            nn.Linear(input_size[0] * 3, out_features=hide_size),
            nn.Dropout(0.2),
            nn.Tanh(),
        )
        self.class_layer = nn.Sequential(
            nn.Linear(hide_size, out_features=output_size * 10),
            nn.Dropout(0.2),
            nn.Tanh()
        )

        self.auxiliary_layer = nn.Sequential(
            nn.Linear(hide_size, out_features=output_size * 10),
            nn.Dropout(0.2),
            nn.Tanh()
        )
        self.class_output_layer = nn.Sequential(
            nn.Linear(output_size * 10, out_features=output_size),
        )
        self.auxiliary_output_layer = nn.Sequential(
            nn.Linear(output_size * 10, out_features=output_size),
        )

    def forward(self, x: torch.tensor):
        if x.dim() == 2:
            x = x.unsqueeze(dim=0)
        x = x.view(x.size(0), -1)

        output = self.share_layer(x)
        output_1 = self.class_layer(output)
        output_1 = self.class_output_layer(output_1)

        output_2 = self.auxiliary_layer(output)
        output_2 = self.auxiliary_output_layer(output_2)
        return F.tanh(output_1 - output_2), output_2


class DiseaseClassifier:
    def __init__(self, device, input_size, hide_size, output_size, lr=1e-3, weight=None, _lambda=1.0):
        self.Transition = namedtuple('Transition', ('state', 'label'))
        # 预测网络
        self.predict_net = Net(input_size, hide_size, output_size).to(device)
        self.best_net = Net(input_size, hide_size, output_size).to(device)
        self.best_net.load_state_dict(self.predict_net.state_dict())
        self.device = device
        self.lr = lr
        self._lambda = _lambda
        self.optimizer = torch.optim.Adam(self.predict_net.parameters(), lr=self.lr, betas=(0.9, 0.99), eps=1e-08)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.999)
        self.criterion = FocalLoss(weight=weight)
        self.auxiliary_criterion = nn.CrossEntropyLoss()

    def train(self, batch):
        batch = self.Transition(*zip(*batch))
        x = torch.stack(batch.state).to(torch.float32).to(self.device)
        # 此时y是[0,1,0,0]类似的张量
        y = torch.stack(batch.label).to(torch.float32).to(self.device)
        # output也是张量
        output_1, output_2 = self.predict_net(x)
        # 输出的是标签标号
        y_pre = torch.argmax(output_1, dim=1).to(torch.float32)
        y_true = torch.argmax(y, dim=1).to(torch.float32)
        acc = (y_true == y_pre).sum() / y_true.size(0)
        # 两个张量拿来做损失
        loss = self.criterion(output_1, y) + self.auxiliary_criterion(output_2, torch.where(y == 1., 0., 1.)) * self._lambda
        loss.requires_grad_(True)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        res = {'loss': loss.item(), 'acc': acc.item()}
        return res

    def save_model(self, save_path='./model/clf_model.pth'):
        self.best_net.load_state_dict(self.predict_net.state_dict())
        torch.save(self.best_net.state_dict(), save_path)

    def rollback_model(self):
        self.predict_net.load_state_dict(self.best_net.state_dict())

    def predict(self, x):
        x = x.to(self.device).to(torch.float32)
        with torch.no_grad():
            y_pre, _ = self.predict_net(x)
            y_pre = torch.argmax(y_pre, dim=1)
        return y_pre

    def predict_proba(self, x):
        x = x.to(self.device).to(torch.float32)
        with torch.no_grad():
            y_pre, _ = self.predict_net(x)
        return y_pre

    def predict_proba_best(self, x):
        x = x.to(self.device).to(torch.float32)
        with torch.no_grad():
            y_pre, _ = self.best_net(x)
        return y_pre

    def train_mode(self):
        self.predict_net.train()
        self.best_net.train()

    def eval_mode(self):
        self.predict_net.eval()
        self.best_net.eval()

    def adjust_lr(self, low=True):
        if low:
            self.optimizer.param_groups[0]['lr'] = 1e-3
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.999)
        else:
            self.optimizer.param_groups[0]['lr'] = 1e-3
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.999)
