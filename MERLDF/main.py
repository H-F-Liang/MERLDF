# -*- coding:utf-8 -*-
import argparse
import torch
import numpy as np
import random
from running_steward import RunningSteward

parser = argparse.ArgumentParser()
parser.add_argument("--device", dest="device", type=str, default="cuda:0", help="cpu / gpu")

# RL训练参数
parser.add_argument("--train_epoch", dest="train_epoch", type=int, default=300, help="训练轮次")  # 300 800 1000
parser.add_argument("--simulation_epoch", dest="simulation_epoch", type=int, default=50, help="每一轮模拟对话次数")
parser.add_argument("--batch_size", dest="batch_size", type=int, default=400, help="批次大小")  # 400 1024 1024
parser.add_argument("--learning_rate", dest="learning_rate", type=float, default=5e-5, help="学习率")  # 5e-5 5e-5 5e-5
parser.add_argument("--gamma", dest="gamma", type=float, default=0.8, help="强化学习贪婪概率")
parser.add_argument("--reduce", dest="reduce", type=float, default=0.92, help="学习率衰减")  # 0.88 0.985 0.998
parser.add_argument("--replay_pool_size", dest="replay_pool_size", type=int, default=50000, help="经验回放池大小")  # 150000 500000 500000
parser.add_argument("--max_turn", dest="max_turn", type=int, default=40, help="最大对话轮次")
parser.add_argument("--test_max_turn", dest="test_max_turn", type=int, default=10, help="最大对话轮次")
parser.add_argument("--reward_limit", dest="reward_limit", type=int, default=200, help="[-reward_limit, reward_limit]")
parser.add_argument("--tau", dest="tau", type=float, default=0.1, help="reshaping阈值")  # 0.1 0.1 0.1

# classifier训练参数
parser.add_argument("--clf_train_epoch", dest="clf_train_epoch", type=int, default=4000, help="疾病分类器训练轮次")
parser.add_argument("--clf_learning_rate", dest="clf_learning_rate", type=float, default=1e-3, help="疾病分类器学习率")
parser.add_argument("--clf_batch_size", dest="clf_batch_size", type=int, default=16, help="疾病分类器批次大小")  # 16 512 512
parser.add_argument("--hide_size", dest="hide_size", type=int, default=1000, help="疾病分类器隐藏层大小")  # 1000 5000 5000
parser.add_argument("--lambda", dest="lambda", type=float, default=1.0, help="疾病分类器辅助损失权重")  # 0.3 0.2 0.2

parser.add_argument("--load_model", dest="load_model", type=bool, default=False, help="加载已有模型(T/F)")
parser.add_argument("--model_load_path", dest="model_load_path", type=str, default="./res/exp_1/model/model.pth", help="agent模型加载路径[仅load_model为True有效]")
parser.add_argument("--train_dataset_path", dest="train_dataset_path", type=str, default=r"../data/dxy/train_set.csv", help="训练数据集所在路径")
parser.add_argument("--test_dataset_path", dest="test_dataset_path", type=str, default=r"../data/dxy/test_set.csv", help="测试数据集所在路径")

args = parser.parse_args()
parameter = vars(args)


def setup_all(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def run():
    setup_all(43)  # 888 233 3407
    steward = RunningSteward(parameter=parameter)
    steward.warm_up()
    steward.train()


if __name__ == "__main__":
    run()
