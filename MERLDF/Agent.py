import random
from D3QN import D3QN


class AgentDQN(object):
    def __init__(self, action_space, label_space, device, learning_rate, gamma, reduce, reward_limit):
        self.action_space = action_space  # type: list
        self.label_space = label_space  # type: list
        self.dqn = D3QN(input_size=(len(self.action_space), 3), output_size=len(self.action_space), action_space=self.action_space, label_space=self.label_space,
                        device=device, learning_rate=learning_rate, gamma=gamma, reduce=reduce,
                        reward_limit=reward_limit)  # 序号0表示问诊结束

    def real_next(self, state, user):
        index = [i for i, val in enumerate(state.sum(dim=1)) if
                 val.item() == 0 and self.action_space[i] in user.symptoms]
        if len(index) > 0:
            action_index = random.choice(index)
        else:
            action_index = random.choice([i for i, val in enumerate(state.sum(dim=1)) if val == 0])
        return action_index

    def next(self, state, greedy_strategy, user):
        action_index = 0
        if greedy_strategy == 'random':
            action_index = random.sample(range(0, len(self.action_space)), 1)[0]
        elif greedy_strategy == 'epsilon-dqn':
            greedy = random.random()
            if greedy < 0.2:
                action_index = random.sample(range(0, len(self.action_space)), 1)[0]
            else:
                action_index = self.dqn.predict(Xs=state)
        else:
            action_index = self.dqn.predict(Xs=state)

        return action_index

    def train(self, batch):
        loss = self.dqn.singleBatch(batch=batch)
        return loss

    def update_target_network(self):
        self.dqn.update_target_network()

    def rollback_model(self):
        self.dqn.rollback_model()

    def save_model(self, model_save_path='./model/model.pth'):
        self.dqn.save_model(model_save_path=model_save_path)

    def load_model(self, model_path):
        self.dqn.load_model(model_path)
