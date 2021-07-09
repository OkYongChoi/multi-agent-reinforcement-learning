#%%
from replay_memory import ReplayMemory
from qnet import QNet

import numpy as np
import random
import torch


class HystAgent():
    def __init__(self, state_size =1, action_size=3, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.01, 
                 capacity=1000, batch_size=32, increase_lr=0.9, decrease_lr=0.1, load_model=False):

        # action, state
        self.state_size = state_size
        self.action_size = action_size

        # Hyper parameters for the Hysteretic Deep Q Learning
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.increase_lr = increase_lr
        self.decrease_lr = decrease_lr

        self.memory = ReplayMemory(capacity, batch_size)
        self.model = QNet()
        self.target = QNet()

        #If load_model is True, load the saved model
        if load_model is True:
            pass

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            action = random.randrange(self.action_size)
        else:
            action = torch.argmax(self.model(state))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay        
        return action

    def push_sample(self, state, action, reward, next_state, done=True):
        self.memory.push((state, action, reward, next_state, done))

    def train_model(self):
        if len(self.memory) < self.batch_size*3:
            return
        mini_batch = self.memory.sample()

# %%
if __name__ == "__main__":
    pass
