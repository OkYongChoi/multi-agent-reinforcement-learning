#%%
from replay_memory import ReplayMemory
from qnet import QNet

import numpy as np
import random
import torch
from torch import optim

#from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HystAgent():
    def __init__(self, state_size =1, action_size=3, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.01, 
                 capacity=1000, batch_size=32, increase_lr=0.1, decrease_lr=0.01, load_model=False):

        # action, state
        self.state_size = state_size
        self.action_size = action_size

        # Hyper parameters for the Hysteretic Deep Q Learning
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.increase_lr = increase_lr
        self.decrease_lr = decrease_lr

        self.memory = ReplayMemory(capacity, batch_size)
        self.model = QNet().to(device)
        self.target = QNet().to(device)
        self.update_target_model()

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)

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
        self.memory.push(state, action, reward, next_state, done)

    def train_model(self):
        if len(self.memory) < self.batch_size:
            return
        mini_batch = self.memory.sample()
        sum = 0
        for mb in mini_batch:
            # r+gamma*maxQ(s',a') - Q(s,a)
            mb_loss= mb[2] + self.gamma*self.target(mb[3]).max() - self.model(mb[0])[mb[1]]
            #print(mb_loss)
            if mb_loss > 0 :
                mb_loss *= self.increase_lr
            else:
                mb_loss *= self.decrease_lr
            sum += mb_loss

        self.optimizer.zero_grad()
        sum.backward()
        self.optimizer.step()


    def update_target_model(self):
        "Update target model to the current model"
        self.target.load_state_dict(self.model.state_dict())

    def save_model(self, path):
        torch.save(self.model.state_dict, f'{path}/hyst_model.pth')
    
    def load_model(self, path):
        self.model.load_state_dict(torch.load(f'{path}/hyst_model.pth'))


# %%
if __name__ == "__main__":
    agent = HystAgent()
    agent.push_sample(1,1,1,1,1)
