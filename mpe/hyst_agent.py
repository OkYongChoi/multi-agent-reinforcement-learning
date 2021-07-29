import numpy as np
import random

import torch
import torch.optim as optim
from qnet import QNet
from replay_memory import ReplayMemory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HystAgent():
    def __init__(self, state_size = 18, action_size=5, discount_factor=0.99, lamb=0.1, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.05, 
                 capacity=1000, batch_size=32, increase_lr=1, decrease_lr=0.5, load_model=False):

        # action, state
        self.state_size = state_size
        self.action_size = action_size

        # Hyper parameters for the Hysteretic Deep Q Learning
        self.gamma = discount_factor
        self.lamb = lamb
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.increase_lr = increase_lr
        self.decrease_lr = decrease_lr
        self.memory = ReplayMemory(capacity, batch_size)
        
        self.model = QNet().to(device)
        self.target = QNet().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)

        ## For Multiple GPUs.
        # If there is an imbalance between using GPUs, 
        # for instance, if you have GPU 1 which has less than 75% of the memory or cores of GPU 0.
        # you better exclude GPU 1 or allocate 
        # if torch.cuda.device_count() > 1:
        #     print("Let's use", torch.cuda.device_count(), "GPUs!")
        #     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        #     self.model = nn.DataParallel(self.model)
        #     self.target = nn.DataParallel(self.target)
        self.model.to(device)
        self.target.to(device)    

        self.update_target_model()        

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
            # r + gamma * maxQ(s',a') - Q(s,a)
            mb_loss_lamb = mb[2] + self.lamb*self.target(mb[3]).max() - self.model(mb[0])[mb[1]]
            mb_loss = mb[2] + self.gamma*self.target(mb[3]).max() - self.model(mb[0])[mb[1]]
            if mb_loss_lamb > 0 :
                mb_loss = mb_loss*mb_loss * self.increase_lr
            else:
                mb_loss = mb_loss*mb_loss * self.decrease_lr
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