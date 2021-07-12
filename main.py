#%%
import numpy as np
import torch

from qnet import QNet
from hyst_agent import HystAgent

import matplotlib.pyplot as plt
# %%
##
ACTION_SIZE = 5 # Dimension of action space of each agent.
REWARD_BOUND = 2 # -bound ~ bound

TESTS = 1e4

REPLAYMEMORY = 1e3
BATCH_SIZE = 32
GAMMA = 0.9
EPSILON = 0.1 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device: ', device)

#Make random reward array
#This is a shared reward setting, not an individualized reward setting 
# reward_matrix = np.random.randint(low = -REWARD_BOUND,
#                                  high = REWARD_BOUND,
#                                  size=(ACTION_SIZE, ACTION_SIZE))
reward_matrix=[[11, -30, 0], 
               [-30, 7, 6], 
               [0, 0, 5]]

if __name__ == "__main__":
    # The state of the game is only one and state is not changed
    state = torch.Tensor([1]).to(device)
    next_state = torch.Tensor([1]).to(device)
    agent1 = HystAgent()
    agent2 = HystAgent()

    reward_list = []

    for i in range(3000):
        act_agent1 = agent1.get_action(state)
        act_agent2 = agent2.get_action(state)
    
        #Get the corresponding reward    
        reward = reward_matrix[act_agent1][act_agent2]
        reward_list.append(reward)

        agent1.push_sample(state,act_agent1,reward,next_state,True)
        agent2.push_sample(state,act_agent2,reward,next_state,True)

        #train the model of each agent.
        agent1.train_model()
        agent2.train_model()

    plt.plot(reward_list)
    plt.show()
    plt.close()
# %%
