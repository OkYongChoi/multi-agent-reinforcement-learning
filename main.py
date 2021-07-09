#%%
import numpy as np
import torch

from qnet import QNet

# %%
##
ACTION_SIZE = 5 # Dimension of action space of each agent.
REWARD_BOUND = 2 # -bound ~ bound

TESTS = 1e4

REPLAYMEMORY = 1e3
BATCH_SIZE = 32
GAMMA = 0.9
EPSILON = 0.1 

#Make random reward array
#This is a shared reward setting, not an individualized reward setting 
# reward_matrix = np.random.randint(low = -REWARD_BOUND,
#                                  high = REWARD_BOUND,
#                                  size=(ACTION_SIZE, ACTION_SIZE))
reward_matrix=[[11, -30, 0], 
               [-30, 7, 6], 
               [0, 0, 5]]

# %%
# State is fixed becuase it's a matrix game.
state = torch.Tensor([[1], [1], [1]])
model = QNet()

#for test in range(TESTS):

agent1 = QNet()
agent2 = QNet()

