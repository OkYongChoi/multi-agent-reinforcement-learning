#%%
import torch
from hyst_agent import HystAgent
import matplotlib.pyplot as plt
import pickle
import csv

import time
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
#                                   high = REWARD_BOUND,
#                                   size = (ACTION_SIZE, ACTION_SIZE))
reward_matrix=[[11, -30, 0], 
               [-30, 7, 6], 
               [0, 0, 5]]

if __name__ == "__main__":
    # The state of the game is only one and state is not changed
    state = torch.Tensor([1]).to(device)
    next_state = torch.Tensor([1]).to(device)

    # 2-agent matrix game
    # If you want to set lambda same with gamma, set lambda 0.99
    agent1 = HystAgent(lamb=0.1, increase_lr=2, decrease_lr=0.2)
    agent2 = HystAgent(lamb=0.1, increase_lr=2, decrease_lr=0.2)

    reward_list = []
    
    start = time.time()
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
    print(f"time:{time.time() - start}")
    
    # with open('reward_list.pkl','wb') as f:
    #     pickle.dump(reward_list, f)
    with open('reward_list.csv','w') as f:
        writer = csv.writer(f)
        writer.writerow(reward_list)
    agent1.save_model('hyst_1')
    agent2.save_model('hyst_2')

    plt.plot(reward_list)
    plt.show()
    plt.close()
# %%
