#%%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        
        # torch.manual_seed(seed)
        # torch.cuda.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)

        self.linear1 = nn.Linear(1,3)
        self.linear2 = nn.Linear(3,3)
        self.linear3 = nn.Linear(3,3)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.log_softmax(self.linear3(x), dim = 0)

        return x

#%%  
# 
if __name__ == '__main__': 
    model = QNet()
    a = torch.Tensor([1])
    print(model(a)[0])

# %%
