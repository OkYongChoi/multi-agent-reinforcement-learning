import random
from collections import deque

class ReplayMemory:
    def __init__(self, capacity=1000, batch_size=32):
        self.capacity = capacity
        self.batch_size = batch_size
        self.buffer = deque(maxlen=self.capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self):
        batch_size = min(self.batch_size, len(self.buffer))
        mini_batch = random.sample(self.buffer, batch_size)
        return mini_batch

    def __len__(self):
        return len(self.buffer)