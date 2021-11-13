import numpy as np
from mdp import MDP
import matplotlib.pyplot as plt

class MAMDP(MDP):
    def __init__(self, state_size = 100, action_size = 10, feature_vector_size = 10, agent_size = 5):
        super().__init__(state_size, action_size, feature_vector_size)
        self.agent_size = agent_size
        
        self.make_random_adjacenecy_graph() # Set A(adjacency matrix) D(degree matrix)
        self.laplacian = self.deg_mat - self.adj_mat # Graph Laplacian
        
        ## transition matrix, graph laplacian, D, phi(i.e., state) are the same with the single-agent case
        ## Only need to concatenate the reward and parameters
        self.reward_matrix = self.make_list_of_matrices_for_agents(self.make_reward) # shape: (agent_size, state_size, 1)
        self.param_list = self.make_list_of_matrices_for_agents(self.make_feature_func) # (agent_size, state_size, feature_vector_size)
        self.avg_reward = np.average(self.reward_matrix, axis=0) # (state_size, 1)
        
        self.sol1 = -np.linalg.inv(self.phi.T@self.D_target@(self.gamma*self.P_target - np.eye(self.state_size))@self.phi)@self.phi.T@self.D_target@self.avg_reward

    def make_random_adjacenecy_graph(self) -> None:
        """
        undirected graph
        """
        size = self.agent_size
        A = np.zeros((size, size))
        D = np.zeros((size, size))
        
        for row in range(size):
            A[row] = np.random.randint(2, size = size)
            A[row,row] = 0 # make diagonal term 0
            for index in range(row):
                A[row, index] = A[index, row] # make graph undirected
            D[row,row] = sum(A[row]) # degree matrix
        
        self.adj_mat = A
        self.deg_mat = D

    def make_list_of_matrices_for_agents(self, func):
        matrices = []
        for agent in range(self.agent_size):
            matrices.append(func())
        return np.array(matrices)

if __name__ == '__main__':
    mamdp = MAMDP()
    
    # lists of parameters of all agents
    theta1 = np.random.rand(mamdp.agent_size, mamdp.feature_vector_size, 1)
    w1 = np.random.rand(mamdp.agent_size, mamdp.feature_vector_size, 1)     

    steps = 1000000
    error_vec1 = np.zeros(steps)
    
    for step in range(steps):
        step_size = 1/(step+100)
        
        theta1_bar = np.vstack(theta1)
        w1_bar = np.vstack(w1)
        laplacian_bar = np.kron(mamdp.laplacian, np.eye(mamdp.feature_vector_size))
        #laplacian_bar = np.kron(np.eye(mamdp.feature_vector_size), mamdp.laplacian)
        
        laplacian_theta = laplacian_bar @ theta1_bar
        laplacian_theta = laplacian_theta.reshape(mamdp.agent_size, mamdp.feature_vector_size, 1)
        
        laplacian_w = laplacian_bar @ w1_bar
        laplacian_w = laplacian_w.reshape(mamdp.agent_size, mamdp.feature_vector_size,1)

        for agent in range(mamdp.agent_size):
            # #Generates a random variable in 1, 2, ..., n given a prob distribution 
            # state = np.random.choice(mamdp.state_size, 1, p = mamdp.d)
            # state = state[0]
            # action = np.random.choice(mamdp.action_size, 1, p = mamdp.target[state])
            # action = action[0]
            # #next_state = np.random.choice(mdp.state_size, 1, p = mdp.trans_maxtices[action][:,state]) # Do I have to use Pb???
            # next_state = np.random.choice(mamdp.state_size, 1, p = mamdp.P_target[:,state])
            # next_state = next_state[0]

            # Importance sampling ratio
            # rho = mamdp.target[state][action]/mamdp.beta[state][action]
        
            theta1[agent] = theta1[agent] + step_size*(mamdp.phi.T@mamdp.D_target@(mamdp.gamma*mamdp.P_target - np.eye(mamdp.state_size))@mamdp.phi@theta1[agent] + mamdp.phi.T@mamdp.D_target@mamdp.reward_matrix[agent] - laplacian_theta[agent] - laplacian_w[agent])
            
            w1[agent] = w1[agent] + step_size*laplacian_w[agent]

        error1 = np.linalg.norm(mamdp.sol1-theta1[0], 2)
        error_vec1[step] = error1
    #print(theta1[0])
    plt.plot(error_vec1)
    #plt.yscale('log')
    plt.savefig('error.png')