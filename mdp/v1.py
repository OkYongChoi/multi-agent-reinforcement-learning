import numpy as np
import matplotlib.pyplot as plt

from mamdp import MAMDP


mamdp = MAMDP()

# lists of parameters of all agents
theta1 = np.random.rand(mamdp.agent_size, mamdp.feature_vector_size, 1)
w1 = np.random.rand(mamdp.agent_size, mamdp.feature_vector_size, 1)     

steps = 100000
error_vec1 = np.zeros(steps)

for step in range(steps):
    step_size = 1/(step+1)
    
    theta1_bar = np.vstack(theta1)
    w1_bar = np.vstack(w1)
    laplacian_bar = np.kron(mamdp.laplacian, np.eye(mamdp.feature_vector_size))
    #laplacian_bar = np.kron(np.eye(mamdp.feature_vector_size), mamdp.laplacian)
    
    laplacian_theta = laplacian_bar @ theta1_bar #Laplacian * theta_bar
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
        td_error = mamdp.phi.T@mamdp.D_target@(mamdp.gamma*mamdp.P_target - np.eye(mamdp.state_size))@mamdp.phi@theta1[agent] + mamdp.phi.T@mamdp.D_target@mamdp.reward_matrix[agent] 
        
        w1[agent] = w1[agent] + step_size*laplacian_theta[agent]
        theta1[agent] = theta1[agent] + step_size*(td_error - laplacian_theta[agent] - laplacian_w[agent])

    error = np.linalg.norm(mamdp.sol1-theta1[0], 2)
    
    error_vec1[step] = error
#print(theta1[0])
plt.plot(error_vec1)
plt.yscale('log')
plt.savefig('log_error1.eps')