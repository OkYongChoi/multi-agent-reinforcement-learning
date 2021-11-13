import numpy as np
import matplotlib.pyplot as plt

from mamdp import MAMDP


mamdp = MAMDP()

# lists of parameters of all agents
theta = np.random.rand(mamdp.agent_size, mamdp.feature_vector_size, 1)
w = np.random.rand(mamdp.agent_size, mamdp.feature_vector_size, 1)     
v = np.random.rand(mamdp.agent_size, mamdp.feature_vector_size, 1)     

steps = 100000
error_vec = np.zeros(steps)

for step in range(steps):
    step_size = 10/(step+100)
    
    theta_bar = np.vstack(theta)
    w_bar = np.vstack(w)
    v_bar = np.vstack(v)
    laplacian_bar = np.kron(mamdp.laplacian, np.eye(mamdp.feature_vector_size))
    
    laplacian_theta = laplacian_bar @ theta_bar
    laplacian_theta = laplacian_theta.reshape(mamdp.agent_size, mamdp.feature_vector_size, 1)
    
    laplacian_w = laplacian_bar @ w_bar
    laplacian_w = laplacian_w.reshape(mamdp.agent_size, mamdp.feature_vector_size,1)

    laplacian_v = laplacian_bar @ v_bar
    laplacian_v = laplacian_w.reshape(mamdp.agent_size, mamdp.feature_vector_size,1)

    # state = np.random.choice(mamdp.state_size, 1, p = mamdp.d_target)[0]
    # action = np.random.choice(mamdp.action_size, 1, p = mamdp.target[state])[0]
    # next_state = np.random.choice(mamdp.state_size, 1, p = mamdp.trans_maxtices[action][:,state])[0]    
    for agent in range(mamdp.agent_size):
        #Generates a random variable in 1, 2, ..., n given a prob distribution 
        state = np.random.choice(mamdp.state_size, 1, p = mamdp.d_target)[0]
        action = np.random.choice(mamdp.action_size, 1, p = mamdp.target[state])[0]
        next_state = np.random.choice(mamdp.state_size, 1, p = mamdp.trans_maxtices[action][:,state])[0]

        # Importance sampling ratio
        # rho = mamdp.target[state][action]/mamdp.beta[state][action]
        
        #delta = rho*s.reward(state) + s.gamma*rho*s.Phi(next_state,:)*y1 - s.Phi(state,:)*y1;
        #y1 = y1 + step_size*(s.Phi(state,:)' - s.gamma*rho*s.Phi(next_state,:)') * (s.Phi(state,:)*u1);
        #u1 = u1 + step_size*(delta - s.Phi(state,:)*u1)*s.Phi(state,:)';
        #
        
        delta = mamdp.reward_matrix[agent][state] + mamdp.gamma*mamdp.phi[next_state]@laplacian_theta[agent] - mamdp.phi[state]@laplacian_theta[agent]
        
        v[agent] = v[agent] + step_size*laplacian_w[agent]
        w[agent] = w[agent] + step_size*(theta[agent] - w[agent] - laplacian_w[agent] - laplacian_v[agent])
        theta[agent] = theta[agent] + step_size*(delta*mamdp.phi[state].reshape(-1,1) - laplacian_theta[agent])
    
    error = np.linalg.norm(mamdp.sol1-w[0], 2)
    error_vec[step] = error

plt.plot(error_vec)
plt.yscale('log')
plt.savefig('log_error3.eps')