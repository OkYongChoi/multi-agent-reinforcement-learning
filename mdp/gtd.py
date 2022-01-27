import numpy as np
import matplotlib.pyplot as plt
from mdp import MDP

if __name__ == '__main__':
    mdp = MDP()
    u1 = np.random.rand(mdp.feature_vector_size, 1)
    y1 = np.random.rand(mdp.feature_vector_size, 1)
    u2 = np.random.rand(mdp.feature_vector_size, 1)
    y2 = np.random.rand(mdp.feature_vector_size, 1)
    u3 = np.random.rand(mdp.feature_vector_size, 1)
    y3 = np.random.rand(mdp.feature_vector_size, 1)            

    steps = 100000
    error_vec1 = np.zeros(steps)
    error_vec2 = np.zeros(steps)
    error_vec3 = np.zeros(steps)
    for step in range(steps):
        #Generates a random variable in 1, 2, ..., n given a prob distribution 
        state = np.random.choice(mdp.state_size, 1, p = mdp.d_beta)
        state = state[0]
        action = np.random.choice(mdp.action_size, 1, p = mdp.beta[state])
        action = action[0]
        #next_state = np.random.choice(mdp.state_size, 1, p = mdp.trans_maxtices[action][:,state]) # Do I have to use Pb???
        next_state = np.random.choice(mdp.state_size, 1, p = mdp.P_beta[:,state])
        next_state = next_state[0]

        # Importance sampling ratio
        rho = mdp.target[state][action]/mdp.beta[state][action]
        
        # Step size
        step_size = 10/(step+100)

        # GTD (off-policy)
        delta = rho*mdp.reward[state] + mdp.gamma*rho*mdp.phi[next_state]@y1 - mdp.phi[state]@y1
        y1 = y1 + step_size * (mdp.phi[state].reshape(-1,1) - mdp.gamma*rho*mdp.phi[next_state].reshape(-1,1)) * mdp.phi[state]@u1
        u1 = u1 + step_size * (delta - mdp.phi[state]@u1) * mdp.phi[state].reshape(-1,1)

        # GTD3
        delta = rho*mdp.reward[state] + mdp.gamma*rho*mdp.phi[next_state]@y2 - mdp.phi[state]@y2
        y2 = y2 + step_size * ((mdp.phi[state].reshape(-1,1) - mdp.gamma*rho*mdp.phi[next_state].reshape(-1,1)) * mdp.phi[state]@u2 - mdp.phi[state].reshape(-1,1)*mdp.phi[state]@y2)
        u2 = u2 + step_size * delta * mdp.phi[state].reshape(-1,1)

        # GTD4
        sigma1 = 100/(steps+1000)
        delta = rho*mdp.reward[state] + mdp.gamma*rho*mdp.phi[next_state]@y3 - mdp.phi[state]@y3
        y3 = y3 + step_size * ((mdp.phi[state].reshape(-1,1) - mdp.gamma*rho*mdp.phi[next_state].reshape(-1,1)) * mdp.phi[state]@u3 - sigma1*mdp.phi[state].reshape(-1,1)*mdp.phi[state]@y3)
        u3 = u3 + step_size * (delta - mdp.phi[state]@u3) * mdp.phi[state].reshape(-1,1)


        error1 = np.linalg.norm(mdp.sol-y1 ,2)
        error2 = np.linalg.norm(mdp.sol-y2, 2)
        error3 = np.linalg.norm(mdp.sol-y3, 2)
        #error = (mdp.sol-y1).T @ (mdp.sol-y1)
        error_vec1[step] = error1
        error_vec2[step] = error2
        error_vec3[step] = error3
    
    plt.plot(error_vec1, 'b', label = 'GTD2')
    plt.plot(error_vec2, 'r', label = 'GTD3')
    plt.plot(error_vec3, 'g', label = 'GTD4')
    plt.legend()
    plt.yscale("log")
    plt.savefig('result.png')