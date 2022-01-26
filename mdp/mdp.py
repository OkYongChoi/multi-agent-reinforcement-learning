import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


class MDP:
    def __init__(self, state_size = 100, action_size = 10, feature_vector_size = 10): 
        self.state_size = state_size
        self.action_size = action_size
        self.feature_vector_size = feature_vector_size

        self.target = self.make_policy()
        self.beta = self.make_policy()

        self.trans_maxtices = self.transition_matrices()
        self.P_target = self.trans_matrix_under_policy(self.target)
        self.P_beta = self.trans_matrix_under_policy(self.beta)

        self.phi = self.make_feature_func()

        self.gamma = 0.9
        self.reward = self.make_reward()

        ## stationary distribution
        self.d_beta = self.stationary_dist(self.P_beta)
        self.D_beta = self.stationary_dist_matrix(self.d_beta)
        self.d_target = self.stationary_dist(self.P_beta)
        self.D_target = self.stationary_dist_matrix(self.d_target)

        #sol=-inv(phi'*D*(gamma*P_target - I(state_size))*phi)*Phi'*D*reward;
        self.sol = -np.linalg.inv(self.phi.T@self.D_beta@(self.gamma*self.P_target - np.eye(self.state_size))@self.phi)@self.phi.T@self.D_beta@self.reward

                                    
    def make_transition_matrix(self):
        """
        To make a transition matrix.
        number of matrices = number of action size
        size of each matrix = (state_size) * (stae_size)
        [i,j] represents the probability 
        """
        # Make random matrix of state x state
        P = np.random.rand(self.state_size, self.state_size)
        
        # Make sure each row forms a probability of going from i to j
        for start_state in range(self.state_size):
            P[start_state] = P[start_state] / sum(P[start_state])

        return P.T

    def transition_matrices(self):
        P_list = []
        for action in range(self.action_size):
            P = self.make_transition_matrix()
            P_list.append(P)
        return np.array(P_list)

    def make_policy(self):
        """
        To build random policy for both target and behavior policy.
        Each row of matrix represents the policy(probability) at each state.
        """
        P = np.random.rand(self.state_size, self.action_size)
        for state in range(self.state_size):
            P[state] = P[state] / sum(P[state])
        return P

    def trans_matrix_under_policy(self, policy):
        """
        To build transition matrix for both target and behavior policy.
        This method returns the transition probability matrix under the certain policy,
        in which [i,j] components represents the probability of going "from j to i" (not from i to j)
        """
        trans_matrix = 0
        P = deepcopy(self.trans_maxtices.transpose([0,2,1])) 
        for action in range(self.action_size):
            for state in range(self.state_size):
                P[action][state] = policy[state][action] * P[action][state]
            trans_matrix += P[action]
        
        trans_matrix = np.array(trans_matrix).T # This way of matrix will be used, which makes [i,j] represent the probability of going from j to i
        return trans_matrix

    def make_reward(self):
        reward = 1-2*np.random.rand(self.state_size, 1)
        sparsity = 0.2
        
        # make rewards sparse
        for state in range(self.state_size):
            if abs(reward[state]) < sparsity:
                reward[state] = 0
        return reward

    def make_feature_func(self):
        """ 
        Make feature functions and construct the feature matrix of which 
        size is (state_size x feature_vector_size)
        """
        # Make sure the feature function sparse
        phi = 1-2*np.random.rand(1, self.state_size)
        # while(np.linalg.matrix_rank(phi) < self.feature_vector_size):
        for i in range(1, self.feature_vector_size):
            vec = 1-2*np.random.rand(1, self.state_size)
            phi = np.append(phi, vec, axis = 0)
        return phi.transpose()

    def stationary_dist(self, P):
        """ Stationary distribution under behavior policy """
        d = np.ones(self.state_size) / self.state_size
        for i in range(10000):
            d = P @ d
        return d
        
    def stationary_dist_matrix(self,d):
        """ Stationary distribution matirx having stationary distribution at diagonal elements"""
        D = np.zeros((self.state_size, self.state_size))
        for index in range(self.state_size):
            D[index][index] = d[index]
        return D


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
    #plt.yscale("log")
    plt.savefig('result.png')
