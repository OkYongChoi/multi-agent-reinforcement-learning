import numpy as np
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

        # sol = -inv(phi'*D*(gamma*P_target - I(state_size))*phi)*Phi'*D*reward;
        self.sol = -np.linalg.inv(self.phi.T@self.D_beta@(self.gamma*self.P_target - np.eye(self.state_size))@self.phi)@self.phi.T@self.D_beta@self.reward

                                    
    def make_transition_matrix(self):
        """
        To make a transition matrix.
        number of matrices = number of action size
        size of each matrix = (state_size) * (state_size)
        [i, j] represents the probability 
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
        
    def stationary_dist_matrix(self, d):
        """ Stationary distribution matrix having stationary distribution at diagonal elements"""
        D = np.zeros((self.state_size, self.state_size))
        for index in range(self.state_size):
            D[index][index] = d[index]
        return D

