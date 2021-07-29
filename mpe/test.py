import argparse
import time
import numpy as np
import random

import torch
from torch import optim
from qnet import QNet
from hyst_agent import HystAgent

from replay_memory import ReplayMemory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple_spread", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=60000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default=None, help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="/tmp/policy/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")
    return parser.parse_args()

def action_onehot(q_list):
    onehot = torch.zeros(3,5)
    for i in range(3):
        onehot[i][q_list[i]] = torch.tensor(1)
    return onehot



if __name__ == '__main__':
    arglist = parse_args()
    # Create environment
    env = make_env(arglist.scenario, arglist, arglist.benchmark)
    num_agents = env.n

    agent_list = [HystAgent() for i in range(num_agents)]

    False_list = [False]*num_agents
    done = False_list
    for episode in range(10000):
        state = env.reset()
        #env.render()
        state = torch.Tensor(state).to(device)
        time_step = 0
        reward_sum = 0
        while done == False_list and time_step < 100:
            
            # Get actions from agents and make them onehot-encoding
            actions = [q_agent.get_action(state[i]) for i, q_agent in enumerate(agent_list)]
            actions_list = action_onehot(actions)

            next_state, reward, done, info = env.step(actions_list)
            reward_sum += reward[0]
            
            next_state = torch.Tensor(next_state).to(device)
            reward = torch.Tensor(reward).to(device)
            actions = torch.IntTensor(actions).to(device)

            ## Push the transition into the replay memory of each agent.
            for i in range(num_agents):
                agent_list[i].push_sample(state[i], actions[i], reward[i], next_state[i], done[i])
                agent_list[i].train_model()
            
            if time_step % 50 ==0:
                for i in range(num_agents):
                    agent_list[i].update_target_model()

            state = next_state
            time_step += 1

            # for displaying learned policies
            if arglist.display:
                time.sleep(0.1)
                env.render()

        print(f"episode {episode} reward: {reward_sum}")

