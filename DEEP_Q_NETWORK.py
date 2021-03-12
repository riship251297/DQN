import gym
import frogger_env
import pygame
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import random
from random import sample

import matplotlib.pyplot as plt

BATCH_SIZE = 32
LR = 0.001
gamma = 0.99
BUFFER = 1000000

env = gym.make("frogger-v0").unwrapped
NUM_ACTIONS = env.action_space.n
NUM_STATES = env.observation_space.shape[0]
ENV_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample.shape

class DEEP_Q_NETWORK(nn.Module):
    def __init__(self):
        super(DEEP_Q_NETWORK, self).__init__()
        self.FULLY_CONNECTED_LAYER_1 = nn.Linear(NUM_STATES, 256)
        self.FULLY_CONNECTED_LAYER_2 = nn.Linear(256, 256)
        self.OUTPUT_LAYER = nn.Linear(256, NUM_ACTIONS)

    def forward(self, X):
        X = self.FULLY_CONNECTED_LAYER_1(X)
        X = F.relu(X)
        X = self.FULLY_CONNECTED_LAYER_2(X)
        X = F.relu(X)
        action_prob = self.OUTPUT_LAYER(X)
        return action_prob

class DQN:

    def __init__(self):
        super(DQN, self).__init__()
        self.LOCAL_NETWORK = DEEP_Q_NETWORK()
        self.TARGET_NETWORK = DEEP_Q_NETWORK()

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((BUFFER, NUM_STATES * 2 + 2))
        self.epsilon = 1
        self.eps_dec = 0.0001
        self.eps_min = 0.01
        self.BATCH_SIZE = 32
        self.optimizer = Adam(self.LOCAL_NETWORK.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

        # <------------------------------------  ACTION_SELECTION  ---------------------------------->

    def action_selection(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        if np.random.random() >= self.epsilon:
            action_value = self.LOCAL_NETWORK.forward(state)
            action = torch.max(action_value, 1)[1].data.numpy()
            if ENV_SHAPE == 0:
                action = action[0]
            else:
                action.reshape(ENV_SHAPE)

        else:
            action = np.random.randint(0, NUM_ACTIONS)
            if ENV_SHAPE == 0:
                action = action
            else:
                action.reshape(ENV_SHAPE)
        return action

    def action_testing_selection(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)  # get a 1D array
        action_value = self.LOCAL_NETWORK.forward(state)
        action = torch.max(action_value, 1)[1].data.numpy()
        if ENV_SHAPE == 0:
            action = action[0]
        else:
            action.reshape(ENV_SHAPE)
        return action

        # <------------------------------------  STORING EXPERIENCES  ---------------------------------->

    def experiences(self, state, action, reward, next_state):
        transition_step = np.hstack((state, [action, reward], next_state))
        index = self.memory_counter % BUFFER
        self.memory[index, :] = transition_step
        self.memory_counter += 1

        # <------------------------------------  DQN TRAINING  ------------------------------------->

    def training_module(self):
        if self.memory_counter < self.BATCH_SIZE:
            return

        if self.learn_step_counter % 100 == 0:
            self.TARGET_NETWORK.load_state_dict(self.LOCAL_NETWORK.state_dict())
        self.learn_step_counter += 1

        index = np.random.choice(BUFFER, BATCH_SIZE)
        memory = self.memory[index, :]
        state = torch.FloatTensor(memory[:, :NUM_STATES])
        action = torch.LongTensor(memory[:, NUM_STATES:NUM_STATES + 1].astype(int))
        reward = torch.FloatTensor(memory[:, NUM_STATES + 1:NUM_STATES + 2])
        next_state = torch.FloatTensor(memory[:, -NUM_STATES:])

        Q_VALUE = self.LOCAL_NETWORK(state).gather(1, action)
        Q_NEXT = self.TARGET_NETWORK(next_state).detach()
        TARGET = reward + gamma * Q_NEXT.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.loss_func(Q_VALUE, TARGET)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.eps_min:
            self.epsilon = self.epsilon - self.eps_dec
        else:
            self.epsilon = self.eps_min

        # print('epsilon value', self.epsilon)
        return loss

        # <------------------------------------  MAIN FUNCTION  ---------------------------------->


if __name__ == '__main__':

    dqn_agent = DQN()

    r_avg_reward = []
    global_step_set = []
    losses = []
    REWARD_SET = []


    # <-----------------------------  PLOTTING FUNCTIONS  --------------------------->

    def PLOT_ROLLOUT_RESULT(global_step_set, r_avg_reward):
        plt.title('ROLLOUT RESULT')
        plt.xlabel('ITERATIONS')
        plt.ylabel('AVERAGE REWARD')
        plt.plot(global_step_set, r_avg_reward)
        plt.show()


    def PLOT_TRAINING_RESULT(episode_set, REWARD_SET):
        plt.title('TOTAL RETURNS')
        plt.xlabel('EPISODES')
        plt.ylabel('REWARDS')
        plt.plot(episode_set, REWARD_SET)
        plt.show()


    # <-----------------------------  SAVING THE REGULAR AND BEST CHECKPOINTS  --------------------------->

    def save_checkpoint(model, global_step):
        filename = '/Users/icg/desktop/my_checkpoints/checkpoint_' + str(global_step) + '.pth.tar'

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': dqn_agent.optimizer.state_dict(),
            'loss': loss,
        }, filename)


    def save_checkpoint_best(model, global_step):
        filename = '/Users/icg/desktop/my_checkpoints/checkpoint_best_' + str(global_step) + '.pth.tar'
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': dqn_agent.optimizer.state_dict(),
            'loss': loss,
        }, filename)


    # <-----------------------------  LOADING THE CHECKPOINT  --------------------------->

    dqn_agent.LOCAL_NETWORK.load_state_dict(
        torch.load("/Users/icg/desktop/my_checkpoints/checkpoint_372992.pth.tar"), strict=False)
    print("<----------SUCCESSFULLY LOADED--------------->")


    # for name, param in dqn.eval_net.state_dict().items():
    #     print(name, param.data)

    # <------------------------------------ ROLLOUT CALCULATING FUNCTION ------------------------------------>

    def rollout(global_step):
        global_step_set.append(global_step)
        rollout_reward_set = []
        state = env.reset()
        done = False
        for rollout_episode in range(100):
            rollout_ep_reward = 0
            while not done:
                action = dqn_agent.action_testing_selection(state)
                next_state, reward, done, info = env.step(action)
                rollout_ep_reward += reward
                state = next_state
            rollout_reward_set.append(rollout_ep_reward)
            pass
        r_avg_reward.append(sum(rollout_reward_set) / len(rollout_reward_set))


    # <------------------------------------------------------------------------------------------------------------->

    go_for_rollouts = False
    env = gym.make("frogger-v0")
    global_step = 0
    rollout_duration = False
    episodes, best_reward = 2500, -1000
    episode_set = []
    # <-------------------------------------------- START OF THE EPISODES ----------------------------------------->

    for i in range(episodes):

        if go_for_rollouts:
            rollout(global_step)
            go_for_rollouts = False

        print("EPISODE_NUMBER", i)
        state = env.reset()
        total_rewards = []
        ep_reward = 0
        done = False

        while not done:
            global_step += 1
            if global_step % 100 == 0:  # ready for rollouts but wait till episode terminates
                go_for_rollouts = True
                rollout(global_step)

            action = dqn_agent.action_selection(state)
            next_state, reward, done, info = env.step(action)
            env.render()
            # print('--', global_step)
            dqn_agent.experiences(state, action, reward, next_state)
            ep_reward += reward
            loss = dqn_agent.training_module()
            losses.append(loss)

            if done:
                REWARD_SET.append(ep_reward)

                if i % 100 == 0:
                    save_checkpoint(dqn_agent.LOCAL_NETWORK, global_step)

                if ep_reward > best_reward:
                    best_reward = ep_reward
                    save_checkpoint_best(dqn_agent.LOCAL_NETWORK, global_step)

                    state = next_state
        episode_set.append(i)

        if i == 2499:
            PLOT_ROLLOUT_RESULT(global_step_set, r_avg_reward)
            PLOT_TRAINING_RESULT(episode_set, REWARD_SET)
    plt.title("LOSS DURING TRAINING")
    plt.plot(losses)
    plt.show()
