#!/usr/bin/env python
# coding=utf-8

import sys,os
curr_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)

import torch
import datetime
from PPO.utils import save_results,make_dir
from PPO.agent import PPO

from PPO.env import Topo,Floyd_L
import numpy as np
import copy
import math

curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

class PPOConfig:
    def __init__(self) -> None:
        self.algo_name = 'PPO'
        self.env_name = 'Topology'
        self.continuous = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_eps = 10
        self.test_eps = 1
        self.batch_size = 2**10
        self.gamma = 1
        self.actor_lr = 0.0003
        self.critic_lr = 0.0003
        self.gae_lambda = 0.95
        self.policy_clip = 0.2
        self.n_epochs = 1
        self.update_fre = 2**12
        self.input_dim = 0
        self.hidden_dim = 0
        self.output_dim = 0

class PlotConfig:
    def __init__(self) -> None:
        self.algo_name = "PPO"
        self.env_name = 'Topology'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.result_path = curr_path+"/outputs/" + self.env_name + \
            '/'+curr_time+'/results/'
        self.model_path = curr_path+"/outputs/" + self.env_name + \
            '/'+curr_time+'/models/'
        self.save = True

def env_agent_config(cfg, paths, weights):
    env = Topo(paths, weights)
    cfg.input_dim = env.state_dim
    cfg.hidden_dim = env.link_dim
    cfg.output_dim = env.action_dim
    
    print('update_fre:', cfg.update_fre)
    print('batch_size:', cfg.batch_size)
    agent = PPO(cfg)
    return env,agent

def train(cfg,env,agent):
    print('Start training...')
    print(f'Env: {cfg.env_name}, Alg: {cfg.algo_name}, Dev: {cfg.device}')
    rewards = []
    ma_rewards = []
    steps = 0
    for i_ep in range(cfg.train_eps):
        state = env.reset()
        done = False
        ep_reward = 0
        while not done:
            action, prob, val = agent.choose_action(state)
            state_, reward, done, _ = env.step(action)
            steps += 1
            ep_reward += reward
            agent.memory.push(state, action, prob, val, reward, done)
            if steps % cfg.update_fre == 0:
                agent.update()
            state = state_
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)
        print('Epoch: {}/{}, Reward: {}'.format(i_ep+1, cfg.train_eps, ep_reward))
    print('Complete training.')
    return rewards,ma_rewards

def eval(cfg,env,agent):
    print('Start testing...')
    print(f'Env: {cfg.env_name}, Alg: {cfg.algo_name}, Dev: {cfg.device}')
    rewards = []
    ma_rewards = []
    for i_ep in range(cfg.test_eps):
        state = env.reset()
        done = False
        ep_reward = 0
        while not done:
            action, prob, val = agent.choose_action(state)
            state_, reward, done, _ = env.step(action)
            ep_reward += reward
            state = state_
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)
        print('Epoch: {}/{}, Reward: {}'.format(i_ep+1, cfg.test_eps, ep_reward))
    print('Complete testing.')
    return rewards,ma_rewards
