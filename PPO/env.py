#!/usr/bin/env python
# coding=utf-8

import sys,os
curr_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)

import datetime

import numpy as np
import copy
import math

curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def Floyd_L(dist):
    for k in range(len(dist)):
        for i in range(len(dist)):
            for j in range(len(dist)):
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]

class Topo:
    def __init__(self, paths, weights):
        self.paths = paths
        self.weights = weights
        max_node = 0
        for i in range(len(self.paths)):
            if self.paths[i][0] > max_node:
                max_node = self.paths[i][0]
            if self.paths[i][1] > max_node:
                max_node = self.paths[i][1]
        self.node_number = max_node + 1
        self.MAX = 10 ** 10
        self.adj = np.ones((self.node_number,self.node_number))*self.MAX
        self.prohibit = []
        for i in range(self.node_number):
            self.adj[i][i] = 0
            self.prohibit.append(i*self.node_number+i)
            for j in range(self.node_number):
                if (i,j) in self.paths:
                    self.adj[i][j] = self.weights[self.paths.index((i,j))]
                    self.prohibit.append(i*self.node_number+j)
                if (j,i) in self.paths:
                    self.adj[i][j] = self.weights[self.paths.index((j,i))]
                    self.prohibit.append(i*self.node_number+j)
        self.dist = copy.deepcopy(self.adj)
        Floyd_L(self.dist)
        
        for i in range(self.node_number):
            for j in range(self.node_number):
                if self.dist[i][j] == self.MAX:
                    self.prohibit.append(i*self.node_number+j)
        
        self.dist_original = copy.deepcopy(self.dist)
        
        self.actions = {}
        self.action_index = {} # idnex of action
        count = 0
        for i in range(self.node_number-1):
            for j in range(i+1,self.node_number):
                act = i*self.node_number+j
                if act not in self.prohibit and self.dist[i][j] != self.MAX:
                    self.actions[count] = act
                    self.action_index[act] = count # index
                    count = count + 1
        
        print('action_number:',len(self.actions))
        self.state_dim = self.node_number * self.node_number
        self.action_dim = len(self.actions)
        self.link_dim = math.ceil(2.5*self.node_number)
        
        print('node_number:',self.node_number)
        print('state_dim:',self.state_dim)
        
        self.action_flag = {}
        for i in range(len(self.actions)):
            self.action_flag[i] = False
        self.action_flag_number = 0
    
    def reset(self):
        self.dist = copy.deepcopy(self.dist_original)
        state = self.print_state()
        
        for i in range(len(self.actions)):
            self.action_flag[i] = False
        self.action_flag_number = 0
        return state
    
    def step(self, action):
        if self.action_flag[action]:
            state = self.print_state()
            reward = -0.05
            return state, reward, False, state
        else:
            state_temp = copy.deepcopy(self.dist)
            act = self.actions[action]
            r_idx = act // self.node_number
            c_idx = act % self.node_number
            
            if self.is_available(state_temp, r_idx, c_idx):
                self.dist[r_idx][c_idx] = self.dist[r_idx][c_idx] - 1
                self.dist[c_idx][r_idx] = self.dist[c_idx][r_idx] - 1
                self.Floyd_update(r_idx, c_idx)
                reward = 1
                state = self.print_state()
                return state, reward, False, state
            else:
                state = self.print_state()
                done = self.is_done()
                if done == True:
                    reward = 0
                else:
                    reward = -0.05
                return state, reward, done, state
    
    def is_available(self, state_temp, i, j):
        state_temp[i][j] = state_temp[i][j] - 1
        state_temp[j][i] = state_temp[j][i] - 1
        if state_temp[i][j] <= 0:
            act = i * self.node_number + j
            idx = self.action_index[act]
            self.action_flag[idx] = True
            self.action_flag_number += 1
            return False
        else:
            return_temp = self.Floyd_trunc(state_temp, i, j)
            if return_temp == False:
                act = i * self.node_number + j
                idx = self.action_index[act]
                self.action_flag[idx] = True
                self.action_flag_number += 1
            return return_temp
    
    def is_done(self):
        for i in range(len(self.actions)):
            if self.action_flag[i]:
                continue
            else:
                act = self.actions[i]
                r_idx = act // self.node_number
                c_idx = act % self.node_number
                state_temp = copy.deepcopy(self.dist)
                if self.is_available(state_temp, r_idx, c_idx):
                    return False
        return True
    
    def print_state(self):
        state = copy.deepcopy(self.dist)
        
        state_max = 0
        for i in range(self.node_number):
            for j in range(self.node_number):
                if state[i][j] > state_max:
                    state_max = state[i][j]
        state_max = state_max * 2
        for i in range(self.node_number):
            for j in range(self.node_number):
                if state[i][j] == self.MAX:
                    state[i][j] = -1
        
        return state.flatten()
    
    def print_dist(self):
        return self.dist
    
    def Floyd_update(self, i_idx, j_idx):
        for i in range(self.node_number):
            for j in range(self.node_number):
                if self.dist[i][j] > self.dist[i][i_idx] + self.dist[i_idx][j]:
                    self.dist[i][j] = self.dist[i][i_idx] + self.dist[i_idx][j]
        for i in range(self.node_number):
            for j in range(self.node_number):
                if self.dist[i][j] > self.dist[i][j_idx] + self.dist[j_idx][j]:
                    self.dist[i][j] = self.dist[i][j_idx] + self.dist[j_idx][j]
    
    def Floyd_trunc(self, dist, i_idx, j_idx):
        for i in range(len(dist)):
            for j in range(len(dist)):
                if dist[i][j] > dist[i][i_idx] + dist[i_idx][j]:
                    dist[i][j] = dist[i][i_idx] + dist[i_idx][j]
                    act = i*len(dist)+j
                    act2 = j*len(dist)+i
                    if act in self.prohibit or act2 in self.prohibit:
                        return False
        for i in range(len(dist)):
            for j in range(len(dist)):
                if dist[i][j] > dist[i][j_idx] + dist[j_idx][j]:
                    dist[i][j] = dist[i][j_idx] + dist[j_idx][j]
                    act = i*len(dist)+j
                    act2 = j*len(dist)+i
                    if act in self.prohibit or act2 in self.prohibit:
                        return False
        return True
