# -*- coding: utf-8 -*-

import sys,os
curr_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import math
import copy
import PPO.train as PPO

def dataset_generator(file_name):
    
    fh = open(file_name, 'r')
    node_origin = []
    node_dict_origin = {}
    edge_dict_origin = {}
    for line in fh:
        line = line.strip('\n')
        line = line.rstrip()
        words = line.split()
        if words[0] not in node_dict_origin:
            node_dict_origin[words[0]] = 1
            node_origin.append(words[0])
        else:
            node_dict_origin[words[0]] += 1
        if words[1] not in node_dict_origin:
            node_dict_origin[words[1]] = 1
            node_origin.append(words[1])
        else:
            node_dict_origin[words[1]] += 1
        temp = (words[0], words[1])
        edge_dict_origin[temp] = float(words[2])
    
    node_number = len(node_origin)
    for key in node_dict_origin:
        node_dict_origin[key] = node_dict_origin[key] // 2
    for i in range(node_number-1):
        for j in range(i+1,node_number):
            if (node_origin[i],node_origin[j]) in edge_dict_origin and (node_origin[j],node_origin[i]) in edge_dict_origin:
                del edge_dict_origin[(node_origin[j],node_origin[i])]
    
    node = []
    node_dict = {}
    edge_dict = {}
    
    node_id_set = set()
    
    node_id = random.randint(0,node_number-1)
    node.append(node_origin[node_id])
    node_dict[node[0]] = node_dict_origin[node_origin[node_id]]
    node_id_set.add(node_id)
    
    # while len(node) < math.ceil(float(node_number)/3):  # /3 for AS3257
    while len(node) < math.ceil(float(node_number)/2):  # /2 for AS3967 and AS1221
        node_id = random.randint(0,node_number-1)
        if node_id in node_id_set:
            continue
        else:
            new_node = node_origin[node_id]
            for i in range(len(node)):
                if (node[i], new_node) in edge_dict_origin:
                    node.append(new_node)
                    node_dict[new_node] = node_dict_origin[new_node]
                    node_id_set.add(node_id)
                    break
                elif (new_node, node[i]) in edge_dict_origin:
                    node.append(new_node)
                    node_dict[new_node] = node_dict_origin[new_node]
                    node_id_set.add(node_id)
                    break
    
    node_number = len(node)
    
    for i in range(node_number-1):
        for j in range(i+1,node_number):
            if (node[i], node[j]) in edge_dict_origin:
                edge_dict[(node[i], node[j])] = edge_dict_origin[(node[i], node[j])]
            elif (node[j], node[i]) in edge_dict_origin:
                edge_dict[(node[j], node[i])] = edge_dict_origin[(node[j], node[i])]
    
    MAX = 10 ** 10
    adj = np.ones((node_number,node_number)) * MAX
    for i in range(node_number):
        adj[i][i] = 0
    for i in range(node_number):
        for j in range(node_number):
            if (node[i],node[j]) in edge_dict:
                adj[i][j] = edge_dict[(node[i],node[j])]
                adj[j][i] = edge_dict[(node[i],node[j])]
        
    
    dist = copy.deepcopy(adj)
    
    for k in range(len(dist)):
        for i in range(len(dist)):
            for j in range(len(dist)):
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    
    host = []
    for i in range(node_number):
        host.append(i)
    host_number = len(host)
    node_pair_number = int(host_number*(host_number-1)/2)
    node_pair = []
    
    node_pair_vector = np.zeros((node_pair_number,host_number))
    label = np.zeros(node_pair_number)
    c = 0
    for i in range(len(host)-1):
        for j in range(i+1,len(host)):
            if dist[host[i]][host[j]] != MAX:
                node_pair.append((i,j))
                node_pair_vector[c][i] = 1
                node_pair_vector[c][j] = 1
                label[c] = dist[host[i]][host[j]]
                c += 1
    
    node_pair_number = len(node_pair)
    index = []
    for i in range(node_pair_number):
        index.append(i)
    node_pair_vector = node_pair_vector[index,:]
    label = label[index]
    
    return host_number,node_pair_vector,label

def dataset_separator(host_number,one_hot,label,percentage,sampling):
    edge_list = []
    for i in range(one_hot.shape[0]):
        edge_list.append(np.where(one_hot[i] == 1)[0])
    
    # random
    if sampling == 'random':
        edge_list_set = [[] for i in range(host_number)]
        for i in range(one_hot.shape[0]):
            node1_idx = edge_list[i][0]
            node2_idx = edge_list[i][1]
            edge_list_set[node1_idx].append((i,node2_idx))
            edge_list_set[node2_idx].append((i,node1_idx))
        
        index = []
        delete_set = []
        for i in range(host_number):
            if len(edge_list_set[i]) == 0:
                continue
            if i not in delete_set:
                random_idx = random.randint(0,len(edge_list_set[i])-1)
                temp_index = edge_list_set[i][random_idx][0]
                index.append(temp_index)
                delete_index = edge_list_set[i][random_idx][1]
                if delete_index not in delete_set:
                    delete_set.append(delete_index)
                delete_set.append(i)
        
        trainset_number = math.ceil(one_hot.shape[0] * percentage)
        rest_number = trainset_number - len(index)
        for i in range(rest_number):
            temp_index = random.randint(0,one_hot.shape[0]-1)
            while temp_index in index:
                temp_index = random.randint(0,one_hot.shape[0]-1)
            index.append(temp_index)
    # monitor
    elif sampling == 'monitor':
        edge_list_set = [[] for i in range(host_number)]
        for i in range(one_hot.shape[0]):
            node1_idx = edge_list[i][0]
            node2_idx = edge_list[i][1]
            edge_list_set[node1_idx].append((i,node2_idx))
            edge_list_set[node2_idx].append((i,node1_idx))
        
        monitor_set = []
        index = []
        
        delete_set = []
        while len(delete_set)<host_number:
            temp = random.randint(0,host_number-1)
            while temp in delete_set:
                temp = random.randint(0,host_number-1)
            monitor_set.append(temp)
            delete_set.append(temp)            
            for i in range(len(edge_list_set[temp])):
                temp_index = edge_list_set[temp][i][0]
                delete_index = edge_list_set[temp][i][1]
                index.append(temp_index)
                if delete_index not in delete_set:
                    delete_set.append(delete_index)
        
        while len(index)/one_hot.shape[0] < percentage:
            temp = random.randint(0,host_number-1)
            while temp in monitor_set:
                temp = random.randint(0,host_number-1)
            monitor_set.append(temp)
            for i in range(len(edge_list_set[temp])):
                temp_index = edge_list_set[temp][i][0]
                index.append(temp_index)
    
    trainset_onehot = one_hot[index,:]
    trainset_label = label[index]
    testset_onehot = np.delete(one_hot,index,0)
    testset_label = np.delete(label,index,0)
    return trainset_onehot,trainset_label,testset_onehot,testset_label

def print_dist(name, dist):
    print(name+':')
    for i in range(len(dist)):
        print(dist[i])

def sum_error(dist1,dist2):
    MAX = 10 ** 10
    s = 0
    for i in range(len(dist1)):
        for j in range(len(dist1)):
            if dist1[i][j] < MAX:
                s = s + abs(dist1[i][j]-dist2[i][j])
    return s

def sum_error2(dist1, dist2, testsample_number):
    MAX = 10 ** 10
    s = 0
    for i in range(len(dist1)):
        for j in range(len(dist1)):
            if dist1[i][j] == 0 or dist1[i][j] >= MAX:
                continue
            else:
                s = s + (abs(dist1[i][j]-dist2[i][j])/dist2[i][j])
    MAPE = s*100/testsample_number
    return MAPE

def average_weighting(dist_pat, dist_rl, MAX):
    dist_pat_rl = copy.deepcopy(dist_rl)
    for i in range(len(dist_pat_rl)):
        for j in range(len(dist_pat_rl[i])):
            if  dist_pat_rl[i][j] == MAX or dist_pat[i][j] == MAX:
                dist_pat_rl[i][j] == MAX
            else:
                dist_pat_rl[i][j] = (dist_pat_rl[i][j] + dist_pat[i][j]) / 2
    return dist_pat_rl

def alpha_weighting(dist_pat, dist_rl, MAX):
    dist_pat_rl = copy.deepcopy(dist_rl)
    w_pat = np.sum(np.sum(dist_pat))
    w_rl = np.sum(np.sum(dist_rl))
    w_pat_rl = float(w_rl) / w_pat
    for i in range(len(dist_pat_rl)):
        for j in range(len(dist_pat_rl[i])):
            if  dist_pat_rl[i][j] == MAX or dist_pat[i][j] == MAX:
                dist_pat_rl[i][j] == MAX
            else:
                dist_pat_rl[i][j] = w_pat_rl*dist_pat_rl[i][j] + (1-w_pat_rl)*dist_pat[i][j]
    return dist_pat_rl

def alpha_weighting2(dist_pat, dist_rl, MAX, a):
    dist_pat_rl = copy.deepcopy(dist_rl)
    for i in range(len(dist_pat_rl)):
        for j in range(len(dist_pat_rl[i])):
            if  dist_pat_rl[i][j] == MAX or dist_pat[i][j] == MAX:
                dist_pat_rl[i][j] == MAX
            else:
                if dist_pat[i][j] == 0:
                    dist_pat_rl[i][j] = 0
                else:
                    alpha = float(a * dist_pat_rl[i][j] + dist_pat[i][j]) / (a * dist_pat[i][j] + dist_pat_rl[i][j])
                    dist_pat_rl[i][j] = alpha*dist_pat_rl[i][j] + (1-alpha)*dist_pat[i][j]
    return dist_pat_rl

def Floyd_detec(dist_ppo, ii, jj, Prohibited_list):
    MAX = 10 ** 10
    dist_temp = copy.deepcopy(dist_ppo)
    if dist_temp[ii][jj] == MAX:
        return False
    dist_temp[ii][jj] -= 1
    dist_temp[jj][ii] -= 1
    if dist_temp[ii][jj] <= 0:
        return False
    else:
        for i in range(len(dist_temp)):
            for j in range(len(dist_temp)):
                if dist_temp[i][j] > dist_temp[i][ii] + dist_temp[ii][j]:
                    dist_temp[i][j] = dist_temp[i][ii] + dist_temp[ii][j]
                    if (i,j) in Prohibited_list:
                        return False
        for i in range(len(dist_temp)):
            for j in range(len(dist_temp)):
                if dist_temp[i][j] > dist_temp[i][jj] + dist_temp[jj][j]:
                    dist_temp[i][j] = dist_temp[i][jj] + dist_temp[jj][j]
                    if (i,j) in Prohibited_list:
                        return False
        return True

def testset_estimator(trainset_onehot,trainset_label,testset_onehot,testset_label):
    es_onehot = np.zeros((testset_onehot.shape[0],testset_onehot.shape[1]))
    es_label = np.zeros(testset_label.shape[0])
    for i in range(testset_onehot.shape[0]):
        for j in range(testset_onehot.shape[1]):
            es_onehot[i][j] = testset_onehot[i][j]
    for i in range(testset_label.shape[0]):
        es_label[i] = testset_label[i]
    MAX = 10 ** 10
    
    node_number = trainset_onehot.shape[1]
    train_list = []
    test_list = []
    for i in range(trainset_onehot.shape[0]):
        train_list.append(np.where(trainset_onehot[i] == 1)[0])
    for i in range(testset_onehot.shape[0]):
        test_list.append(np.where(testset_onehot[i] == 1)[0])
    
    adj = np.ones((node_number,node_number)) * MAX
    for i in range(node_number):
        adj[i][i] = 0
    for i in range(trainset_onehot.shape[0]):
        adj[train_list[i][0]][train_list[i][1]] = trainset_label[i]
        adj[train_list[i][1]][train_list[i][0]] = trainset_label[i]
    
    # ground truth
    ground_truth = copy.deepcopy(adj)
    for i in range(testset_onehot.shape[0]):
        ground_truth[test_list[i][0]][test_list[i][1]] = testset_label[i]
        ground_truth[test_list[i][1]][test_list[i][0]] = testset_label[i]
    
    # PAT
    dist_pat = copy.deepcopy(adj)
    PPO.Floyd_L(dist_pat)
    for i in range(testset_onehot.shape[0]):
        es_label[i] = dist_pat[test_list[i][0]][test_list[i][1]]
    
    # PPO
    paths = []
    weights = []
    for i in range(trainset_onehot.shape[0]):
        paths.append((train_list[i][0],train_list[i][1]))
        weights.append(trainset_label[i])
    
    cfg  = PPO.PPOConfig()
    plot_cfg = PPO.PlotConfig()
    
    env,agent = PPO.env_agent_config(cfg, paths, weights)
    rewards, ma_rewards = PPO.train(cfg, env, agent)
    PPO.make_dir(plot_cfg.result_path, plot_cfg.model_path)
    agent.save(path=plot_cfg.model_path)
    PPO.save_results(rewards, ma_rewards, tag='train', path=plot_cfg.result_path)

    env,agent = PPO.env_agent_config(cfg, paths, weights)
    agent.load(path=plot_cfg.model_path)
    rewards,ma_rewards = PPO.eval(cfg,env,agent)
    PPO.save_results(rewards, ma_rewards, tag='eval', path=plot_cfg.result_path)
    
    dist_ppo = copy.deepcopy(env.print_dist())
    
    es_onehot_ppo = copy.deepcopy(es_onehot)
    es_label_ppo = copy.deepcopy(es_label)
    for i in range(testset_onehot.shape[0]):
        es_label_ppo[i] = dist_ppo[test_list[i][0]][test_list[i][1]]
    
    # PPO+PAT1
    dist_ppo_pat = average_weighting(dist_pat, dist_ppo, MAX)
    
    es_onehot_ppo_pat = copy.deepcopy(es_onehot)
    es_label_ppo_pat = copy.deepcopy(es_label)
    for i in range(testset_onehot.shape[0]):
        es_label_ppo_pat[i] = dist_ppo_pat[test_list[i][0]][test_list[i][1]]
    
    # PPO+PAT2
    dist_ppo_pat2 = alpha_weighting2(dist_pat, dist_ppo, MAX, 3)
    
    es_onehot_ppo_pat2 = copy.deepcopy(es_onehot)
    es_label_ppo_pat2 = copy.deepcopy(es_label)
    for i in range(testset_onehot.shape[0]):
        es_label_ppo_pat2[i] = dist_ppo_pat2[test_list[i][0]][test_list[i][1]]
    
    # print('pat', sum_error(ground_truth, dist_pat))
    # print('ppo', sum_error(ground_truth, dist_ppo))
    # print('ppo+pat1', sum_error(ground_truth, dist_ppo_pat))
    # print('ppo+pat2', sum_error(ground_truth, dist_ppo_pat2))
    # print('change pat->ppo', sum_error(dist_pat, dist_ppo))
    
    MAPE_RL = [sum_error2(ground_truth, dist_pat, testset_onehot.shape[0]), 
               sum_error2(ground_truth, dist_ppo, testset_onehot.shape[0]),
               sum_error2(ground_truth, dist_ppo_pat, testset_onehot.shape[0]),
               sum_error2(ground_truth, dist_ppo_pat2, testset_onehot.shape[0])]
    
    # print('pat MAPE:', sum_error2(ground_truth, dist_pat, testset_onehot.shape[0]))
    # print('ppo MAPE:', sum_error2(ground_truth, dist_ppo, testset_onehot.shape[0]))
    # print('ppo+pat1 MAPE:', sum_error2(ground_truth, dist_ppo_pat, testset_onehot.shape[0]))
    # print('ppo+pat2 MAPE:', sum_error2(ground_truth, dist_ppo_pat2, testset_onehot.shape[0]))
    
    return es_onehot, es_label, es_onehot_ppo, es_label_ppo, \
        es_onehot_ppo_pat, es_label_ppo_pat, es_onehot_ppo_pat2, es_label_ppo_pat2, MAPE_RL
