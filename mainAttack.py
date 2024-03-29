import numpy as np
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from model import DGN
from RLUtils import ReplayBuffer
import time
import pickle
import os
import argparse

from attackDefendEnv import *
from config import *
from config import epsilon

USE_CUDA = torch.cuda.is_available()
description = 'train attack model main function'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('--config', type=str, default='normal', help='the name of config')
args = parser.parse_args()

config_name = args.name
if config_name == 'normal':
    from config import *
elif config_name == 'dis':
    from configs.config_dis import *
elif config_name == 'dis_done':
    from configs.config_dis_done import *
else:
    print(f'invalid config name:{config_name}!')
    exit(0)

USE_CUDA = torch.cuda.is_available()
os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device

g_env = GlobalAgentsEnv(RandomDefendStrategy(),
                        SimpleAttackStrategy(),
                        render=False
                        )
env = AttackAgentsEnv(g_env)

n_ant = env.n_agent
observation_space = env.n_observation
n_actions = env.n_action
buff = ReplayBuffer(capacity, observation_space, n_actions, n_ant)
model = DGN(n_ant, observation_space, hidden_dim, n_actions)
model_tar = DGN(n_ant, observation_space, hidden_dim, n_actions)
model = model.cuda()
model_tar = model_tar.cuda()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

M_Null = torch.Tensor(np.array([np.eye(n_ant)] * batch_size)).cuda()
KL = nn.KLDivLoss()

# data = {"mean": means, "std": stds}
cumulative_rewards = []
losses = []
evaluating_indicator = {"Cumulative reward": cumulative_rewards, "losses": losses}

model_dirs = "attack_models"
if not os.path.exists(model_dirs):
    os.makedirs(model_dirs)

data_dirs = "attack_train_data"
if not os.path.exists(data_dirs):
    os.makedirs(data_dirs)

while i_episode < n_episode:

    if i_episode > 40:
        epsilon -= 0.001
        if epsilon < 0.01:
            epsilon = 0.01
    i_episode += 1
    steps = 0
    obs, adj = env.reset()
    score = 0
    while steps < max_step:
        steps += 1
        action = []
        n_adj = adj + np.eye(n_ant)
        a_ = np.array([obs])
        b_ = np.array([n_adj])
        a__ = torch.Tensor(np.array([obs]))
        b__ = torch.Tensor(np.array([n_adj]))
        q, a_w = model(torch.Tensor(np.array([obs])).cuda(), torch.Tensor(np.array([n_adj])).cuda())
        q = q[0]
        for i in range(n_ant):
            if np.random.rand() < epsilon:
                a = np.random.randint(n_actions)
            else:
                a = q[i].argmax().item()
            action.append(a)

        next_obs, next_adj, reward, terminated = env.step(action)
        buff.add(np.array(obs), action, reward, np.array(next_obs), n_adj, next_adj, terminated)
        score += sum(reward)
        if terminated:
            break
        obs = next_obs
        adj = next_adj

    cumulative_rewards.append(score)

    print(f"i_episode: {i_episode}")
    print(f"score:{score}")

    if i_episode < 40:
        continue
    loss_rollouts = []
    for e in range(n_epoch):

        O, A, R, Next_O, Matrix, Next_Matrix, D = buff.getBatch(batch_size)
        O = torch.Tensor(O).cuda()
        Matrix = torch.Tensor(Matrix).cuda()
        Next_O = torch.Tensor(Next_O).cuda()
        Next_Matrix = torch.Tensor(Next_Matrix).cuda()
        Next_Matrix = Next_Matrix + M_Null

        q_values, attention = model(O, Matrix)
        target_q_values, target_attention = model_tar(Next_O, Next_Matrix)
        target_q_values = target_q_values.max(dim=2)[0]
        target_q_values = np.array(target_q_values.cpu().data)
        expected_q = np.array(q_values.cpu().data)

        for j in range(batch_size):
            for i in range(n_ant):
                expected_q[j][i][A[j][i]] = R[j][i] + (1 - D[j]) * GAMMA * target_q_values[j][i]

        attention = F.log_softmax(attention, dim=2)
        target_attention = F.softmax(target_attention, dim=2)
        target_attention = target_attention.detach()
        loss_kl = F.kl_div(attention, target_attention, reduction='mean')

        loss = (q_values - torch.Tensor(expected_q).cuda()).pow(2).mean() + lamb * loss_kl
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            for p, p_targ in zip(model.parameters(), model_tar.parameters()):
                p_targ.data.mul_(tau)
                p_targ.data.add_((1 - tau) * p.data)
        loss_value = loss.item()
        loss_rollouts.append(loss_value)

    losses.append(np.mean(loss_rollouts))

    if i_episode % 1000 == 0:
        # 存储网络参数， 完成预测
        time_tuple = time.localtime(time.time())

        model_save_path = os.path.join(model_dirs,
                                       "model_{}_{}_{}_{}_{}_{}".format(config_name, time_tuple[1], time_tuple[2],
                                                                        time_tuple[3],
                                                                        time_tuple[4], i_episode))
        torch.save(model.state_dict(), model_save_path)

time_tuple = time.localtime(time.time())
with open(os.path.join(data_dirs,
                       "evaluating_indicator_{}_{}_{}_{}_{}".format(config_name, time_tuple[1], time_tuple[2],
                                                                    time_tuple[3],
                                                                    time_tuple[4]) + ".pkl"), "wb") as f:
    pickle.dump(evaluating_indicator, f, pickle.HIGHEST_PROTOCOL)
