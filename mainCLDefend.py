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
from model import DGNDouble

# from config import epsilon

USE_CUDA = torch.cuda.is_available()
description = 'train defend model main function'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('path', type=str, help='the path of CL model')
parser.add_argument('--config', type=str, default="normal", help='the name of config file')
args = parser.parse_args()
config_name = args.config

if config_name == 'normal':
    from config import *
elif config_name == '1':
    from configs.config1 import *
elif config_name == '2':
    from configs.config2 import *
else:
    print(f'invalid config name:{config_name}!')
    exit(0)

USE_CUDA = torch.cuda.is_available()
os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device

g_env = GlobalAgentsEnv(RandomDefendStrategy(),
                        SimpleAttackStrategy(),
                        render=False
                        )
env = DefendAgentsEnv(g_env)

n_ant = env.n_agent
observation_space = env.n_observation+2 if add_role else env.n_observation
n_actions = env.n_action
buff = ReplayBuffer(capacity, observation_space, n_actions, n_ant)
model_path = args.path
if use_double_dgn:
    model = DGNDouble(n_ant, observation_space, hidden_dim, n_actions)
    model_tar = DGNDouble(n_ant, observation_space, hidden_dim, n_actions)
else:
    model = DGN(n_ant, observation_space, hidden_dim, n_actions)
    model_tar = DGN(n_ant, observation_space, hidden_dim, n_actions)


model.load_state_dict(torch.load(model_path))
model = model.cuda()
model_tar.load_state_dict(torch.load(model_path))
model_tar = model_tar.cuda()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

M_Null = torch.Tensor(np.array([np.eye(n_ant)] * batch_size)).cuda()
KL = nn.KLDivLoss()

model_dirs = "defend_models"
if not os.path.exists(model_dirs):
    os.makedirs(model_dirs)

data_dirs = "defend_train_data"
if not os.path.exists(data_dirs):
    os.makedirs(data_dirs)

# data = {"mean": means, "std": stds}
cumulative_rewards = []
losses = []
loss1s = []
loss2s = []
episode_steps = []
evaluating_indicator = {"Cumulative reward": cumulative_rewards, "losses": [loss1s, loss2s],
                        "episode_steps": episode_steps}

while i_episode < n_episode:

    # 900个 0.01
    if i_episode > 40:
        epsilon -= 0.001 * epsilon_factor
        if epsilon < 0.02:
            epsilon = 0.02
    i_episode += 1
    steps = 0
    env.reset()
    obs, adj = env.get_obs()
    if add_role:
        for i in range(len(obs)):
            if i % 3 == 0:
                obs[i].extend([0, 1])
            else:
                obs[i].extend([1, 0])
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

        reward, terminated, _ = env.step(action)
        next_obs, next_adj = env.get_obs()
        if add_role:
            for i in range(len(obs)):
                if i % 3 == 0:
                    next_obs[i].extend([0, 1])
                else:
                    next_obs[i].extend([1, 0])
        buff.add(np.array(obs), action, reward, np.array(next_obs), n_adj, next_adj, terminated)
        score += sum(reward)
        if terminated:
            break
        obs = next_obs
        adj = next_adj

    cumulative_rewards.append(score)
    episode_steps.append(steps)

    if i_episode < 40:
        print(f"i_episode: {i_episode}\t\tscore:{score}")
        continue
    loss_rollouts = []
    loss1_rollouts = []
    loss2_rollouts = []
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

        # loss = (q_values - torch.Tensor(expected_q).cuda()).pow(2).mean() + lamb * loss_kl/;
        loss1 = (q_values - torch.Tensor(expected_q).cuda()).pow(2).mean()
        loss2 = lamb * loss_kl
        loss = loss1 + loss2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            for p, p_targ in zip(model.parameters(), model_tar.parameters()):
                p_targ.data.mul_(tau)
                p_targ.data.add_((1 - tau) * p.data)
        loss_value = loss.item()
        loss1_value = loss1.item()
        loss2_value = loss2.item()
        losses.append(np.mean(loss_rollouts))
        logging.debug(f"loss1: {np.mean(loss1_rollouts)}")
        # logging.debug(f"loss2: {np.mean(loss2_rollouts)}")
        loss_rollouts.append(loss_value)
        loss1_rollouts.append(loss1_value)
        loss2_rollouts.append(loss2_value)

    losses.append(np.mean(loss_rollouts))
    print(f"i_episode: {i_episode}\t\tloss:{np.mean(loss1_rollouts)}\t\tscore:{score}")
    loss1s.append(np.mean(loss1_rollouts))
    loss2s.append(np.mean(loss2_rollouts))

    if i_episode % 1000 == 0:
        # 存储网络参数， 完成预测defend
        time_tuple = time.localtime(time.time())
        model_save_path = os.path.join(model_dirs,
                                       "{}v{}_{}_{}_{}_{}_{}".format(defend_num, attack_num, n_episode, epsilon_factor, time_tuple[1],
                                                                       time_tuple[2], i_episode))
        torch.save(model.state_dict(), model_save_path)

time_tuple = time.localtime(time.time())
with open(os.path.join(data_dirs,
                       "d_{}v{}_{}_{}_{}_{}".format(defend_num, attack_num, n_episode, epsilon_factor, time_tuple[1], time_tuple[2]) + ".pkl"),
          "wb") as f:
    pickle.dump(evaluating_indicator, f, pickle.HIGHEST_PROTOCOL)
