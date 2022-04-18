import numpy as np
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from model import DGN
from buffer import ReplayBuffer
from surviving import Surviving
import time

from attackDefendEnv import *
from config import *
from config import epsilon, score

USE_CUDA = torch.cuda.is_available()

lamb = float(sys.argv[1])

g_env = GlobalAgentsEnv(RandomDefenfStrategy(),
                        SimpleAttackStrategy(
                            threat_angle=45,
                            threat_dis=0.5,  # 攻击策略
                            threat_angle_delta=10,
                            small_angle=10,
                            delta_t=0
                        ),
                        done_dis=1.4,
                        attack_num=4,
                        attack_radius=10,
                        defend_num=5,
                        defend_radius=2.5,
                        forbidden_radius=1.5,
                        threat_angle=45,
                        threat_angle_delta=10,
                        threat_dis=2,  # 奖励
                        capture_dis=0.001,
                        reward_agent_num=2,
                        render=False
                        )
env = DefendAgentsEnv(g_env)

n_ant = env.n_agent
observation_space = env.n_observation
n_actions = env.n_action

model_save_path = "model_path4_15_7_8_100000"
model = DGN(n_ant, observation_space, hidden_dim, n_actions)
model.load_state_dict(torch.load(model_save_path))
model = model.cuda()
while i_episode < 10:

    i_episode += 1
    score = 0    
    steps = 0
    obs, adj = env.reset()
    while steps < max_step:
        steps += 1
        action = []
        n_adj = adj + np.eye(n_ant)
        #a_ = np.array([obs])
        #b_ = np.array([n_adj])
        #a__ = torch.Tensor(np.array([obs]))
        #b__ = torch.Tensor(np.array([n_adj]))
        q, a_w = model(torch.Tensor(np.array([obs])).cuda(), torch.Tensor(np.array([n_adj])).cuda())
        q = q[0]
        for i in range(n_ant):
            a = q[i].argmax().item()
            action.append(a)

        next_obs, next_adj, reward, terminated = env.step(action)
        obs = next_obs
        score += sum(reward)
        adj = next_adj

    print(score)
#
# model.eval()
