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
                        render=True
                        )
env = DefendAgentsEnv(g_env)

n_ant = env.n_agent
observation_space = env.n_observation
n_actions = env.n_action

model_save_path = "model_path4_15_7_8_100000"
model = DGN(n_ant, observation_space, hidden_dim, n_actions)
model.load_state_dict(torch.load(model_save_path, map_location="cpu"))

while i_episode < run_n_episode:
    i_episode += 1
    steps = 0
    score = 0
    obs, adj = env.reset()
    while steps < max_step:
        steps += 1
        action = []
        n_adj = adj + np.eye(n_ant)
        q, a_w = model(torch.Tensor(np.array([obs])), torch.Tensor(np.array([n_adj])))
        #q, a_w = model(torch.Tensor(np.array([obs])).cuda(), torch.Tensor(np.array([n_adj])).cuda())
        # todo
        q = q[0]
        for i in range(n_ant):
            a = q[i].argmax().item()
            action.append(a)



        next_obs, next_adj, reward, terminated = env.step(action)
        # if terminated:
        #     print("so_done")
        #     print(action)
            # break
        for ac in action:
            if 15 <= ac <=19:
                print("iop")
                print(action)
                break
        obs = next_obs
        adj = next_adj
        score += sum(reward)
        time.sleep(time_to_render)

    print(f"i_episode: {i_episode}")
    print(f"score:{score}")
