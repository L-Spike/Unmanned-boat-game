import sys
import os
import argparse
import torch
from model import DGN
from attackDefendEnv import *
from config import *

USE_CUDA = torch.cuda.is_available()
description = 'run attack model'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('path', type=str, help='the path of model')
parser.add_argument('--config', type=str, default='normal', help='the name of config')
args = parser.parse_args()
model_path = args.path

config_name = args.config
if config_name == 'normal':
    from config import *
elif config_name == 'dis':
    from configs.config_dis import *
elif config_name == 'dis_done':
    from configs.config_dis_done import *
else:
    print(f'invalid config name:{config_name}!')
    exit(0)

g_env = GlobalAgentsEnv(RandomDefendStrategy(),
                        SimpleAttackStrategy(),
                        render=True
                        )
env = AttackAgentsEnv(g_env)

n_ant = env.n_agent
observation_space = env.n_observation
n_actions = env.n_action

model = DGN(n_ant, observation_space, hidden_dim, n_actions)
model.load_state_dict(torch.load(model_path, map_location="cpu"))

while i_episode < run_n_episode:
    i_episode += 1
    steps = 0
    score = 0
    obs, adj = env.reset()
    cur_result = 0
    while steps < max_step:
        steps += 1
        action = []
        n_adj = adj + np.eye(n_ant)
        q, a_w = model(torch.Tensor(np.array([obs])), torch.Tensor(np.array([n_adj])))
        q = q[0]
        for i in range(n_ant):
            a = q[i].argmax().item()
            action.append(a)

        next_obs, next_adj, reward, terminated = env.step(action)
        # print(reward)
        if terminated:
            p.addUserDebugText(
                text="Succeed!",
                textPosition=[0, 0, 3],
                textColorRGB=[5/255, 39/255, 175/255],#5；G:39；B:175
                textSize=3,
                lifeTime=1
            )
            cur_result = 1
            time.sleep(1)
            break
        # print(steps)
        # for ac in action:
        #     if 15 <= ac <=19:
        #         print("iop")
        #         print(action)
        #         break
        obs = next_obs
        adj = next_adj
        score += sum(reward)
        time.sleep(time_to_render)
    if cur_result == 0:
        p.addUserDebugText(
            text="Fail！",
            textPosition=[0, 0, 3],
            textColorRGB=[255/255, 0, 0],
            textSize=3,
            lifeTime=1
        )
        cur_result = 1
        time.sleep(1)
    print(f"i_episode: {i_episode}")
    print(f"score:{score}")
