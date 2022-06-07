import sys
import os
import argparse
import torch
from model import DGNDouble
from attackDefendEnv import *
from config import *

USE_CUDA = torch.cuda.is_available()
description = 'run Defend model'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('path', type=str, help='the path of defend model')
parser.add_argument('--config', type=str, default="normal", help='the name of config')
parser.add_argument('--attack_model_path', '-amp', type=str, default=None, help='the path of attack model, None for rules')
args = parser.parse_args()
model_path = args.path

config_name = args.config
if config_name == "normal":
    from config import *
elif config_name == "1":
    from configs.config1 import *
elif config_name == "2":
    from configs.config2 import *
else:
    print(f'invalid config name:{config_name}!')
    exit(0)

attack_model_path = args.attack_model_path
g_env = GlobalAgentsEnv(RandomDefendStrategy(),
                        attack_strategy=SimpleAttackStrategy() if attack_model_path is None else DRLAttackStrategy(model_path=attack_model_path),
                        render=True
                        )
env = DefendAgentsEnv(g_env)

n_ant = env.n_agent
observation_space = env.n_observation
n_actions = env.n_action

if use_double_dgn:
    model = DGNDouble(n_ant, observation_space, hidden_dim, n_actions)
else:
    model = DGN(n_ant, observation_space, hidden_dim, n_actions)
model.load_state_dict(torch.load(model_path, map_location="cpu"))

while i_episode < run_n_episode:
    i_episode += 1
    steps = 0
    score = 0
    cur_result = 0
    env.reset()
    obs, adj = env.get_obs()
    if add_role:
        for i in range(len(obs)):
            if i % 3 == 0:
                obs[i].extend([0, 1])
            else:
                obs[i].extend([1, 0])
    while steps < max_step:
        steps += 1
        action = []
        n_adj = adj + np.eye(n_ant)
        q, a_w = model(torch.Tensor(np.array([obs])), torch.Tensor(np.array([n_adj])))
        q = q[0]
        for i in range(n_ant):
            a = q[i].argmax().item()
            action.append(a)

        reward, terminated, _ = env.step(action)
        next_obs, next_adj = env.get_obs()
        if add_role:
            for i in range(len(obs)):
                if i % 3 == 0:
                    obs[i].extend([0, 1])
                else:
                    obs[i].extend([1, 0])
        if add_role:
            for i in range(len(obs)):
                if i % 3 == 0:
                    obs[i].extend([0, 1])
                else:
                    obs[i].extend([1, 0])
        print(reward)
        # print(reward)
        if terminated:
            print("so_done")
            print(action)
            p.addUserDebugText(
                text="Fail!",
                textPosition=[0, 0, 3],
                textColorRGB=[255/255, 0, 0],  # 5；G:39；B:175
                textSize=3,
                lifeTime=1
            )
            cur_result = 1
            time.sleep(1)
            break
        # print(steps)
        obs = next_obs
        adj = next_adj
        score += sum(reward)
        time.sleep(time_to_render)

    if cur_result == 0:
        p.addUserDebugText(
            text="Succeed！",
            textPosition=[0, 0, 3],
            textColorRGB=[5 / 255, 39 / 255, 175 / 255],
            textSize=3,
            lifeTime=1
        )
        cur_result = 1
        time.sleep(1)

    print(f"i_episode: {i_episode}")
    print(f"score:{score}")
