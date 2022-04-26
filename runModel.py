import sys
import os
import argparse
import torch
from model import DGN
from attackDefendEnv import *
from config import *

USE_CUDA = torch.cuda.is_available()
description = 'run model'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('path', type=str, help='the path of model')
parser.add_argument('--name', type=int, default=0, help='the name of config')
args = parser.parse_args()
model_path = args.path
# model_path = os.path.join('models', file_path)

config_name = args.name
if config_name == 0:
    from config import *
elif config_name == 1:
    from configs.config1 import *
elif config_name == 2:
    from configs.config2 import *
else:
    print(f'invalid config name:{config_name}!')
    exit(0)


g_env = GlobalAgentsEnv(RandomDefenfStrategy(),
                        SimpleAttackStrategy(
                            threat_angle=threat_angle,
                            threat_dis=attack_threat_dis,  # 攻击策略
                            threat_angle_delta=threat_angle_delta,
                            small_angle=small_angle
                        ),
                        not_find_reward=not_find_reward,
                        done_dis=done_dis,
                        attack_num=attack_num,
                        attack_radius=attack_radius,
                        defend_num=defend_num,
                        defend_radius=defend_radius,
                        forbidden_radius=forbidden_radius,
                        threat_angle=threat_angle,
                        threat_angle_delta=threat_angle_delta,
                        threat_dis=defend_threat_dis,  # 奖励
                        capture_dis=capture_dis,
                        reward_agent_num=reward_agent_num,
                        max_velocity=max_velocity,
                        max_turn_angle=max_turn_angle,
                        render=True
                        )
env = DefendAgentsEnv(g_env)

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
        print(reward)
        if terminated:
            print("so_done")
            print(action)
            break
        print(steps)
        # for ac in action:
        #     if 15 <= ac <=19:
        #         print("iop")
        #         print(action)
        #         break
        obs = next_obs
        adj = next_adj
        score += sum(reward)
        time.sleep(time_to_render)

    print(f"i_episode: {i_episode}")
    print(f"score:{score}")
