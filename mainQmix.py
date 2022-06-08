import os
import numpy as np
from attackDefendEnv import *
from agent import Agents
from RLUtils import RolloutWorker, QmixReplayBuffer
import matplotlib.pyplot as plt
import gc
from config import Config
import pickle

conf = Config()


def train():
    g_env = GlobalAgentsEnv(RandomDefendStrategy(),
                            SimpleAttackStrategy(),
                            render=False
                            )
    env = DefendAgentsEnv(g_env)

    n_ant = env.n_agent
    observation_space = env.n_observation
    n_actions = env.n_action
    env_info = env.get_env_info()  # {'state_shape': 61, 'obs_shape': 42, 'n_actions': 10, 'n_agents': 3, 'episode_limit': 200}
    conf.set_env_info(env_info)
    agents = Agents(conf)
    rollout_worker = RolloutWorker(env, agents, conf)
    buffer = QmixReplayBuffer(conf)

    # save plt and pkl
    save_path = conf.result_dir
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    win_rates = []
    episode_rewards = []
    train_steps = 0

    data_dirs = "qmix_data"
    if not os.path.exists(data_dirs):
        os.makedirs(data_dirs)

    cumulative_rewards = []
    losses = []
    episode_steps = []
    evaluating_indicator = {"Cumulative reward": cumulative_rewards, "losses": losses,
                            "episode_steps": episode_steps}

    for epoch in range(conf.n_epochs):
        with torch.no_grad():
            episodes = []
            # print("a")
            for episode_idx in range(conf.n_eposodes):
                # print("b")
                episode, cumulative_reward, episode_step, wintag= rollout_worker.generate_episode(episode_idx)
                cumulative_rewards.append(cumulative_reward)

                episodes.append(episode)
                episode_steps.append(episode_step)
                # print("c")

            episode_batch = episodes[0]
            episodes.pop(0)
            for episode in episodes:
                for key in episode_batch.keys():
                    episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)

            buffer.store_episode(episode_batch)
        # print(episode_batch['o'].shape)  # (1, 200, 3, 42)
        # print(episode_batch['s'].shape)  # (1, 200, 61)
        # print(episode_batch['u'].shape)  # (1, 200, 3, 1)
        # print(episode_batch['r'].shape)  # (1, 200, 1)
        # print(episode_batch['o_'].shape) # (1, 200, 3, 42)
        # print(episode_batch['s_'].shape) # (1, 200, 61)
        # print(episode_batch['avail_u'].shape)  # (1, 200, 3, 10)
        # print(episode_batch['avail_u_'].shape) # (1, 200, 3, 10)
        # print(episode_batch['padded'].shape)   # (1, 200, 1)
        # print(episode_batch['terminated'].shape) # (1, 200, 1)
        # print("d")
        for train_step in range(conf.train_steps):
            # print("e")
            mini_batch = buffer.sample(min(buffer.current_size, conf.batch_size))  # obs； (64, 200, 3, 42)
            # print(mini_batch['o'].shape)
            # print("1:{}".format(torch.cuda.memory_allocated(0)))
            loss = agents.train(mini_batch, train_steps)
            losses.append(loss)
            print(f"i_episode:{epoch}\t\tloss:{loss}\t\tcumulative_reward:{cumulative_reward}\t\tepisode_step:{episode_step}")
            # print("2:{}".format(torch.cuda.memory_allocated(0)))
            train_steps += 1

        if epoch % conf.evaluate_per_epoch == 0:
            win_rate, episode_reward = evaluate(rollout_worker)
            win_rates.append(win_rate)
            episode_rewards.append(episode_reward)
            print("train epoch: {}, win rate: {}, episode reward: {}".format(epoch, win_rate, episode_reward))
            # show_curves(win_rates, episode_rewards)

    time_tuple = time.localtime(time.time())
    with open(os.path.join(data_dirs,
                           "d_{}v{}_{}_{}_{}_{}_{}_{}".format(n_episode, defend_num, attack_num, conf.use_soft_update,
                                                              time_tuple[1], time_tuple[2], time_tuple[3],
                                                              time_tuple[4]) + ".pkl"),
              "wb") as f:
        pickle.dump(evaluating_indicator, f, pickle.HIGHEST_PROTOCOL)

    show_curves(win_rates, episode_rewards)


def evaluate(rollout_worker):
    # print("="*15, " evaluating ", "="*15)
    win_num = 0
    episode_rewards = 0
    # print("3:{}".format(torch.cuda.memory_allocated(0)))
    with torch.no_grad():
        for epoch in range(conf.evaluate_epoch):
            _, episode_reward, episode_step, win_tag = rollout_worker.generate_episode(epoch, evaluate=True)
            episode_rewards += episode_reward
            if win_tag:
                win_num += 1
    # print("4:{}".format(torch.cuda.memory_allocated(0)))
    return win_num / conf.evaluate_epoch, episode_rewards / conf.evaluate_epoch


def show_curves(win_rates, episode_rewards):
    print("=" * 15, " generate curves ", "=" * 15)
    plt.figure()
    plt.axis([0, conf.n_epochs, 0, 100])
    plt.cla()
    plt.subplot(2, 1, 1)
    plt.plot(range(len(win_rates)), win_rates)
    plt.xlabel('epoch*{}'.format(conf.evaluate_per_epoch))
    plt.ylabel("win rate")

    plt.subplot(2, 1, 2)
    plt.plot(range(len(episode_rewards)), episode_rewards)
    plt.xlabel('epoch*{}'.format(conf.evaluate_per_epoch))
    plt.ylabel("episode reward")

    plt.savefig(conf.result_dir + '/result_plt.png', format='png')
    np.save(conf.result_dir + '/win_rates', win_rates)
    np.save(conf.result_dir + '/episode_rewards', episode_rewards)


if __name__ == "__main__":
    if conf.train:
        train()
