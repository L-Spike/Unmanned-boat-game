cuda_device = "1"
use_double_dgn = True
use_global_reward = True

DEBUG = False  # draw debug item
action_setting = "speed"
actinIndex = "all"

use_done = True
use_angle = True  # only for attack

defend_reward_factor = 4
turn_radius = 2
attack_reward_factor = 1
not_find_reward = -0.1
forbidden_reward = -10
defend_succeed_reward = 50
capture_reward = 1.5
not_to_catch_reward = -1
# punishment =
attack_num = 8
attack_radius = 15
defend_num = 12
done_dis = 1
forbidden_radius = 2
defend_radius = 4
threat_angle = 45
threat_angle_delta = 10
small_angle = 10
attack_threat_dis = 1.5
defend_threat_dis = 4
capture_dis = 1.5
reward_agent_num = 3
max_velocity = 5
max_turn_angle = 5
communicate_radius = 100
observe_radius = 100
ignore_radius = 10
target_position = [0, 0]

# attack reward
# attack_forbidden_reward = 10
attack_succeed_reward = 50
too_far_reward = -2
defend_ok_reward = 1

hidden_dim = 512
max_step = 200
GAMMA = 0.99
n_episode = 15000
run_n_episode = 10
i_episode = 0
time_to_render = 1 / 12
capacity = 100000
batch_size = 64
n_epoch = 25
epsilon = 0.9
epsilon_factor = 0.25
score = 0
lamb = 0.1
tau = 0.98


import torch


class Config:
    def __init__(self):
        self.train = True
        self.seed = 133
        self.cuda = True

        # train setting
        self.last_action = True  # 使用最新动作选择动作
        self.reuse_network = True  # 对所有智能体使用同一个网络
        self.n_epochs = 100000  # 20000
        self.evaluate_epoch = 20  # 20
        self.evaluate_per_epoch = 100  # 100
        self.batch_size = 32  # 32
        self.buffer_size = int(1e2)
        self.save_frequency = 5000  # 5000
        self.n_eposodes = 1  # 每个epoch有多少episodes
        self.train_steps = 1  # 每个epoch有多少train steps
        self.gamma = 0.99
        self.grad_norm_clip = 10  # prevent gradient explosion
        self.update_target_params = 200  # 200
        self.result_dir = './results/'

        # test setting
        self.load_model = False

        # SC2 env setting
        self.replay_dir = './replay_buffer/'

        if self.cuda:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        # model structure
        # drqn net
        self.drqn_hidden_dim = 64
        # qmix net
        # input: (batch_size, n_agents, qmix_hidden_dim)
        self.qmix_hidden_dim = 32
        self.two_hyper_layers = False
        self.hyper_hidden_dim = 64
        self.model_dir = './models/'
        self.optimizer = "RMS"
        self.learning_rate = 5e-4

        # epsilon greedy
        self.start_epsilon = 1
        self.end_epsilon = 0.05
        self.anneal_steps = 50000  # 50000
        self.anneal_epsilon = (self.start_epsilon - self.end_epsilon) / self.anneal_steps
        self.epsilon_anneal_scale = 'step'

    def set_env_info(self, env_info):
        self.n_actions = env_info["n_actions"]
        self.state_shape = env_info["state_shape"]
        self.obs_shape = env_info["obs_shape"]
        self.n_agents = env_info["n_agents"]
        self.episode_limit = env_info["episode_limit"]

