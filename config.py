cuda_device = "1"

DEBUG = False  # draw debug item
action_setting = "speed"
actinIndex = "all"

use_forbidden_reward = False

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
reward_agent_num = 2
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

hidden_dim = 64
max_step = 100
GAMMA = 0.99
n_episode = 6000
run_n_episode = 10
i_episode = 0
time_to_render = 1 / 12
capacity = 100000
batch_size = 64
n_epoch = 25
epsilon = 0.9
epsilon_factor = 1
score = 0
lamb = 0.1
tau = 0.98
