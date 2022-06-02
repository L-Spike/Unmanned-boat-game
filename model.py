import math, random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available()
# Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args,
#                                                                                                                 **kwargs)


class Encoder(nn.Module):
    def __init__(self, din=32, hidden_dim=128):
        super(Encoder, self).__init__()
        self.fc = nn.Linear(din, hidden_dim)

    def forward(self, x):
        embedding = F.relu(self.fc(x))
        return embedding


class AttModel(nn.Module):
    def __init__(self, n_node, din, hidden_dim, dout):
        super(AttModel, self).__init__()
        self.fcv = nn.Linear(din, hidden_dim)
        self.fck = nn.Linear(din, hidden_dim)
        self.fcq = nn.Linear(din, hidden_dim)
        self.fcout = nn.Linear(hidden_dim, dout)

    def forward(self, x, mask):
        v = F.relu(self.fcv(x))
        q = F.relu(self.fcq(x))
        k = F.relu(self.fck(x)).permute(0, 2, 1)
        h = torch.clamp(torch.mul(torch.bmm(q, k), mask), 0, 9e13) - 9e15 * (1 - mask)
        att = F.softmax(h, dim=2)

        out = torch.bmm(att, v)
        # out = torch.add(out,v)
        # out = F.relu(self.fcout(out))
        return out, h


class Q_Net(nn.Module):
    def __init__(self, hidden_dim, dout):
        super(Q_Net, self).__init__()
        self.fc = nn.Linear(hidden_dim, dout)

    def forward(self, x):
        q = self.fc(x)
        return q


class DGN(nn.Module):
    def __init__(self, n_agent, num_inputs, hidden_dim, num_actions):
        super(DGN, self).__init__()

        self.encoder = Encoder(num_inputs, hidden_dim)
        self.att = AttModel(n_agent, hidden_dim, hidden_dim, hidden_dim)
        self.q_net = Q_Net(hidden_dim, num_actions)

    def forward(self, x, mask):
        h1 = self.encoder(x)
        h2, a_w = self.att(h1, mask)
        q = self.q_net(h2)
        return q, a_w


class DGNDouble(nn.Module):
    def __init__(self, n_agent, num_inputs, hidden_dim, num_actions):
        super(DGNDouble, self).__init__()

        self.encoder = Encoder(num_inputs, hidden_dim)
        self.att = AttModel(n_agent, hidden_dim, hidden_dim, hidden_dim)
        self.att2 = AttModel(n_agent, hidden_dim, hidden_dim, hidden_dim)
        self.q_net = Q_Net(hidden_dim, num_actions)

    def forward(self, x, mask):
        h1 = self.encoder(x)
        h2, a_w = self.att(h1, mask)
        h3, a_w2 = self.att(h2, mask)
        q = self.q_net(h3)
        return q, a_w


class DRQN(nn.Module):
    def __init__(self, input_shape, conf):
        super(DRQN, self).__init__()
        self.conf = conf
        self.fc1 = nn.Linear(input_shape, conf.drqn_hidden_dim)
        self.rnn = nn.GRUCell(conf.drqn_hidden_dim, conf.drqn_hidden_dim)
        self.fc2 = nn.Linear(conf.drqn_hidden_dim, conf.n_actions)

    def forward(self, obs, hidden_state):
        x = F.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, self.conf.drqn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h


class QMIXNET(nn.Module):
    def __init__(self, conf):
        super(QMIXNET, self).__init__()
        """
        生成的hyper_w1需要是一个矩阵，但是torch NN的输出只能是向量；
        因此先生成一个（行*列）的向量，再reshape
        """
        # print(conf.state_shape)
        self.conf = conf
        if self.conf.two_hyper_layers:
            self.hyper_w1 = nn.Sequential(nn.Linear(self.conf.state_shape, self.conf.hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(self.conf.hyper_hidden_dim,
                                                    self.conf.n_agents * self.conf.qmix_hidden_dim))
            self.hyper_w2 = nn.Sequential(nn.Linear(self.conf.state_shape, self.conf.hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(self.conf.hyper_hidden_dim, self.conf.qmix_hidden_dim))
        else:
            self.hyper_w1 = nn.Linear(self.conf.state_shape, self.conf.n_agents * self.conf.qmix_hidden_dim)
            self.hyper_w2 = nn.Linear(self.conf.state_shape, self.conf.qmix_hidden_dim * 1)

        self.hyper_b1 = nn.Linear(self.conf.state_shape, self.conf.qmix_hidden_dim)
        self.hyper_b2 = nn.Sequential(nn.Linear(self.conf.state_shape, self.conf.qmix_hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(self.conf.qmix_hidden_dim, 1))

    # input: (batch_size, n_agents, qmix_hidden_dim)
    # q_values: (episode_num, max_episode_len, n_agents)
    # states shape: (episode_num, max_episode_len, state_shape)
    def forward(self, q_values, states):
        # print(self.conf.state_shape)
        episode_num = q_values.size(0)
        q_values = q_values.view(-1, 1, self.conf.n_agents)
        states = states.reshape(-1, self.conf.state_shape)

        w1 = torch.abs(self.hyper_w1(states))
        b1 = self.hyper_b1(states)
        w1 = w1.view(-1, self.conf.n_agents, self.conf.qmix_hidden_dim)
        b1 = b1.view(-1, 1, self.conf.qmix_hidden_dim)

        hidden = F.elu(torch.bmm(q_values, w1) + b1)

        w2 = torch.abs(self.hyper_w2(states))
        b2 = self.hyper_b2(states)
        w2 = w2.view(-1, self.conf.qmix_hidden_dim, 1)
        b2 = b2.view(-1, 1, 1)

        q_total = torch.bmm(hidden, w2) + b2
        q_total = q_total.view(episode_num, -1, 1)

        return q_total
