import torch as th
import torch.nn as nn
import torch.nn.functional as f
import numpy as np


class QMixerNet(nn.Module):
    def __init__(self, args):
        super(QMixerNet, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))

        # hidden layer
        self.embed_dim = args.mixing_embed_dim

        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        elif getattr(args, "hypernet_layers", 1) == 2:
            hypernet_embed = self.args.hypernet_embed
            self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
            self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                               nn.ReLU(),
                                               nn.Linear(hypernet_embed, self.embed_dim))
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise NotImplementedError
        else:
            raise ValueError("Error setting layers")

        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        self.hyper_b_final = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, q_values, states):
        """Returns global q value based on local q values
        q_values: qValues of agents, shape is (episode_num, max_episode_len, n_agents)
        states: shape of states is (episode_num, max_episode_len, state_shape)
        """
        episode_num = q_values.size(0)
        q_values = q_values.view(-1, 1, self.args.n_agents)
        states = states.reshape(-1, self.args.state_shape)

        w1 = th.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)

        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)

        hidden = f.elu(th.bmm(q_values, w1) + b1)

        w2 = th.abs(self.hyper_w_final(states))
        b2 = self.hyper_b_final(states)

        w2 = w2.view(-1, self.embed_dim, 1)
        b2 = b2.view(-1, 1, 1)

        q_tot = th.bmm(hidden, w2) + b2
        q_tot = q_tot.view(episode_num, -1, 1)
        return q_tot
