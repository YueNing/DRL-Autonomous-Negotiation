from drl_negotiation.core.train._model import Runner


class QMixerModel(Runner):
    def __init__(self, env):
        self.env = env
        self.max_world_cnt = self.env.world.n_steps

    def reset(self):
        batch = self.new_batch()

        if self.learn_cnt > self.max_world_cnt:
            self.env.reset()

    def setup(self):
        self.new_batch = None

    def learn(self):
        self.env.reset()

        while True:
            self.reset()
            episode_batch = self.env.run()

import torch as th
import os
from drl_negotiation.core.networks.base_network import RNN
from drl_negotiation.core.networks.mixer.qmixer import QMixerNet

class QMixer:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        input_shape = self.obs_shape

        if args.last_action:
            input_shape += self.n_actions
        if args.reuse_network:
            input_shape += self.n_agents

        self.eval_rnn = RNN(input_shape, args) # network select action  of agents
        self.target_rnn = RNN(input_shape, args)

        self.eval_qmixer_net = QMixerNet(args) # network add qvalues of agents
        self.target_qmixer_net = QMixerNet(args)

        self.args = args

        self.model_dir = args.model_dir + '/' + args.alg + '/' + args.map

        if self.args.load_model:
            if os.path.exists(self.model_dir + '/rnn_net_params.pkl'):
                path_rnn = self.model_dir + '/rnn_net_params.pkl'
                path_qmix = self.model_dir + '/qmix_net_params.pkl'
                map_location = 'cuda:0' if self.args.cuda else 'cpu'
                self.eval_rnn.load_state_dict(th.load(path_rnn, map_location=map_location))
                self.eval_qmix_net.load_state_dict(th.load(path_qmix, map_location=map_location))
                print('Successfully load the model: {} and {}'.format(path_rnn, path_qmix))
            else:
                raise Exception("No model!")

        self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
        self.target_qmixer_net.load_state_dict(self.eval_qmixer_net.state_dict())

        self.eval_parameters = list(self.eval_qmixer_net.parameters()) + list(self.eval_rnn.parameters())
        if args.optimizer == "RMS":
            self.optimizer = th.optim.RMSprop(self.eval_parameters, lr=args.lr)

        self.eval_hidden = None
        self.target_hidden = None
        print("Init alg QMix")

    def learn(self, batch, max_episode_len, train_step, epsilon=None):
        episode_num = batch["o"].shape[0]
        self.init_hidden(episode_num)
        for key in batch.keys():
            if key == "u":
                batch[key] = th.tensor(batch[key], dtype=th.long)
            else:
                batch[key] = th.tensor(batch[key], dtype=th.float32)

        s, s_next, u, r, avail_u, avail_u_next, terminated = batch['s'], batch['s_next'], batch['u'], \
                                                             batch['r'], batch['avail_u'], batch['avail_u_next'], \
                                                             batch['terminated']
        mask = 1 - batch["padded"].float()

        q_evals, q_targets = self.get_q_values(batch, max_episode_len)

        q_evals = th.gather(q_evals, dim=3, index=u).squeeze(3)

        q_targets[avail_u_next == 0.0] = - 9999999
        q_targets = q_targets.max(dim=3)[0]

        q_total_eval = self.eval_qmixer_net(q_evals, s)
        q_total_target = self.target_qmixer_net(q_targets, s_next)

        targets = r + self.args.gamma * q_total_target * (1 - terminated)

        td_error = (q_total_eval - targets.detach())
        masked_td_error = mask * td_error

        if mask.sum == 0:
            loss = (masked_td_error ** 2).sum()
        else:
            loss = (masked_td_error ** 2).sum() / mask.sum()

        self.optimizer.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(self.eval_parameters, self.args.grad_norm_clip)
        self.optimizer.step()

        if train_step > 0 and train_step % self.args.target_update_cycle == 0:
            self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
            self.target_qmixer_net.load_state_dict(self.eval_qmixer_net.state_dict())


    def _get_inputs(self, batch, transition_idx):
        obs, obs_next, u_onehot = batch['o'][:, transition_idx], \
                                  batch['o_next'][:, transition_idx], batch['u_onehot'][:]
        episode_num = obs.shape[0]
        inputs, inputs_next = [], []
        inputs.append(obs)
        inputs_next.append(obs_next)

        if self.args.last_action:
            if transition_idx == 0:
                inputs.append(th.zeros_like(u_onehot[:, transition_idx]))
            else:
                inputs.append(u_onehot[:, transition_idx - 1])
            inputs_next.append(u_onehot[:, transition_idx])

        if self.args.reuse_network:
            inputs.append(th.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
            inputs_next.append(th.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
        inputs = th.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs], dim=1)
        inputs_next = th.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs_next], dim=1)
        return inputs, inputs_next

    def get_q_values(self, batch, max_episode_len):
        episode_num = batch["o"].shape[0]
        q_evals, q_targets = [], []
        for transition_idx in range(max_episode_len):
            inputs, inputs_next = self._get_inputs(batch, transition_idx)
            q_eval, self.eval_hidden = self.eval_rnn(inputs, self.eval_hidden)
            q_target, self.target_hidden = self.target_rnn(inputs_next, self.target_hidden)

            q_eval = q_eval.view(episode_num, self.n_agents, -1)
            q_target = q_target.view(episode_num, self.n_agents, -1)
            q_evals.append(q_eval)
            q_targets.append(q_target)

        q_evals = th.stack(q_evals, dim=1)
        q_targets = th.stack(q_targets, dim=1)
        return q_evals, q_targets

    def init_hidden(self, episode_num):
        self.eval_hidden = th.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
        self.target_hidden = th.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))

    def save_model(self, train_step):
        num = str(train_step // self.args.save_cycle)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        th.save(self.eval_qmix_net.state_dict(), self.model_dir + '/' + num + '_qmix_net_params.pkl')
        th.save(self.eval_rnn.state_dict(),  self.model_dir + '/' + num + '_rnn_net_params.pkl')