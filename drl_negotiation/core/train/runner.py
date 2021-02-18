import os
import numpy as np
from drl_negotiation.core.games._scml_oneshot import Agents
from drl_negotiation.core.components.episode_buffer import ReplayBuffer
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class EpisodeResult:
    episode: dict
    episode_reward: float
    step: int


from drl_negotiation.core.envs.multi_agents_negotiation import MultiNegotiationSCM


class RolloutWorker:
    def __init__(self, env, agents, args):
        self.env: MultiNegotiationSCM = env
        self.agents = agents
        self.episode_limit = args.episode_limit
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.args = args

        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon
        print("Init RolloutWorker")

    def generate_episode(self, episode_idx, evaluate=False):
        self.tmp_step = 0
        self.tmp_epsilon = 0 if evaluate else self.epsilon
        self.tmp_episode_reward = 0
        self.tmp_o, self.tmp_u, self.tmp_r, self.tmp_s, \
        self.tmp_avail_u, self.tmp_u_onehot, self.tmp_terminate, self.tmp_padded = [], [], [], [], [], [], [], []

        episode_result: EpisodeResult = self.env.run()

        obs = self.env.get_obs()
        state = self.env.get_state()
        self.tmp_o.append(obs)
        self.tmp_s.append(state)
        self.tmp_o_next = self.tmp_o[1:]
        self.tmp_s_next = self.tmp_s[1:]
        self.tmp_o = self.tmp_o[:-1]
        self.tmp_s = self.tmp_s[:-1]
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_action = self.env.get_avail_actions(agent_id)
            avail_actions.append(avail_action)
        self.tmp_avail_u.append(avail_actions)
        self.tmp_avail_u_next = self.tmp_avail_u[1:]
        self.tmp_avail_u = self.tmp_avail_u[:-1]

        # if step < self.episode_limit, padding
        for i in range(self.step, self.episode_limit):
            self.tmp_o.append(np.zeros((self.n_agents, self.obs_shape)))
            self.tmp_u.append(np.zeros([self.n_agents, 1]))
            self.tmp_s.append(np.zeros(self.state_shape))
            self.tmp_r.append([0.])
            self.tmp_o_next.append(np.zeros((self.n_agents, self.obs_shape)))
            self.tmp_s_next.append(np.zeros(self.state_shape))
            self.tmp_u_onehot.append(np.zeros((self.n_agents, self.n_actions)))
            self.tmp_avail_u.append(np.zeros((self.n_agents, self.n_actions)))
            self.tmp_avail_u_next.append(np.zeros((self.n_agents, self.n_actions)))
            self.tmp_padded.append([1.])
            self.tmp_terminate.append([1.])

        episode = dict(o=self.tmp_o.copy(),
                       s=self.tmp_s.copy(),
                       u=self.tmp_u.copy(),
                       r=self.tmp_r.copy(),
                       avail_u=self.tmp_avail_u.copy(),
                       o_next=self.tmp_o_next.copy(),
                       s_next=self.tmp_s_next.copy(),
                       avail_u_next=self.tmp_avail_u_next.copy(),
                       u_onehot=self.tmp_u_onehot.copy(),
                       padded=self.tmp_padded.copy(),
                       terminated=self.tmp_terminate.copy())

        for key in episode.keys():
            episode[key] = np.array([episode[key]])

        if not evaluate:
            self.epsilon = self.tmp_epsilon

        if evaluate and episode_idx == self.args.evaluate_epoch - 1 and self.args.replay_dir != '':
            self.env.save_replay()
            self.env.close()
        return episode, self.tmp_episode_reward, self.tmp_step


class Runner:
    def __init__(self, env, args):
        self.env = env
        self.env._rl_runner = self

        self.agents = Agents(args)
        self.rollout_worker = RolloutWorker(env, self.agents, args)
        self.buffer = ReplayBuffer(args)

        self.args = args
        self.episode_rewards = []

        self.save_path = self.args.result_dir + '/' + args.alg + '/' + args.map
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def run(self, num):
        time_steps, train_steps, evaluate_steps = 0, 0, -1
        while time_steps < self.args.n_steps:
            print(f"Run {num}, time_steps {time_steps}")
            if time_steps // self.args.evaluate_cycle > evaluate_steps:
                episode_reward = self.evaluate()
                self.episode_rewards.append(episode_reward)
                self.plt(num)
                evaluate_steps += 1
            episodes = []

            for episode_idx in range(self.args.n_episodes):
                episode, _, steps = self.rollout_worker.generate_episode(episode_idx)
                episodes.append(episode)
                time_steps += steps

            episode_batch = episodes[0]
            episodes.pop(0)
            for episode in episodes:
                for key in episode_batch.keys():
                    episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)

            self.buffer.store_episode(episode_batch)
            for train_step in range(self.args.train_steps):
                mini_batch = self.buffer.sample(min(self.buffer.current_size, self.args.batch_size))
                self.agents.train(mini_batch, train_steps)
                train_steps += 1
        episode_reward = self.evaluate()
        self.episode_rewards.append(episode_reward)
        self.plt(num)

    def evaluate(self):
        episode_rewards = 0
        for epoch in range(self.args.evaluate_epoch):
            _, episode_reward, _ = self.rollout_worker.generate_episode(epoch, evaluate=True)
            episode_rewards += episode_reward
        return episode_rewards / self.args.evaluate_epoch

    def plt(self, num):
        plt.figure()
        plt.ylim([0, 105])
        plt.cla()

        plt.subplot(1, 1, 1)
        plt.plot(range(len(self.episode_rewards)), self.episode_rewards)
        plt.xlabel('step*{}'.format(self.args.evaluate_cycle))
        plt.ylabel('episode_rewards')

        plt.savefig(self.save_path + '/plt_{}.png'.format(num), format='png')
        np.save(self.save_path + '/episode_rewards_{}'.format(num), self.episode_rewards)
        plt.close()
