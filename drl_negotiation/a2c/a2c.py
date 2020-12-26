"""
A2C model
"""
import os
import time
import argparse
from drl_negotiation.a2c.policy import Policy
from drl_negotiation.env import SCMLEnv
import drl_negotiation.utils as U
import numpy as np
import pickle
from tqdm import tqdm

class MADDPGModel:
    def __init__(self,
                 env: SCMLEnv = None,
                 policy=None,

                 # training
                 n_steps=2,
                 lr=1e-2,
                 gamma=0.95,
                 save_dir="/tmp/policy/",
                 save_rate=5,
                 load_dir='',
                 exp_name="",
                 batch_size=1,
                 num_units=64,
                 # env
                 n_envs=1,
                 num_episodes=60,
                 max_episode_len=10,
                 num_adversaries=0,
                 good_policy="maddpg",
                 adv_policy="heuristic",

                 # evaluation
                 benchmark=False,
                 benchmark_iters=1000,
                 benchmark_dir="./benchmark_files/",
                 restore=False,
                 display=False,
                 plots_dir="./learning_curves/",

                 _init_setup_model=True,
                 **kwargs,
        ):
        self.policy = policy
        self.env = env

        # training
        self.n_steps = n_steps
        self.lr = lr
        self.gamma = gamma
        self.save_dir = save_dir
        self.save_rate = save_rate
        self.load_dir = load_dir
        self.exp_name = exp_name
        self.batch_size = batch_size
        self.num_units = num_units

        # env
        self.n_envs = n_envs
        self.num_episodes = num_episodes
        self.max_eipsode_len = max_episode_len
        self.num_adversaries = num_adversaries
        self.good_policy = good_policy
        self.adv_policy = adv_policy

        # evaluation
        self.benchmark = benchmark
        self.benchmark_iters = benchmark_iters
        self.benchmark_dir = benchmark_dir
        self.restore = restore
        self.display = display
        self.plots_dir = plots_dir

        self.arglist = kwargs

        self.trainers = None


        if _init_setup_model:
            self.setup_model()

    def setup_model(self):
        return
        assert issubclass(self.policy, Policy), "Error: the input policy for the maddpg model must be an" \
                                                "instance of a2c.policy.Policy"

        # self.graph = tf.Graph()

    def _setup_learn(self):
        """
        Check the environment
        """
        if self.env is None:
            raise ValueError("Error: cannot train the model without a valid environment, please set an environment "
                             "with set_env(self, env) method.")

    def learn(self, train_episodes=None):
        self._setup_learn()
        if train_episodes is not None:
            self.num_episodes = train_episodes

        with U.single_threaded_session():
            obs_shape_n = [self.env.observation_space[i].shape for i in range(self.env.n)]
            num_adversaries = min(self.env.n, self.num_adversaries)
            arglist = argparse.Namespace(**{"good_policy":self.good_policy,
                       "adv_policy": self.adv_policy,
                       "lr": self.lr,
                       "num_units": self.num_units,
                       "batch_size": self.batch_size,
                       "max_episode_len": self.max_eipsode_len,
                       "gamma": self.gamma,
                        "n_steps": self.n_steps
                       })
            self.trainers = U.get_trainers(self.env, num_adversaries, obs_shape_n, arglist)
            print(f"Using good policy {self.good_policy} and adv policy {self.adv_policy}")

            U.initialize()
            if self.load_dir == '':
                self.load_dir = self.save_dir

            if self.display or self.restore or self.benchmark:
                print("Loading previous state...")
                U.load_state(self.load_dir)

            episode_rewards = [0.0]
            agent_rewards = [[0.0] for _ in range(self.env.n)]

            final_ep_rewards = []
            final_ep_ag_rewards = []
            agent_info = [[[]]]
            saver = U.get_saver()
            obs_n = self.env.reset()

            episode_step = 0
            current_episode = 0
            train_step = 0
            t_start = time.time()
            pbar = tqdm(total=self.num_episodes)

            while True:
                #print(f'episodes: {len(episode_rewards)}, train steps: {train_step}')
                action_n = self.predict(obs_n)
                new_obs_n, rew_n, done_n, info_n = self.env.step(action_n)

                episode_step +=1
                done = all(done_n)
                terminal = (episode_step > self.max_eipsode_len)

                # experience
                for i, agent in enumerate(self.trainers):
                    agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)

                obs_n = new_obs_n

                for i, rew in enumerate(rew_n):
                    episode_rewards[-1] +=rew
                    agent_rewards[i][-1] +=rew

                if done or terminal:
                    obs_n = self.env.reset()
                    episode_step = 0
                    pbar.update(1)
                    episode_rewards.append(0)
                    for a in agent_rewards:
                        a.append(0)
                    agent_info.append([[]])

                train_step += 1

                # Evaluate, benchmarking learned policies
                if self.benchmark:
                    for i, info in enumerate(info_n):
                        agent_info[-1][i].append(info_n['n'])

                    if train_step > self.benchmark_iters and (done or terminal):
                        file_name = self.benchmark_dir + self.exp_name + '.pkl'
                        print("Finished benchmarking, now saving....")
                        with open(file_name, 'wb') as fp:
                            pickle.dump(agent_info[:-1], fp)
                        break
                    continue

                # for displaying learned policies, not learning
                if self.display:
                    time.sleep(0.1)
                    self.env.render()
                    continue

                # learn, update all policies in trainers, if not in display or benchmark mode
                loss = None
                for agent in self.trainers:
                    agent.preupdate()
                for agent in self.trainers:
                    loss = agent.update(self.trainers, train_step)
                    if loss is not None:
                        pass
                        #print(f"{agent}'s loss is {loss}")

                ##############################################################################
                # save model
                # display training output
                ##############################################################################
                if terminal and (len(episode_rewards) % self.save_rate == 0):
                    U.save_state(self.save_dir + "model", saver=saver)
                    if num_adversaries == 0:
                        print(f"steps: {train_step}, episodes: {len(episode_rewards)}, "
                              f"mean episode reward: {np.mean(episode_rewards[-self.save_rate:])}, "
                              f"time: {round(time.time() - t_start, 3)}")
                    t_start = time.time()
                    final_ep_rewards.append(np.mean(episode_rewards[-self.save_rate:]))
                    for rew in agent_rewards:
                        final_ep_ag_rewards.append(np.mean(rew[-self.save_rate:]))

                ##############################################################################
                # saves final episode reward for plotting training curve
                ##############################################################################
                if len(episode_rewards) > self.num_episodes:
                    module_path = os.getcwd()
                    rew_file_name = self.plots_dir + self.exp_name + "_rewards.pkl"
                    rew_file_name = os.path.join(module_path, rew_file_name)
                    with open(rew_file_name, 'wb') as fp:
                        pickle.dump(final_ep_rewards, fp)
                    agrew_file_name = self.plots_dir + self.exp_name + '_agrewards.pkl'
                    with open(agrew_file_name, 'wb') as fp:
                        pickle.dump(final_ep_ag_rewards, fp)
                    print(f'...Finished total of {len(episode_rewards)} episodes')
                    break

    def predict(self, obs_n):
        action_n = [agent.action(obs) for agent, obs in zip(self.trainers, obs_n)]
        return action_n