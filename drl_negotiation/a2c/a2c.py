"""
A2C model
"""
import os
import time
import argparse
import tensorflow as tf
import gym
from drl_negotiation.env import SCMLEnv
import drl_negotiation.utils as U
import numpy as np
import pickle
from tqdm import tqdm
from drl_negotiation.hyperparameters import *
import logging

class MADDPGModel:
    trained_model = None

    def __init__(self,
                 env: SCMLEnv = None,
                 policy=None,
                 only_seller=ONLY_SELLER,
                 logging_level = LOGGING_LEVEL,

                 # training
                 # trainer update steps
                 n_steps=2,
                 # learning rate
                 lr=1e-2,
                 # discount factor
                 gamma=0.95,
                 # model save dir
                 save_dir=SAVE_DIR,
                 # model name
                 model_name=MODEL_NAME,
                 # model save rate
                 save_rate=SAVE_RATE,
                 # model load dir
                 load_dir='',
                 # experiment name
                 exp_name="",
                 # batch size * max_episode_len = replay buffer
                 batch_size=1,
                 num_units=64,
                 # env
                 n_envs=1,
                 # number of training episodes
                 num_episodes=60,
                 # max length of every episode
                 max_episode_len=10,
                 # number of adversaries
                 num_adversaries=0,
                 # policy of good agent
                 good_policy="maddpg",
                 # policy of adversary agent
                 adv_policy="heuristic",

                 # evaluation
                 benchmark=False,
                 benchmark_iters=1000,
                 benchmark_dir="./benchmark_files/",
                 restore=False,
                 display=False,
                 plots_dir="./learning_curves/",
                 # init the model, used for evaluation
                 _init_setup_model=False,
                 save_trainers=SAVE_TRAINERS,
                 **kwargs,
        ):
        self.policy = policy
        self.env = env
        self.only_seller = only_seller
        self.logging_level = logging_level

        # training
        self.n_steps = n_steps
        self.lr = lr
        self.gamma = gamma
        self.save_dir = save_dir
        self.save_rate = save_rate
        self.model_name = model_name
        self.load_dir = load_dir
        self.exp_name = exp_name
        self.batch_size = batch_size
        self.num_units = num_units

        # env
        self.n_envs = n_envs
        self.num_episodes = num_episodes
        self.max_episode_len = max_episode_len
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

        U.logging_setup()

        if _init_setup_model:
            self.setup_model()

        self.save_trainers = save_trainers

    def setup_model(self):
        with U.single_threaded_session():
            if not ONLY_SELLER:
                obs_shape_n = []
                for i in range(self.env.n):
                    obs_shape_n.append(self.env.observation_space[i].shape)
                    obs_shape_n.append(self.env.observation_space[i+1].shape)
            else:
                obs_shape_n = [self.env.observation_space[i].shape for i in range(self.env.n)]

            num_adversaries = min(self.env.n, self.num_adversaries)
            arglist = argparse.Namespace(**{"good_policy": self.good_policy,
                                            "adv_policy": self.adv_policy,
                                            "lr": self.lr,
                                            "num_units": self.num_units,
                                            "batch_size": self.batch_size,
                                            "max_episode_len": self.max_episode_len,
                                            "gamma": self.gamma,
                                            "n_steps": self.n_steps
                                            })
            self.trainers = U.get_trainers(self.env, num_adversaries, obs_shape_n, arglist)
            logging.info(f"Using good policy {self.good_policy} and adv policy {self.adv_policy}")

        # assert issubclass(self.policy, Policy), "Error: the input policy for the maddpg model must be an" \
        #                                         "instance of a2c.policy.Policy"

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
            if not ONLY_SELLER:
                obs_shape_n = []
                for i in range(self.env.n):
                    obs_shape_n.append(self.env.observation_space[i].shape)
                    obs_shape_n.append(self.env.observation_space[i+1].shape)
            else:
                obs_shape_n = [self.env.observation_space[i].shape for i in range(self.env.n)]

            num_adversaries = min(self.env.n, self.num_adversaries)
            arglist = argparse.Namespace(**{"good_policy":self.good_policy,
                       "adv_policy": self.adv_policy,
                       "lr": self.lr,
                       "num_units": self.num_units,
                       "batch_size": self.batch_size,
                       "max_episode_len": self.max_episode_len,
                       "gamma": self.gamma,
                        "n_steps": self.n_steps
                       })
            self.trainers = U.get_trainers(self.env, num_adversaries, obs_shape_n, arglist)
            logging.info(f"Using good policy {self.good_policy} and adv policy {self.adv_policy}")

            U.initialize()
            if self.load_dir == '':
                self.load_dir = self.save_dir

            saver = None
            if self.display or self.restore or self.benchmark:
                logging.info("Loading previous state...")
                saver = tf.train.import_meta_graph(self.load_dir+self.model_name+'.meta')
                U.load_state(tf.train.latest_checkpoint(self.load_dir), saver=saver)

            if saver is None:
                saver = U.get_saver()

            episode_rewards = [0.0]
            agent_rewards = [[0.0] for _ in range(self.env.n)]

            final_ep_rewards = []
            final_ep_ag_rewards = []
            agent_info = [[[]]]
            obs_n = self.env.reset()

            episode_step = 0
            current_episode = 0
            train_step = 0
            t_start = time.time()
            pbar = tqdm(total=self.num_episodes)

            while True:
                #print(f'episodes: {len(episode_rewards)}, train steps: {train_step}')
                action_n = self.predict(obs_n)

                clipped_action_n = action_n
                for i, _ in enumerate(self.env.action_space):
                    if isinstance(_ , gym.spaces.Box):
                        clipped_action_n[i] = np.clip(action_n[i], self.env.action_space[i].low, self.env.action_space[i].high)

                #print(f"action_n: {action_n}")
                new_obs_n, rew_n, done_n, info_n = self.env.step(clipped_action_n)

                episode_step +=1
                done = all(done_n)
                terminal = (episode_step > self.max_episode_len)

                # experience
                for i, agent in enumerate(self.trainers):
                    agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)

                obs_n = new_obs_n

                for i, rew in enumerate(rew_n):
                    episode_rewards[-1] += rew
                    if not ONLY_SELLER:
                        agent_rewards[int(i / 2)][-1] += rew
                    else:
                        agent_rewards[i][-1] += rew

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
                        logging.info("Finished benchmarking, now saving....")
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
                        logging.debug(f"{agent}'s loss is {loss}")

                ##############################################################################
                # save model
                # display training output
                ##############################################################################
                if terminal and (len(episode_rewards) % self.save_rate == 0):
                    # save the model separately
                    if self.save_trainers:
                        for _ in self.trainers:
                            U.save_as_scope(_.name, save_dir=self.save_dir, model_name=self.model_name)
                    # save all model paramters
                    U.save_state(self.save_dir + self.model_name, saver=saver)

                    if num_adversaries == 0:
                        logging.info(f"steps: {train_step}, episodes: {len(episode_rewards)}, "
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
                    logging.info(f'...Finished total of {len(episode_rewards)} episodes')
                    break

    def predict(self, obs_n, train=True):
        if train:
            action_n = [agent.action(obs) for agent, obs in zip(self.trainers, obs_n)]
            return action_n
        else:
            with U.single_threaded_session():
                U.initialize()
                if self.load_dir == '':
                    self.load_dir = self.save_dir

                logging.info("loading model...")
                saver = tf.train.import_meta_graph(self.save_dir + self.model_name + ".meta")
                U.load_state(tf.train.latest_checkpoint(self.save_dir), saver=saver)

                action_n = [agent.action(obs) for agent, obs in zip(self.trainers, obs_n)]
                return action_n
