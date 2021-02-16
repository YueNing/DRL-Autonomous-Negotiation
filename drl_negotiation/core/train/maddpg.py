"""
A2C model
"""
import os
import time
import argparse
import tensorflow as tf
import gym
from drl_negotiation.core.envs.multi_agents_scml import SCMLEnv
import drl_negotiation.core.utils.tf_utils as U
from drl_negotiation.core.utils.multi_agents_utils import get_trainers
import numpy as np
import pickle
from tqdm import tqdm
from drl_negotiation.core.config.hyperparameters import (ONLY_SELLER, LOGGING_LEVEL, UPDATE_TRAINER_STEP, SAVE_DIR,
                                                         MODEL_NAME, SAVE_RATE, BATCH_SIZE, TRAIN_EPISODES,
                                                         MAX_EPISODE_LEN, PLOTS_DIR, SAVE_TRAINERS)
from drl_negotiation.core.train._model import ModelResult
import logging


class MADDPGModel:
    trained_model = None

    def __init__(self,
                 env: SCMLEnv = None,
                 policy=None,
                 only_seller=ONLY_SELLER,
                 logging_level=LOGGING_LEVEL,

                 # training
                 # train update steps
                 n_steps=UPDATE_TRAINER_STEP,
                 # learning rate
                 lr=1e-2,
                 # discount factor
                 gamma=0.99,
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
                 batch_size=BATCH_SIZE,
                 num_units=64,
                 # env
                 n_envs=1,
                 # number of training episodes
                 num_episodes=TRAIN_EPISODES,
                 # max length of every episode
                 max_episode_len=MAX_EPISODE_LEN,
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
                 plots_dir=PLOTS_DIR,
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

        # U.logging_setup()

        if _init_setup_model:
            self.setup_model()

        self.save_trainers = save_trainers

    def setup_model(self):
        with U.single_threaded_session():
            if ONLY_SELLER:
                obs_shape_n = [self.env.observation_space[i].shape for i in range(self.env.n)]
            else:
                obs_shape_n = []
                for i in range(self.env.n):
                    obs_shape_n.append(self.env.observation_space[i].shape)
                    obs_shape_n.append(self.env.observation_space[i + 1].shape)

            num_adversaries: int = min(self.env.n, self.num_adversaries)
            self._get_trainers(num_adversaries, obs_shape_n)
        # assert issubclass(self.policy, Policy), "Error: the input policy for the maddpg model must be an" \
        #                                         "instance of a2c.policy.Policy"

        # self.graph = tf.Graph()

    def _get_trainers(self, num_adversaries, obs_shape_n):
        arglist = argparse.Namespace(**{"good_policy"    : self.good_policy,
                                        "adv_policy"     : self.adv_policy,
                                        "lr"             : self.lr,
                                        "num_units"      : self.num_units,
                                        "batch_size"     : self.batch_size,
                                        "max_episode_len": self.max_episode_len,
                                        "gamma"          : self.gamma,
                                        "n_steps"        : self.n_steps
                                        })
        self.trainers = get_trainers(self.env, num_adversaries, obs_shape_n, arglist)
        logging.info(f"Using good policy {self.good_policy} and adv policy {self.adv_policy}")

    def _setup_learn(self):
        """
        Check the environment
        """
        if self.env is None:
            raise ValueError("Error: cannot train the model without a valid environment, please set an environment "
                             "with set_env(self, env) method.")

    def learn(self, train_episodes: int = None) -> object:
        """
        learning process
        Args:
            train_episodes:

        Returns:
            final_ep_rewards: mean episode rewards
            final_ep_ag_rewards: mean episode agents rewards
            episode_rewards: episode rewards
            self.env: final environment
        """
        self._setup_learn()
        if train_episodes is not None:
            self.num_episodes = train_episodes

        with U.single_threaded_session():
            if not ONLY_SELLER:
                obs_shape_n = []
                for i in range(self.env.n):
                    obs_shape_n.append(self.env.observation_space[i * 2].shape)
                    obs_shape_n.append(self.env.observation_space[i * 2 + 1].shape)
            else:
                obs_shape_n = [self.env.observation_space[i].shape for i in range(self.env.n)]

            num_adversaries = min(self.env.n, self.num_adversaries)
            self._get_trainers(num_adversaries, obs_shape_n)

            U.initialize()
            if self.load_dir == '':
                self.load_dir = self.save_dir

            saver = None
            if self.display or self.restore or self.benchmark:
                logging.info("Loading previous state...")
                # saver = tf.train.import_meta_graph(self.load_dir + self.model_name + '.meta')
                U.load_state(tf.train.latest_checkpoint(self.load_dir), saver=saver)

            if saver is None:
                saver = U.get_saver()

            episode_rewards = [0.0]
            episode_extra_rewards = [0.0]
            agent_rewards = [[0.0] for _ in range(self.env.n)]
            extra_agent_rewards = [[0.0] for _ in range(self.env.extra_n)]

            final_ep_rewards = []
            final_ep_extra_rewards = []
            final_ep_ag_rewards = []
            final_ep_extra_ag_rewards = []
            agent_info = [[[]]]
            obs_n, _ = self.env.reset()

            # episode_step = 0
            train_step = 0
            t_start = time.time()
            pbar = tqdm(total=self.num_episodes)

            while True:
                logging.debug(
                    f'episodes: {len(episode_rewards)}, episode_step: {self.env.step_cnt}, train steps: {train_step}')
                action_n = self.predict(obs_n)

                clipped_action_n = action_n
                for i, _ in enumerate(self.env.action_space):
                    if isinstance(_, gym.spaces.Box):
                        clipped_action_n[i] = np.clip(action_n[i], self.env.action_space[i].low,
                                                      self.env.action_space[i].high)

                # print(f"action_n: {action_n}")
                es = self.env.step(clipped_action_n)
                done = all(es.last)
                logging.debug(f"espisode last step type: {es.last}")
                # experience
                for i, agent in enumerate(self.trainers):
                    agent.experience(obs_n[i], action_n[i], es.reward[i], es.observation[i], es.last[i], es.timeout)

                obs_n = es.observation

                logging.debug(f"episode_step:{self.env.step_cnt}, es.reward are {es.reward}")
                for i, rew in enumerate(es.reward):
                    episode_rewards[-1] += rew
                    if not ONLY_SELLER:
                        agent_rewards[int(i / 2)][-1] += rew
                    else:
                        agent_rewards[i][-1] += rew

                for i, rew in enumerate(es.extra_rew):
                    episode_extra_rewards[-1] += rew
                    if not ONLY_SELLER:
                        extra_agent_rewards[int(i / 2)][-1] += rew
                    else:
                        extra_agent_rewards[i][-1] += rew

                ##############################################################################
                # save model
                # display training output
                ##############################################################################
                if all(es.timeout) and (len(episode_rewards) % self.save_rate == 0):
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
                    final_ep_extra_rewards.append(np.mean(episode_extra_rewards[-self.save_rate:]))
                    for rew in agent_rewards:
                        final_ep_ag_rewards.append(np.mean(rew[-self.save_rate:]))
                    for rew in extra_agent_rewards:
                        final_ep_extra_ag_rewards.append(np.mean(rew[-self.save_rate:]))

                # Evaluate, benchmarking learned policies
                if self.benchmark:
                    for i, info in enumerate(es.env_info):
                        agent_info[-1][i].append(es.env_info['n'])

                    if train_step > self.benchmark_iters and (done or es.timeout):
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

                # learn, update all policies in train, if not in display or benchmark mode
                for agent in self.trainers:
                    agent.preupdate()
                for agent in self.trainers:
                    info = agent.update(self.trainers, train_step)
                    if info is not None:
                        logging.debug(f"{agent}'s [q_loss, p_loss, np.mean(target_q), np.mean(rew), "
                                      f"np.mean(target_q_next), np.std(target_q)] are {info}")

                #####################################################################################
                # Prepare for next episode, if done
                #####################################################################################
                if done:
                    obs_n, _ = self.env.reset()
                    pbar.update(1)
                    episode_rewards.append(0)
                    episode_extra_rewards.append(0)
                    for a in agent_rewards:
                        a.append(0)
                    for a in extra_agent_rewards:
                        a.append(0)
                    agent_info.append([[]])

                train_step += 1

                ##############################################################################
                # saves final episode reward for plotting training curve
                ##############################################################################
                if len(episode_rewards) > self.num_episodes:
                    os.makedirs(self.plots_dir, exist_ok=True)
                    module_path = os.getcwd()
                    rew_file_name = self.plots_dir + self.exp_name + "_rewards.pkl"
                    rew_file_name = os.path.join(module_path, rew_file_name)
                    with open(rew_file_name, 'wb') as fp:
                        pickle.dump(final_ep_rewards, fp)
                    agrew_file_name = self.plots_dir + self.exp_name + '_agrewards.pkl'
                    extra_agrew_file_name = self.plots_dir + self.exp_name + '_extra_agrewards.pkl'
                    with open(agrew_file_name, 'wb') as fp:
                        pickle.dump(final_ep_ag_rewards, fp)
                    with open(extra_agrew_file_name, 'wb') as fp:
                        pickle.dump(final_ep_extra_ag_rewards, fp)
                    logging.info(f'...Finished total of {len(episode_rewards)} episodes')
                    break

            return ModelResult(
                final_ep_rewards=np.array(final_ep_rewards),
                final_ep_extra_rewards=np.array(final_ep_extra_rewards),
                final_ep_ag_rewards=np.array(final_ep_ag_rewards),
                final_ep_extra_ag_rewards=np.array(final_ep_extra_ag_rewards),
                episode_rewards=np.array(episode_rewards),
                episode_extra_rewards=np.array(episode_extra_rewards),
                env=self.env
            )
            # return final_ep_rewards, final_ep_extra_rewards, final_ep_ag_rewards, final_ep_extra_ag_rewards, episode_rewards, episode_extra_rewards, self.env

    def predict(self, obs_n, train=True):
        if train:
            action_n = [agent.action(obs) for agent, obs in zip(self.trainers, obs_n)]
            return np.array(action_n)
        else:
            with U.single_threaded_session():
                U.initialize()
                if self.load_dir == '':
                    self.load_dir = self.save_dir

                logging.info("loading model...")
                saver = tf.train.import_meta_graph(self.save_dir + self.model_name + ".meta")
                U.load_state(tf.train.latest_checkpoint(self.save_dir), saver=saver)

                action_n = [agent.action(obs) for agent, obs in zip(self.trainers, obs_n)]
                return np.array(action_n)
