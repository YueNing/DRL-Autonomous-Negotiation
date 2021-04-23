import scml
from abc import ABC
from drl_negotiation.core.games._game import TrainableAgent
from scml.oneshot.agent import OneShotAgent
from scml import QUANTITY, UNIT_PRICE
from scml.oneshot.awi import OneShotAWI

class MyOneShotAgent(OneShotAgent, TrainableAgent, ABC):
    """Running in the SCML OneShot"""
    def done(self):
        pass

    def info(self):
        pass

    def reward(self, index):
        current_offer = []
        agent_id = f"{self.awi.agent.id}_{index}"
        reward = []
        # for negotiator_id, negotiator in self.negotiators.items():
        #     current_offer.append(negotiator[0].ami.state.current_offer)
        #     reward.append(-1 if negotiator[0].ami.state.agreement is None else 1)
        if self.awi._world.train_world.broken[agent_id]:
            reward.append(-1)
        elif self.awi._world.train_world.success[agent_id]:
            contract = self.awi._world.train_world.contract[agent_id]
            if contract is None:
                reward.append(0)
            else:
                if scml.__version__ == "0.3.1":
                    reward.append(self.ufun([contract]).mean())
                else:
                    reward.append(1)
                    # reward.append(contract.agreement["quantity"]*0.6+contract.agreement["unit_price"]*1)
        elif self.awi._world.train_world.running[agent_id]:
            reward.append(0)
        else:
            reward.append(-1)

        if not reward:
            return 0

        return reward[0]

    def observation(self):
        """TODO: Returns observation for agent, the observation is composed of:
             - negotiation issues ranges,
             - my last proposed offer
             - current offer in the negotiation
         """
        my_last_proposal = []
        issues = []
        current_offers = []
        catalog_prices = []
        for negotiator_id, negotiator in self.negotiators.items():
            my_last_proposal.append(negotiator[0].my_last_proposal)
            issues.append(negotiator[0].ami.issues)
            current_offers.append(negotiator[0].ami.state.current_offer)

        # convert offer to onehot encoding
        try:
            # control_offer = current_offers[self.awi.current_step*2:(self.awi.current_step+1)*2]
            current_offer = []
            control_offers = [current_offers[i::len(self.awi.my_consumers)] for i in range(len(self.awi.my_consumers))]
            for control_offer in control_offers:
                if not control_offer or self.awi.current_step == self.awi.n_steps:
                    current_offer.append(np.zeros(11 * 101))
                elif control_offer[self.awi.current_step] is None:
                    current_offer.append(np.zeros(11 * 101))
                else:
                    current_offer.append(np.eye(11 * 101)[int(control_offer[self.awi.current_step][QUANTITY]*101 +
                                                              control_offer[self.awi.current_step][UNIT_PRICE])])
        except Exception as e:
            print(f"Error when observing {e}!")

        if not current_offer:
            print("Helll")
        # catalog price 20 * 40,
        catalog_prices = self.awi.catalog_prices
        if catalog_prices[1] > 30:
            raise NotImplementedError

        if catalog_prices[2] > 50:
            raise NotImplementedError

        # catalog_prices = np.eye(20*40)[int((catalog_prices[1] - 10 + 1) * (catalog_prices[2] - 10 + 1))]
        # catalog_prices = np.eye(21*41)[int((catalog_prices[1] - 10) * 41 + (catalog_prices[2] - 10))]
        # production_cost = np.eye(10)[int(self.awi.profile.cost) - 1]
        # step = np.eye(11)[int(self.awi.current_step)]

        # return np.concatenate(
        #     (
        #         current_offer,
        #         catalog_prices,
        #         production_cost,
        #         # step
        #     )
        # )
        return current_offer

    def reset(self):
        pass

    def set_action(self, action):
        pass

    def last_action(self):
        pass

    def get_avail_actions(self):
        pass

    def policy(self, negotiator_id, state):
        """return the action, get policy_callback from the trainer"""
        # agent_num = int(self.awi.agent.name[-1])
        if self.negotiators[negotiator_id][0].ami.annotation["seller"] == self.awi.agent.id:
            # agent_num = list(self.awi._world.train_world.policy_agents.keys()).index(self.awi.agent.id) * \
            #             len(self.awi.my_consumers) + self.awi.my_consumers.index(self.negotiators[negotiator_id][0].ami.annotation["buyer"])
            # agent_num = list(self.awi._world.train_world.policy_agents.keys()).index(self.awi.agent.id) + \
            #             self.awi.my_consumers.index(self.negotiators[negotiator_id][0].ami.annotation["buyer"]) * 2
            agent_id = f"{self.awi.agent.id}_{self.awi.my_consumers.index(self.negotiators[negotiator_id][0].ami.annotation['buyer'])}"
            agent_num = 0
        else:
            raise NotImplementedError
        epsilon = self.awi._world.train_world.rollout_worker.tmp_epsilon
        obs = self.awi._world.train_world.tmp_obs_dict[agent_id]
        issues = self.negotiators[negotiator_id][0].ami.issues
        last_action = self.awi._world.train_world.rollout_worker.tmp_last_action_dict[agent_id]
        avail_action = self.awi._world.train_world.env.get_avail_agent_actions(agent_id, issues)
        action = self.awi._world.train_world.rl_runner.agents.choose_action(obs, last_action, self.awi._world.train_world.env.trainable_agents.index(agent_id), avail_action, epsilon)

        action_onehot = np.zeros(self.awi._world.train_world.rl_runner.args.n_actions)
        action_onehot[action] = 1
        # self.awi._world.train_world.tmp_actions.append(int(action))
        self.awi._world.train_world.tmp_actions_dict[agent_id] = int(action)
        self.awi._world.train_world.tmp_actions_onehot_dict[agent_id] = action_onehot
        self.awi._world.train_world.tmp_avail_actions_dict[agent_id] = avail_action
        self.awi._world.train_world.rollout_worker.tmp_last_action_dict[agent_id] = action_onehot
        self.awi._world.train_world.set_agent.append(agent_id)
        return int(action)

import torch
import numpy as np
from torch.distributions import Categorical


class Agents:
    """Trained Agents, relates to Agents running in the SCML OneShot"""
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        from drl_negotiation.core.trainer.mixer.qmixer import QMixer
        self.policy = QMixer(args)

        self.args = args

    def choose_action(self, obs, last_action, agent_num, avail_actions, epsilon, maven_z=None, evaluate=False):
        inputs = obs.copy()
        avail_actions_ind = np.nonzero(avail_actions)[0]  # index of actions which can be choose

        # transform agent_num to onehot vector
        agent_id = np.zeros(self.n_agents)
        agent_id[agent_num] = 1.

        if self.args.last_action:
            inputs = np.hstack((inputs, last_action))
        if self.args.reuse_network:
            inputs = np.hstack((inputs, agent_id))
        hidden_state = self.policy.eval_hidden[:, agent_num, :]

        # transform the shape of inputs from (42,) to (1,42)
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
            hidden_state = hidden_state.cuda()

        # get q value
        if self.args.alg == 'maven':
            maven_z = torch.tensor(maven_z, dtype=torch.float32).unsqueeze(0)
            if self.args.cuda:
                maven_z = maven_z.cuda()
            q_value, self.policy.eval_hidden[:, agent_num, :] = self.policy.eval_rnn(inputs, hidden_state, maven_z)
        else:
            q_value, self.policy.eval_hidden[:, agent_num, :] = self.policy.eval_rnn(inputs, hidden_state)

        # choose action from q value
        if self.args.alg == 'coma' or self.args.alg == 'central_v' or self.args.alg == 'reinforce':
            action = self._choose_action_from_softmax(q_value.cpu(), avail_actions, epsilon, evaluate)
        else:
            q_value[avail_actions == 0.0] = - float("inf")
            if np.random.uniform() < epsilon:
                action = np.random.choice(avail_actions_ind)  # action is int
            else:
                action = torch.argmax(q_value)
        return action

    def _choose_action_from_softmax(self, inputs, avail_actions, epsilon, evaluate=False):
        """
        :param inputs: # q_value of all actions
        """
        action_num = avail_actions.sum(dim=1, keepdim=True).float().repeat(1, avail_actions.shape[-1])  # num of avail_actions
        prob = torch.nn.functional.softmax(inputs, dim=-1)
        # add noise of epsilon
        prob = ((1 - epsilon) * prob + torch.ones_like(prob) * epsilon / action_num)
        prob[avail_actions == 0] = 0.0

        if epsilon == 0 and evaluate:
            action = torch.argmax(prob)
        else:
            action = Categorical(prob).sample().long()
        return action

    def _get_max_episode_len(self, batch):
        try:
            terminated = batch['terminated']
        except Exception as e:
            raise ValueError("get key terminated error!")
        episode_num = terminated.shape[0]
        max_episode_len = 0
        for episode_idx in range(episode_num):
            for transition_idx in range(self.args.episode_limit):
                if terminated[episode_idx, transition_idx, 0] == 1:
                    if transition_idx + 1 >= max_episode_len:
                        max_episode_len = transition_idx + 1
                    break
        return max_episode_len

    def train(self, batch, train_step, epsilon=None):  # coma needs epsilon for training

        # different episode has different length, so we need to get max length of the batch
        max_episode_len = self._get_max_episode_len(batch)
        for key in batch.keys():
            if key != 'z':
                batch[key] = batch[key][:, :max_episode_len]
        self.policy.learn(batch, max_episode_len, train_step, epsilon)
        if train_step > 0 and train_step % self.args.save_cycle == 0:
            self.policy.save_model(train_step)
