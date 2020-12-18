'''
    For SCML: train a maddpg agent for supply chain management
    Author: naodongbanana
    E-Mail: n1085633848@outlook.com
'''

import os, sys
sys.path.append("/home/nauen/PycharmProjects/tn_source_code")

import time
import argparse
import drl_negotiation.utils as U
import tensorflow as tf
import tensorflow.contrib.layers as layers
from drl_negotiation.a2c.trainer import MADDPGAgentTrainer

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent supply chain managerment environments")
    
    # env
    parser.add_argument('--scenario', type=str, default="scml", help="name of the scenario script")
    parser.add_argument('--num-episodes', type=int, default=60000, help="number of episodes")
    parser.add_argument('--max-episode-len', type=int, default=100, help="maximum episode length")
    parser.add_argument('--num-adversaries', type=int, default=0, help="number of adversaries")
    parser.add_argument('--good-policy', type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="heuristic", help="policy of adversaries")

    # Training
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    parser.add_argument("--exp-name", type=str, default=None, help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="/tmp/policy/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are compeleted")
    parser.add_argument("--load-dir", type=str, default='', help="directory in which training state and model are loaded")
    
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")

    return parser.parse_args()

def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        out = input
        output = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
   
    # first set up the adversaries, default num_adversaries is 0
    for i in range(num_adversaries):
        trainers.append(trainer(
            env.agents[i].name.replace("@", '-'), model, obs_shape_n, env.action_space, i, arglist,
            local_q_func = (arglist.adv_policy =='ddpg')
            ))

    # set up the good agent 
    for i in range(num_adversaries, env.n):
        trainers.append(trainer(
            env.agents[i].name.replace("@", '-'), model, obs_shape_n, env.action_space, i, arglist,
            local_q_func = (arglist.good_policy=="ddpg")
            )
        )
    
    return trainers

def make_env(scenario_name, arglist):
    from drl_negotiation.env import SCMLEnv
    import drl_negotiation.scenarios as scenarios
    
    # load scenario from script
    scenario = scenarios.load(scenario_name + '.py').Scenario()
    
    # create world/game
    world = scenario.make_world()

    # create multi-agent supply chain management environment
    env = SCMLEnv(
            world,
            reset_callback=scenario.reset_world,
            reward_callback=scenario.reward,
            observation_callback=scenario.observation,
            info_callback=None,
            done_callback=scenario.done,
            shared_viewer=False
            )
    
    return env

# scope
def scope_name():
    return tf.get_variable_scope().name

def scope_vars(scope, trainable_only=False):
    """
        get the paramters inside a scope
    """
    return tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES if trainable_only else tf.GraphKeys.GLOBAL_VARIABLES,
            scope=scope if isinstance(scope, str) else scope.name
            )
def absolute_scope_name(relative_scope_name):
    return scope_name() + "/" + relative_scope_name

def train(arglist):

    with U.single_threaded_session():
        # create environment
        env = make_env(arglist.scenario, arglist)
        
        # create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        import ipdb
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        print(f"Using good policy {arglist.good_policy} and adv policy {arglist.adv_policy}")

        U.initialize()

        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir

        if arglist.display or arglist.restore or arglist.benchmark:
            print("Loading previous state...")
            U.load_state(arglist.load_dir)

        
        episode_rewards = [0.0]
        agent_rewards = [[0.0] for _ in range(env.n)]

        final_ep_rewards = []
        final_ep_ag_rewards = []
        agent_info = [[[]]]

        saver = tf.compat.v1.train.Saver()
        obs_n = env.reset()

        episode_step = 0
        train_step = 0
        t_start = time.time()

        print('Starting iterations....')
        while True:

            # get the joint action based on joint obs
            action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
            
            # environment
            #ipdb.set_trace()
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)

            episode_step +=1
            done = all(done_n)
            terminal = (episode_step >=arglist.max_episode_len)

            # experience
            for i, agent in enumerate(trainers):
                agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)

            obs_n = new_obs_n

            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew

            if done or terminal:
                obs_n = env.reset()
                episode_step = 0
                episode_rewards.append(0)
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([[]])

            train_step +=1

            # for benchmarking learned policies
            if arglist.benchmark:
                for i, info in enumerate(info_n):
                    agent_info[-1][i].append(info_n['n'])

                if train_step > arglist.benchmark_iters and (done or terminal):
                    file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
                    print("Finished benchmarking, now saving...")
                    with open(file_name, 'wb') as fp:
                        pickle.dump(agent_info[:-1], fp)
                    break
                
                continue

            if arglist.display:
                time.sleep(0.1)
                env.render()
                continue

            loss = None
            for agent in trainers:
                agent.preupdate()
            for agent in trainers:
                loss = agent.update(trainers, train_step)

            ####################################################
            # save model
            # display traning output
            ####################################################
            if terminal and(len(episode_rewards) % arglist.save_rate==0):
                U.save_state(arglist.save_dir, saver=saver)

                if num_adversaries ==0:
                    print()


if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)
