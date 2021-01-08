'''
    Logic of training the maddpg
    For SCML: train a maddpg agent for supply chain management
    Author: naodongbanana
    E-Mail: n1085633848@outlook.com
'''

import sys
sys.path.append("/home/nauen/PycharmProjects/tn_source_code")

import time
import drl_negotiation.utils as U
import tensorflow.compat.v1 as tf
import pickle

def train(arglist):

    with U.single_threaded_session():
        # create environment
        env = U.make_env(arglist.scenario, arglist)
        
        # create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        import ipdb
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = U.get_trainers(env, num_adversaries, obs_shape_n, arglist)
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
            print(f"episodes {len(episode_rewards)}")
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

            #######################################################
            # saves final episode reward for plotting training curve
            #
            ########################################################
            if len(episode_rewards) > arglist.num_episodes:
                print(f"...Finished total of {len(episode_rewards)} episodes")
                break

if __name__ == '__main__':
    arglist = U.parse_args()
    train(arglist)
