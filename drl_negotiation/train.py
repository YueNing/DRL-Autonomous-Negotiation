from .env import NEnv
from datetime import datetime

MODEL_NEGOTIATION = [
    "DQN",
    "PPO1"
]

def train_negotiation(plot=True, model="DQN", env=None, monitor=True, num_timesteps=1000, eval_freq=100, eval_episodes=2, seed=721, LOGDIR=None):
    
    if model not in MODEL_NEGOTIATION:
        return False, None
    
    import os
    from typing import Optional

    from stable_baselines import logger
    from stable_baselines.common.callbacks import EvalCallback
    from stable_baselines.common.policies import MlpPolicy

    NUM_TIMESTEPS = int(num_timesteps)
    SEED = seed
    EVAL_FREQ = eval_freq
    EVAL_EPISODES = eval_episodes
    if LOGDIR is None:
        LOGDIR = "train_negotiation_" + model

    logger.configure(folder=LOGDIR)
    
    if env is None:
        assert False, "env is None, Must set env!"
    
    env = env

    def _monitor_env(env: Optional[NEnv]=None, monitor: bool=False):
        
        assert isinstance(env,  NEnv), "must set the env corretly!"

        if monitor:
            from stable_baselines.bench import Monitor
            env = Monitor(env, LOGDIR)
        
        env.seed(SEED)
        return env

    if model == "DQN":
        # train the acceptance strategy
        from stable_baselines import DQN
        env = _monitor_env(
            env=env,
            monitor=True
        )
        
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=1e-3,
            prioritized_replay=True,
            verbose=1,
            tensorboard_log=os.path.join(LOGDIR, "tensorboard")
        )

        eval_callback = EvalCallback(
            env, best_model_save_path=LOGDIR, log_path=LOGDIR, eval_freq=EVAL_FREQ, n_eval_episodes=EVAL_EPISODES
        )

        model.learn(total_timesteps=NUM_TIMESTEPS)
        model.save(os.path.join(LOGDIR, "final_model"))
        
        env.close()

        if plot:
            import matplotlib.pyplot as plt
            from stable_baselines import results_plotter

            results_plotter.plot_results(
                [LOGDIR],
                NUM_TIMESTEPS,
                results_plotter.X_TIMESTEPS,
                f"{model}_negotiation"
            )
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            plt.savefig(
                f"{now}_{model}_result.png",
                dpi=600
            )

    if model == "PPO1":
        from stable_baselines.ppo1 import PPO1
        # TODO: train the offer/bidding strategy
        model = PPO1(MlpPolicy, env, verbose=1, tensorboard_log=os.path.join(LOGDIR, "tensorboard"))
        model.learn(total_timesteps=1000)
        model.save(os.path.join(LOGDIR, "final_model"))

    return True, f"Finished! you can get the result at {LOGDIR}"

############################################################################
# For SCML
#
#
############################################################################
import MADDPGAgentTrainer

def parse_args():
    parse = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent supply chain managerment environments")
    
    # env
    parser.add_argument('--scenario', type=str, default="scml", help="name of the scenario script")
    parser.add_argument('--num-episodes', type=int, default=60000, help="number of episodes")
    parser.add_argument('--max-episode-len', type=int, default=100, help="maximum episode length")
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

def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        output = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_conencted(out, num_outputs=num_outputs, activation_fn=None)
        return out

def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
    
    for i in range(num_adversaries):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func = (arglist.adv_policy =='ddpg')
            ))
    
    for i in range(num_adversaries, env.n):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func = (arglist.good_policy=="ddpg")
            )
        )
    
    return trainers

def make_env(scenario_name, arglist):
    from drl_negoiation.env import SCMLEnv
    import  drl_negotiation.scenarios as scenarios
    
    # load scenario from script
    scenario = scenarios.load(scenario_name + '.py').Scenario()
    
    # create world/game
    world = scenario.make_world()

    # create multi-agent supply chain management environment
    env = SCMLEnv()
    return env


def train_scml(arglist):
    import common.tf_util as U

    with U.single_threaded_session():
        # create environment
        env = make_env(arglist.scenario, arglist, arglist.benchmark)

        # create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
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

        saver = tf.train.Saver()
        obs_n = env.reset()

        episode_step = 0
        train_step = 0
        t_start = time.time()

        print('Starting iterations....')
        while True:
            # get the joint action based on joint obs
            action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
            # environment
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
