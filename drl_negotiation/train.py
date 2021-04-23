from .env import NEnv
from datetime import datetime

MODEL_NEGOTIATION = [
    "DQN",
    "PPO1",
    "PPO2",
    "GAIL",
    "A2C",
    "ACER",
    "DDPG",
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
    if model == "ACER":
        from stable_baselines import ACER

        model = ACER(MlpPolicy, env, verbose=1, tensorboard_log=os.path.join(LOGDIR, "tensorboard"))
        model.learn(total_timesteps=NUM_TIMESTEPS)
        model.save(os.path.join(LOGDIR, "final_model"))

    if model == "PPO1" or model == "PPO2":
        from stable_baselines.ppo1 import PPO1
        from stable_baselines import PPO2
        # TODO: train the offer/bidding strategy
        if model == "PPO1":
            model = PPO1(MlpPolicy, env, verbose=1, tensorboard_log=os.path.join(LOGDIR, "tensorboard"))
        else:
            model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log=os.path.join(LOGDIR, "tensorboard"))
        model.learn(total_timesteps=NUM_TIMESTEPS)
        model.save(os.path.join(LOGDIR, "final_model"))

    if model == "GAIL":
        from stable_baselines import PPO2, GAIL
        from stable_baselines.gail import generate_expert_traj, ExpertDataset

        model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log=os.path.join(LOGDIR, "tensorboard"))
        generate_expert_traj(model, "expert_pendulum", n_timesteps=1000)

        dataset = ExpertDataset(expert_path="expert_pendulum.npz", traj_limitation=10, verbose=1)
        model = GAIL(MlpPolicy, env, dataset, verbose=1, tensorboard_log=os.path.join(LOGDIR, "tensorboard"))
        model.learn(total_timesteps=num_timesteps)
        model.save(os.path.join(LOGDIR, "final_model"))

    if model == "A2C":
        from stable_baselines import A2C
        model = A2C(MlpPolicy, env, verbose=1, tensorboard_log=os.path.join(LOGDIR, "tensorboard"))
        model.learn(total_timesteps=NUM_TIMESTEPS)
        model.save(os.path.join(LOGDIR, "final_model"))

    if model == "DDPG":
        import numpy as np
        from stable_baselines import DDPG
        from stable_baselines.common.noise import OrnsteinUhlenbeckActionNoise
        from stable_baselines.ddpg.policies import MlpPolicy

        n_actions = env.action_space.shape[-1]
        param_noise = None
        action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))
        model = DDPG(MlpPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise,
                     tensorboard_log=os.path.join(LOGDIR, "tensorboard"))
        model.learn(total_timesteps=NUM_TIMESTEPS)
        model.save(os.path.join(LOGDIR, "final_model"))

    return True, f"Finished! you can get the result at {LOGDIR}"