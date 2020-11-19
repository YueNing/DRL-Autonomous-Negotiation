from scml_env import DRLNegotiationEnv, NEnv

MODEL_NEGOTIATION = [
    "DQN",
    "PPO1"
]

def train_negotiation(plot=True, model="DQN", env=None, monitor=True):
    
    if model not in MODEL_NEGOTIATION:
        return False, None
    
    import os
    import gym
    from typing import Optional

    from stable_baselines import logger
    from stable_baselines.common.base_class import BaseRLModel
    from stable_baselines.common.callbacks import EvalCallback

    NUM_TIMESTEPS = int(1000)
    SEED = 721
    EVAL_FREQ = 100
    EVAL_EPISODES = 2
    LOGDIR = "train_negotiation_" + model

    logger.configure(folder=LOGDIR)
    
    if env is None:
        env = DRLNegotiationEnv(
            name="default_negotiation_env"
        )
    
    env = env

    def _monitor_env(env: Optional[NEnv]=None, monitor: bool=False):
        
        assert isinstance(env,  NEnv), "must set the env corretly!"

        if monitor:
            from stable_baselines.bench import Monitor
            env = Monitor(env, LOGDIR)
        
        env.seed(SEED)
        return env

    if model == "DQN":
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
            verbose=1
        )

        eval_callback = EvalCallback(
            env, best_model_save_path=LOGDIR, log_path=LOGDIR, eval_freq=EVAL_FREQ, n_eval_episodes=EVAL_EPISODES
        )

        model.learn(total_timesteps=NUM_TIMESTEPS)
        model.save(os.path.join(LOGDIR, "final_model"))
        
        env.close()

    
    if model == "PPO1":
        from stable_baselines.ppo1 import PPO1
        # TODO:

    if plot:
        import matplotlib.pyplot as plt
        from stable_baselines import results_plotter

        results_plotter.plot_results(
            [LOGDIR],
            NUM_TIMESTEPS,
            results_plotter.X_TIMESTEPS,
            f"{model}_negotiation"
        )
        
        plt.savefig(
            f"{model}_result.png",
            dpi=600
        )
    
    return True, f"Finished! you can get the result at {LOGDIR}"
        
    

    

def train_scml():
    pass