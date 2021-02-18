import os, ray
from ray import tune
from ray.tune import register_env, grid_search
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.agents import qmix
from ray.rllib.agents import pg
from drl_negotiation.core.utils.common import parse_args
from drl_negotiation.core.utils.multi_agents_utils import make_env
from drl_negotiation.core.envs.multi_agent_ray import RaySCMLEnv

if __name__ == '__main__':
    args = parse_args()
    env_creator = lambda config: make_env(config.get("name", "scml_oneshot"), seed=config.get("seed", 10))

    # get observation spaces, act_spaces
    test_env = RaySCMLEnv(env_creator({}))
    obs_spaces = test_env.observation_spaces
    act_spaces = test_env.action_spaces
    agents = test_env.agents

    register_env('scml_oneshot_env', lambda config: RaySCMLEnv(env_creator(config)))

    if args.run == "QMIX":
        config = {
            "rollout_fragment_length": 4,
            "train_batch_size": 32,
            "exploration_config": {
                "epsilon_timesteps": 5000,
                "final_epsilon": 0.05,
            },
            "num_workers": 0,
            "mixer": grid_search([None, "qmix", "vdn"]),
            "env_config": {
                "separate_state_space": True,
                "one_hot_state_encoding": True
            },
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
            "framework": "torch" if args.torch else "tf",
        }
        group = True
    else:
        config = {
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
            "framework": "torch" if args.torch else "tf",
            "env_config": {
                "name": "scml_oneshot",
                "normalize": False,
                "seed": 10
            }
        }
        group = False

    ray.init(num_cpus=args.num_cpus or None)

    stop = {
        "episode_reward_mean": args.stop_reward,
        "timesteps_total": args.stop_timesteps
    }

    config = dict(config, **{
        "env": "scml_oneshot_env"
    })

    trainer = pg.PGTrainer(env="scml_oneshot_env", config=config)

    while True:
        print(trainer.train())