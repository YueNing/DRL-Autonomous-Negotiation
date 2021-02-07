import os, ray
from ray import tune
from ray.tune import register_env, register_trainable
from ray.rllib.utils.test_utils import check_learning_achieved
from drl_negotiation.utils.utils import parse_args, make_env
from drl_negotiation.core.env import RaySCMLEnv
from drl_negotiation.a2c.policy import MADDPGTFPolicy

if __name__ == "__main__":
    args = parse_args()
    config = {}
    env_creator = lambda config: make_env(config.get("name", "scml_concurrent"),
                                          normalize= config.get("normalize", True))
    test_env = RaySCMLEnv(env_creator({}))
    obs_spaces = test_env.observation_spaces
    act_spaces = test_env.action_spaces
    agents = test_env.agents.tolist()

    register_env('scml_env', lambda config: RaySCMLEnv(env_creator(config)))

    if args.run == "contrib/MADDPG":
        config['learning_starts'] = 100
        config['env_config'] = {"name": "scml_concurrent", "normalize": True}
        config['multiagent'] = {"policies":{f"pol_{index}": (MADDPGTFPolicy, obs_spaces[space[0]], act_spaces[space[1]],
                                            {"agent_id": index, "gamma": 0.85})
                                           for index, space in enumerate(zip(obs_spaces, act_spaces))
                                           },
                                "policy_mapping_fn": lambda agent_id: f"pol_{agents.index(agent_id)}"}
        config['framework'] = "torch" if args.torch else "tf"
        config['num_gpus'] = int(os.environ.get("RLLIB_NUM_GPUS", "0"))
        group = False

    ray.init(address="auto")

    stop = {
        "episode_reward_mean": args.stop_reward,
        "timesteps_total": args.stop_timesteps
    }

    config = dict(config, **{
        "env": "scml_env"
    })

    results = tune.run(args.run, stop=stop, config=config, verbose=1, num_samples=4)

    if args.as_test:
        check_learning_achieved(results)

    ray.shutdown()
