from drl_negotiation.core.train.runner import Runner
from drl_negotiation.core.envs.multi_agents_negotiation import MultiNegotiationSCM
from drl_negotiation.core.utils.common import get_common_args, get_mixer_args
import drl_negotiation.core.scenarios as scenarios


if __name__ == '__main__':
    args = get_common_args()
    args = get_mixer_args(args)

    scenario = scenarios.load("scml_oneshot"+".py").Scenario()
    env = MultiNegotiationSCM(
        world=scenario.make_world(),
        scenario=scenario,
        seed=10
    )

    env_info = env.get_env_info()
    args.n_actions = env_info["n_actions"]
    args.n_agents = env_info["n_agents"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]
    args.episode_limit = env_info["episode_limit"]

    runner = Runner(env, args)
    if not args.evaluate:
        runner.run(0)
    else:
        episode_reward = runner.evaluate()
        print(f"episode_reward is {episode_reward}")
    env.close()