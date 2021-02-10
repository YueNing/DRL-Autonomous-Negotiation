from drl_negotiation.a2c.a2c import MADDPGModel
from drl_negotiation.core.hyperparameters import *
from drl_negotiation.utils.utils import make_env
from drl_negotiation.utils.utils import logging_setup
from drl_negotiation.utils.plots import show_agent_rewards, show_ep_rewards, cumulative_reward, multi_layer_charts
import ray
import logging

logging_setup(logging.INFO)


def test_main():

    def _main():
        env = make_env('scml_concurrent',
                       save_config=SAVE_WORLD_CONFIG,
                       load_config=LOAD_WORLD_CONFIG,
                       save_dir=SAVE_WORLD_CONFIG_DIR,
                       load_dir=LOAD_WORLD_CONFIG_DIR,
                       normalize=True
                       )

        model = MADDPGModel(env=env, verbose=0, restore=RESTORE)
        # final_ep_rewards, final_ep_extra_rewards, agent_rewards, extra_agent_rewards, episode_rewards,
        # episode_extra_rewards, _ = model.learn(train_episodes=10)

        model_result: "ModelResult" = model.learn(train_episodes=TRAIN_EPISODES)

        fig_ep_rewards = show_ep_rewards(model_result.total_final_ep_rewards, model, extra=True)
        fig_agent_rewards = show_agent_rewards(model_result.total_final_ep_ag_rewards, model, extra=True)
        fig_cumulative_reward = cumulative_reward(model_result.total_episode_rewards, model, extra=True)

        if EVALUATION:
            obs_n, _ = env.reset()
            for i in range(MAX_EPISODE_LEN):
                action_n = model.predict(obs_n, train=False)
                es = env.step(action_n)
                obs_n = es.observation
                render_result = env.render("ascii")
                print(render_result)
            env.close()

        return {
            "ep_mean_rewards": fig_ep_rewards,
            "agent_mean_rewards": fig_agent_rewards,
            "cumulative_rewards": fig_cumulative_reward
        }
    result = [_main()]
    multi_layer_charts(result)

def main(num_samples=1):

    @ray.remote
    def _main(count):
        env = make_env('scml_concurrent',
                       save_config=SAVE_WORLD_CONFIG,
                       load_config=LOAD_WORLD_CONFIG,
                       save_dir=SAVE_WORLD_CONFIG_DIR,
                       load_dir=LOAD_WORLD_CONFIG_DIR,
                       normalize=True
                       )

        model = MADDPGModel(env=env, verbose=0, restore=RESTORE)
        # final_ep_rewards, final_ep_extra_rewards, agent_rewards, extra_agent_rewards, episode_rewards,
        # episode_extra_rewards, _ = model.learn(train_episodes=10)

        model_result: "ModelResult" = model.learn(train_episodes=TRAIN_EPISODES)

        fig_ep_rewards = show_ep_rewards(model_result.total_final_ep_rewards, model, extra=True, count=count)
        fig_agent_rewards = show_agent_rewards(model_result.total_final_ep_ag_rewards, model, extra=True, count=count)
        fig_cumulative_reward = cumulative_reward(model_result.total_episode_rewards, model, extra=True, count=count)

        if EVALUATION:
            obs_n, _ = env.reset()
            for i in range(MAX_EPISODE_LEN):
                action_n = model.predict(obs_n, train=False)
                es = env.step(action_n)
                obs_n = es.observation
                render_result = env.render("ascii")
                print(render_result)
            env.close()

        return {
            "ep_mean_rewards": fig_ep_rewards,
            "agent_mean_rewards": fig_agent_rewards,
            "cumulative_rewards": fig_cumulative_reward
        }
        # return [fig_ep_rewards, fig_agent_rewards, fig_cumulative_reward]

    futures = [_main.remote(_) for _ in range(num_samples)]
    results = ray.get(futures)
    multi_layer_charts(results)
    # print(results)


if __name__ == '__main__':
    debug = True
    if debug:
        test_main()
    else:
        ray.init(address="auto")
        main(num_samples=8)
