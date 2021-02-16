import os
import sys
import pickle
from drl_negotiation.core.config.hyperparameters import *
from drl_negotiation.core.games.scml import TrainWorld
from drl_negotiation.third_party.negmas.negmas.helpers import get_class
from drl_negotiation.third_party.scml.src.scml.scml2020 import SCML2020World
from drl_negotiation.core.envs.normalized_env import NormalizedEnv


###########################################################
# env
###########################################################
def make_env(scenario_name,
             arglist=None,
             save_config=False,
             load_config=False,
             save_dir=None,
             load_dir=None,
             normalize=False,
             ):
    """

    Args:
        normalize: normalize wrapper
        scenario_name: the name of scenario, e.g. scml
        arglist:
        save_config: bool, save config or not
        load_config: bool, load config or not
        save_dir: pass into save dir, or use default dir which set up in hyperparameters.py
        load_dir: pass into load dir, or use default dir which set up in hyperparamters.py

    Returns:
        SCMLEnv, gym.env

    >>> isinstance(make_env('scml',
    ...         save_config=True,
    ...         load_config=True,
    ...         save_dir="/tmp/drl_negotiation/doctest/world.config",
    ...         load_dir="/tmp/drl_negotiation/doctest/world.config"
    ...     ), drl_negotiation.core.env.SCMLEnv)
    True
    """

    from drl_negotiation.core.envs.multi_agents_scml import SCMLEnv
    import drl_negotiation.core.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + '.py').Scenario()

    # create world/game
    if load_config:
        config = get_world_config(load_dir)
    else:
        config = None

    world = scenario.make_world(config)

    if save_config:
        save_dir = save_dir if save_dir is not None else SAVE_WORLD_CONFIG_DIR
        try:
            world.save_config(file_name=save_dir)
        except FileNotFoundError:
            path = '/'.join(save_dir.split("/")[:-1])
            os.makedirs(path)
            logging.info(f"Creates dirs, {path}")

            world.save_config(file_name=save_dir)
            logging.info(f"save {world} into {save_dir}")

    # create multi-agent supply chain management environment
    env = SCMLEnv(
        world,
        reset_callback=scenario.reset_world,
        reward_callback=scenario.reward,
        observation_callback=scenario.observation,
        info_callback=scenario.benchmark_data,
        done_callback=scenario.done,
        shared_viewer=False
    )

    if normalize:
        env = NormalizedEnv(env)

    logging.info(f"Make {env} successfully!")
    return env


def make_world(config=None):
    if config is None:
        agent_types = [get_class(agent_type, ) for agent_type in TRAINING_AGENT_TYPES_CONCURRENT]
        n_steps = N_STEPS
        world_configuration = SCML2020World.generate(
            agent_types=agent_types,
            n_steps=n_steps
        )
        world_configuration['negotiation_speed'] = NEGOTIATION_SPEED
    else:
        world_configuration = config

    world = TrainWorld(configuration=world_configuration)
    return world


def get_world_config(load_dir):
    try:
        load_dir = load_dir if load_dir is not None else LOAD_WORLD_CONFIG_DIR
        with open(load_dir + '.pkl', 'rb') as file:
            config = pickle.load(file)
            logging.info(f"load world config successfully from {load_dir}")
    except FileNotFoundError as e:
        logging.error(f"Error when Try to load the file from {load_dir}, "
                      f"please ensure world config file in the path {load_dir}")
        logging.debug(str(e))
        logging.info("will not load world config!")
        config = None
        sys.exit()

    return config


#####################################################################
# train
#####################################################################
import drl_negotiation
from drl_negotiation.core.modules.maddpg.policy import mlp_model


def get_trainers(env, num_adversaries=0, obs_shape_n=None, arglist=None):
    # TODO: train seller and buyer together, env.action_space?

    trainers = []
    model = mlp_model
    trainer = drl_negotiation.a2c.trainer.MADDPGAgentTrainer

    action_space = env.action_space

    # if not only_seller:
    #     obs_shape_n = obs_shape_n * 2
    #     action_space = action_space * 2
    #     assert len(obs_shape_n)==env.n * 2, "Error, length of obs_shape_n is not same as 2*policy agents"
    #     assert len(action_space)==len(obs_shape_n), "Error, length of act_space_n and obs_space_n are not equal!"

    # first set up the adversaries, default num_adversaries is 0
    for i in range(num_adversaries):
        trainers.append(trainer(
            env.agents[i].name.replace("@", '-') + "_seller", model, obs_shape_n, action_space, i, arglist,
            local_q_func=(arglist.adv_policy == 'ddpg')
        ))
        if not ONLY_SELLER:
            trainers.append(
                trainer(
                    env.agents[i].name.replace("@", '-') + "_buyer", model, obs_shape_n, action_space,
                    i + 1, arglist,
                    local_q_func=(arglist.adv_policy == 'ddpg')
                )
            )
    # if not only_seller:
    #     for i in range(num_adversaries):
    #         train.append(
    #             train(
    #                 env.agents[i].name.replace("@", '-')+"_buyer", model, obs_shape_n, action_space, i+ int(len(obs_shape_n) / 2), arglist,
    #                 local_q_func=(arglist.adv_policy == 'ddpg')
    #             )
    #         )

    # set up the good agent
    for i in range(num_adversaries, env.n):
        trainers.append(trainer(
            env.agents[i].name.replace("@", '-') + "_seller", model, obs_shape_n, action_space, i * 2, arglist,
            local_q_func=(arglist.good_policy == "ddpg")
        )
        )
        if not ONLY_SELLER:
            trainers.append(trainer(
                env.agents[i].name.replace("@", '-') + "_buyer", model, obs_shape_n, action_space,
                i * 2 + 1, arglist,
                local_q_func=(arglist.good_policy == 'ddpg')
            ))

    # if not only_seller:
    #     for i in range(num_adversaries, env.n):
    #         train.append(train(
    #             env.agents[i].name.replace("@", '-')+"_buyer", model, obs_shape_n, action_space, i+int(len(obs_shape_n) / 2), arglist,
    #             local_q_func=(arglist.good_policy == 'ddpg')
    #         ))

    return trainers


if __name__ == "__main__":
    import doctest
    doctest.testmod()
