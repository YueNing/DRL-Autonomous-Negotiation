#####################################################################
# trainer
#####################################################################
import drl_negotiation
from drl_negotiation.a2c.policy import mlp_model
from drl_negotiation.core.hyperparameters import *

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
    #         trainers.append(
    #             trainer(
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
    #         trainers.append(trainer(
    #             env.agents[i].name.replace("@", '-')+"_buyer", model, obs_shape_n, action_space, i+int(len(obs_shape_n) / 2), arglist,
    #             local_q_func=(arglist.good_policy == 'ddpg')
    #         ))

    return trainers