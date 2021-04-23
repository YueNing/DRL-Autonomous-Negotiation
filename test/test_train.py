import os
import sys

path = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
sys.path.append(path)

from drl_negotiation.train import train_negotiation
from drl_negotiation.env import NegotiationEnv
from drl_negotiation.utils import generate_config, genearate_observation_space, generate_action_space
from drl_negotiation.game import NegotiationGame
from drl_negotiation.negotiator import MyDRLNegotiator, MyOpponentNegotiator
from drl_negotiation.utility_functions import MyUtilityFunction


def test_train_negotiation():
    n_issues = 1
    config = generate_config(n_issues=n_issues)

    game = NegotiationGame(
        name="negotiation_game",
        game_type="DRLNegotiation",
        issues=config.get("issues"),
        competitors=[MyDRLNegotiator(
            name="my_drl_negotiator",
            ufun=MyUtilityFunction(weights=config.get("weights")[0]),
            # ufun=ANegmaUtilityFunction(
            #     name="anegma_utility_function",
            #     max_t=config.get("max_t"),
            # ),
            init_proposal=False,
            # rp_range=config.get("rp_range"),
            # ip_range=config.get("ip_range"),
        ), MyOpponentNegotiator(
            name="my_opponent_negotiator",
            ufun=MyUtilityFunction(weights=config.get("weights")[1])
        )],
        n_steps=config.get("n_steps")
    )

    env = {
        "ac_s": NegotiationEnv(
            name="negotiation_env_ac_s",
            strategy="ac_s",
            game=game,
            observation_space=genearate_observation_space(config),
            action_space=3
        ),
        "of_s": NegotiationEnv(
            name="negotiation_env_of_s",
            strategy="of_s",
            game=game,
            observation_space=genearate_observation_space(config),
            # action_space=[[config.get("issues")[0].values[0], ], [config.get("issues")[0].values[1], ]]
            # action_space=[[-1, ], [1, ]]
            action_space = generate_action_space(config)

    ),
        "hybrid": None,
    }

    plot = True
    train_steps = 100000
    log_dir = "acceptance_strategy"
    # acceptance strategy
    game.set_env(env["ac_s"])
    model = "DQN"
    done, _ = train_negotiation(plot=plot, model=model, env=env["ac_s"], monitor=False, num_timesteps=train_steps,
                                LOGDIR=log_dir)
    assert done,  f'train false by the model {model}'

    game.set_env(env["ac_s"])
    model = "ACER"
    done, _ = train_negotiation(plot=plot, model=model, env=env["ac_s"], monitor=False, num_timesteps=train_steps,
                                LOGDIR=log_dir)
    assert done,  f'train false by the model {model}'

    game.set_env(env["ac_s"])
    model = "PPO1"
    done, _ = train_negotiation(plot=plot, model=model, env=env["ac_s"], monitor=False, num_timesteps=train_steps,
                                LOGDIR=log_dir)
    assert done,  f'train false by the model {model}'

    game.set_env(env["ac_s"])
    model = "PPO2"
    done, _ = train_negotiation(plot=plot, model=model, env=env["ac_s"], monitor=False, num_timesteps=train_steps,
                                LOGDIR=log_dir)
    assert done, f'train false by the model {model}'

    game.set_env(env["ac_s"])
    model = "A2C"
    done, _ = train_negotiation(plot=plot, model=model, env=env["ac_s"], monitor=False, num_timesteps=train_steps,
                                LOGDIR=log_dir)
    assert done, f'train false by the model {model}'

    log_dir = "offer_strategy"
    # offer strategy
    game.set_env(env["of_s"])
    model = "DDPG"
    done, _ = train_negotiation(plot=plot, model=model, env=env["of_s"], monitor=False, num_timesteps=train_steps,
                                LOGDIR=log_dir)
    assert done, f'train false by the model {model}'

    game.set_env(env["of_s"])
    model = "PPO1"
    done, _ = train_negotiation(plot=plot, model=model, env=env["of_s"], monitor=False, num_timesteps=train_steps,
                                LOGDIR=log_dir)
    assert done,  f'train false by the model {model}'

    game.set_env(env["of_s"])
    model = "PPO2"
    done, _ = train_negotiation(plot=plot, model=model, env=env["of_s"], monitor=False, num_timesteps=train_steps,
                                LOGDIR=log_dir)
    assert done, f'train false by the model {model}'

    game.set_env(env["of_s"])
    model = "A2C"
    done, _ = train_negotiation(plot=plot, model=model, env=env["of_s"], monitor=False, num_timesteps=train_steps,
                                LOGDIR=log_dir)
    assert done, f'train false by the model {model}'


if __name__ == '__main__':
    test_train_negotiation()