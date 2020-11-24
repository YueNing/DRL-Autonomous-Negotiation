import sys
import os
sys.path.append(r'/home/nauen/PycharmProjects/tn_source_code')

from train import train_negotiation
from scml_env import DRLNegotiationEnv, NEnv
from scml_env import NegotiationEnv
from utils import generate_config, genearate_observation_space, generate_action_space
from scml_game import NegotiationGame
from mynegotiator import MyDRLNegotiator, MyOpponentNegotiator
from myutilityfunction import MyUtilityFunction



def test_train_negotiation():
    #  train based on NegotiationEnv 
    
    # env = DRLNegotiationEnv(
    #     name="my_negotiation_env"
    # )
    n_issues = 3
    config = generate_config(n_issues=n_issues)

    # game
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
    # assert isinstance(env, NEnv)

    # game.set_env(env["ac_s"])
    # model = "DQN"
    # done, _ = train_negotiation(plot=plot, model=model, env=env["ac_s"], monitor=False)

    game.set_env(env["of_s"])
    model = "PPO1"
    done, _ = train_negotiation(plot=plot, model=model, env=env["of_s"], monitor=False)

    assert done,  f'train false by the model {model}'

    # model = "PPO1"
    # done, _ = train_negotiation(plot=plot, model=model, env=env)
    # assert done,  f'train false by the model {model}'
    #
    # model = "Test"
    # done, _ = train_negotiation(plot=plot, model=model, env=env)
    # assert done,  f'train false by the model {model}'


if __name__ == '__main__':
    test_train_negotiation()