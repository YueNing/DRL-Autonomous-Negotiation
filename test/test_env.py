from drl_negotiation.env import NEnv, NegotiationEnv, DRLNegotiationEnv, MyNegotiationEnv
from drl_negotiation.game import MyDRLNegotiationGame, DRLNegotiationGame, NegotiationGame
from drl_negotiation.negotiator import MyDRLNegotiator, MyOpponentNegotiator
from drl_negotiation.utility_functions import ANegmaUtilityFunction, MyUtilityFunction
from drl_negotiation.utils import  generate_config, genearate_observation_space
# from ANLearning.lab_notebook.scml_negotiation.scml_game import Game

def test_negotiation_env():
    """
    Test NegotiationEnv, Default ANegma setting, using method based on reinforcement learning
    single issue
    """
    from negmas import Issue

    name = "test_n_env"
    config = generate_config()
    n_steps = 100

    # test acceptance strategy
    print("Begining test of the Acceptance strategy!")

    # game
    game = NegotiationGame(
        name="negotiation_game",
        game_type="DRLNegotiation",
        issues = config.get("issues"),
        competitors=[MyDRLNegotiator(
            name="my_drl_negotiator",
            ufun=MyUtilityFunction(weights=(-0.35,)),
            # ufun=ANegmaUtilityFunction(
            #     name="anegma_utility_function",
            #     max_t=config.get("max_t"),
            # ),
            init_proposal=False,
            # rp_range=config.get("rp_range"),
            # ip_range=config.get("ip_range"),
        ), MyOpponentNegotiator(
            name="my_opponent_negotiator",
            ufun= MyUtilityFunction(weights=(0.25, ))
        )],
        n_steps=n_steps
    )

    # env
    observation_space = genearate_observation_space(config)
    action_space = 3
    strategy = "ac_s"
    n_env = NegotiationEnv(
        name=name,
        game=game,
        strategy=strategy,
        observation_space=observation_space,
        action_space=action_space
    )
    # game.set_env(env=n_env)

    # test of basic attributes
    assert n_env.__str__() == f'The name of Env is {name}'
    
    # import pdb;pdb.set_trace()
    if game is None:
        assert n_env.game is None
    else:
        assert n_env.game.__dict__["_name"] == game.__dict__["_name"]

    init_obs = n_env.reset()
    
    for _ in range(n_steps):
        action = n_env.action_space.sample()
        obs, reward, done, info=n_env.step(action=action)
        if done:
            break

    print("Finish test of the Acceptance strategy!\n")

    # test offer strategy
    print("Begining test of the Offer/bidding strategy")

    # dynamic changing the attributes in order to switch the strategy
    strategy = "of_s"
    action_space = [[config.get("issues")[0].values[0],], [config.get("issues")[0].values[1],]]

    n_env.set_action_space(action_space=action_space)
    n_env.strategy = strategy

    # before learning need to reset the environment and reset the game
    n_env.reset()

    for _ in range(n_steps):
        action = n_env.action_space.sample()
        obs, reward, done, info = n_env.step(action=action)
        if done:
            break

    print("Finish test of the Offer/bidding strategy!")
    # test hybrid strategy

# def test_drl_negotiation_env():
#     """
#     Test DLRNegotiationEnv, based on the settings comes from scml Negotiation
#     """
#     name = "test_drl_negotiation_env"
#     drl_n_env = DRLNegotiationEnv(
#         name=name,
#     )
#     # import pdb;pdb.set_trace()
#     assert drl_n_env.__str__() == f'The name of Env is {name}'
#
#     # mainly test the step function inherit from gym.env
#     # design 1, [offer of my negotiator, time]
#     init_obs = drl_n_env.reset()
#     assert init_obs == [0 for _ in drl_n_env.game.format_issues[0]] + [0.0]
#
#     for _ in range(100):
#         action = drl_n_env.action_space.sample()
#         obs, reward, done, info = drl_n_env.step(action=action)
#         if done:
#             break




# def test_my_negotiation_env():
#     name = "TestMyNEnv"
#
#     my_n_env = MyNegotiationEnv(
#         name=name
#     )
#
#     # assert test used by pytest
#     assert my_n_env.__str__() == f'The name of Env is {name}'
#     assert isinstance(
#         my_n_env.game,
#         MyDRLNegotiationGame
#     )
#
#     # pdb Test Block
#     # import pdb
#     # pdb.set_trace()


if __name__ == "__main__":
    test_negotiation_env()
    # test_drl_negotiation_env()
    # test_my_negotiation_env()