import sys
import os
sys.path.append(r'/home/nauen/PycharmProjects/tn_source_code')

from scml_env import NEnv, NegotiationEnv, DRLNegotiationEnv, MyNegotiationEnv
from scml_game import MyDRLNegotiationGame, DRLNegotiationGame, NegotiationGame
from mynegotiator import MyDRLNegotiator, MyOpponentNegotiator
# from ANLearning.lab_notebook.scml_negotiation.scml_game import Game

def test_negotiation_env():
    """
    Test NegotiationEnv, Default ANegma setting, using method based on reinforcement learning
    """
    from negmas import Issue

    name = "test_n_env"
    
    # game
    game = NegotiationGame(
        name="negotiation_game",
        issues = [Issue((300, 550))],
        competitors=[MyDRLNegotiator(), MyOpponentNegotiator()]
    )

    # test acceptance strategy
    # env
    n_env = NegotiationEnv(
        name=name,
        game=game,
        strategy="ac_s",
        observation_space=[[300, 0, 300, 500], [550, 210, 350, 550]],
        action_space=5
    )
    game.set_env(env=n_env)


    assert n_env.__str__() == f'The name of Env is {name}'
    
    # import pdb;pdb.set_trace()
    if game is None:
        assert n_env.game is None
    else:
        assert n_env.game.__dict__["_name"] == game.__dict__["_name"]

    print("Finish test of the Acceptance strategy!")
    # test offer strategy


    # test hybrid strategy

def test_drl_negotiation_env():
    """
    Test DLRNegotiationEnv, based on the settings comes from scml Negotiation
    """
    name = "test_drl_negotiation_env"
    drl_n_env = DRLNegotiationEnv(
        name=name,
    )
    # import pdb;pdb.set_trace()
    assert drl_n_env.__str__() == f'The name of Env is {name}'

    # mainly test the step function inherit from gym.env
    # design 1, [offer of my negotiator, time]
    init_obs = drl_n_env.reset()
    assert init_obs == [0 for _ in drl_n_env.game.format_issues[0]] + [0.0]

    for _ in range(100):
        action = drl_n_env.action_space.sample()
        obs, reward, done, info = drl_n_env.step(action=action)
        if done:
            break




def test_my_negotiation_env():
    name = "TestMyNEnv"
    
    my_n_env = MyNegotiationEnv(
        name=name
    )

    # assert test used by pytest
    assert my_n_env.__str__() == f'The name of Env is {name}'
    assert isinstance(
        my_n_env.game,
        MyDRLNegotiationGame
    )
    
    # pdb Test Block 
    # import pdb
    # pdb.set_trace()


if __name__ == "__main__":
    test_negotiation_env()
    test_drl_negotiation_env()
    test_my_negotiation_env()