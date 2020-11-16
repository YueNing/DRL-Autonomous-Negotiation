import sys
import os
sys.path.append(r'/home/nauen/ANLearning/lab_notebook') 

from scml_negotiation.scml_env import NEnv, NegotiationEnv, DRLNegotiationEnv, MyNegotiationEnv
from scml_negotiation.scml_game import MyDRLNegotiationGame, DRLNegotiationGame, NegotiationGame
from scml_negotiation.mynegotiator import MyDRLNegotiator, MyOpponentNegotiato
# from ANLearning.lab_notebook.scml_negotiation.scml_game import Game

def test_negotiation_env():
    """
    Test NegotiationEnv, Default ANegma setting
    """
    from negmas import Issue

    name = "TestNEnv"
    
    # game
    game = NegotiationGame(
        name="negotiation_game",
        issues = [Issue((300, 550))],
        competitors=[MyDRLNegotiator(), MyOpponentNegotiator()]
    )

    # env
    n_env = NegotiationEnv(
        name=name,
        game=game,
        observation_space=[[300, 0, 300, 500], [550, 210, 350, 550]],
        action_space=5
    )
    
    assert n_env.__str__() == f'The name of Env is {name}'
    
    # import pdb;pdb.set_trace()
    if game is None:
        assert n_env.game is None
    else:
        assert n_env.game.__dict__["_name"] == game.__dict__["_name"]

    # assert type(n_env.get_obs()) == list
    # import pdb;pdb.set_trace()

def test_dlr_negotiation_env():
    """
    Test DLRNegotiationEnv, based on the settings comes from scml Negotiation
    """
    name = "TestDLRNEnv"
    dlr_n_env = DRLNegotiationEnv(
        name=name,
    )
    # import pdb;pdb.set_trace()
    assert dlr_n_env.__str__() == f'The name of Env is {name}'

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
    # test_dlr_negotiation_env() 
    test_my_negotiation_env()