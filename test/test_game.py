import sys
import os
sys.path.append(r'/home/naodong/giteethesiskit/ANLearning/lab_notebook') 

from scml_negotiation.scml_game import MyDRLNegotiationGame
import numpy as np

def test_my_drl_negotiation_game():
    name = "test_my_drl_negotiation_game"
    my_drl_negotiation_game = MyDRLNegotiationGame()

    # assert
    issues = my_drl_negotiation_game.issues
    format_issues = my_drl_negotiation_game.format_issues
    checked_result_format_issues = [
        [],
        []
    ]

    for _ in issues:
        checked_result_format_issues[0].append(_.values[0])
        checked_result_format_issues[1].append(_.values[1])

    assert my_drl_negotiation_game.name == name
    assert (np.array(format_issues) == np.array(checked_result_format_issues)).all()
    
    # pdb
    # import pdb;pdb.set_trace()