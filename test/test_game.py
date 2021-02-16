import numpy as np
from drl_negotiation.core.games.bilateral_game import MyDRLNegotiationGame, DRLNegotiationGame


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

def test_drl_negotiation_game():
    from negmas import Mechanism
    # issues: None, competitors: None
    name = "test_drl_negotiation_game"
    game = DRLNegotiationGame(
        name=name
    )

    # test attributes and function in session
    # create session and add competitors, in negotiations game means adding the negotiators into session mechanism
    game.init_game()
    assert isinstance(game.session, Mechanism)
    assert len(game.session.negotiators) == len(game.competitors)

    # run game
    # result = game.run()
    # assert isinstance(result, MechanismState)

    # run game step by step
    # inspect the action and reward history
    from gym.spaces import Discrete
    action_space = Discrete(5)
    action_history = []
    reward_history = []
    result = None
    for _ in range(game.n_steps):
        result = game.step(action=action_space.sample())
        action_history.append(game.competitors[0].action)
        reward_history.append(result)
        obs = game.get_observation()
        if not game.get_life():
            break
    assert result is not None, "get the reward go one step forward!"


if __name__ == '__main__':
    test_my_drl_negotiation_game()
    test_drl_negotiation_game()