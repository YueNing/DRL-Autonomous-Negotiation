def test_drl_negotiator():
    from drl_negotiation.agents.negotiator import MyDRLNegotiator
    from drl_negotiation.core.utils.utility_functions import MyUtilityFunction
    from drl_negotiation.third_party.negmas.negmas import SAOMechanism, Issue, ResponseType
    from drl_negotiation.core.envs.bilateral_negotiation import DRLNegotiationEnv, NEnv
    
    name = "drl_negotiator"
    issues = [
        Issue(values=10, name="quantity"), 
        Issue(values=100, name="delivery_time"), 
        Issue(values=100, name="unit_price")
    ]
    # buyer
    weights = (0, -0.5, -0.8)

    # set the utility function with MyUtilityFunction
    mechanism = SAOMechanism(
        issues=issues, n_steps=100
    )
    ufun = MyUtilityFunction(
        weights=weights,
    )
    
    drl_negotiator = MyDRLNegotiator(
        name=name,
        ufun=ufun
    )
    
    mechanism.add(
        drl_negotiator
    )

    drl_negotiator_two = MyDRLNegotiator(
        name=name+"_two"
    )

    mechanism.add(
        drl_negotiator_two, ufun=MyUtilityFunction(
            weights=(0, -0.2, -0.6)
        )
    )
    assert drl_negotiator.name == name
    assert drl_negotiator.ufun == ufun

    # Test the basic attributes and functions
    # None
    assert  drl_negotiator.env is None
    assert  drl_negotiator.action is None

    #TODO: initial value and reserved value

    # Test the respond and propose of negotiator
    offer = Issue.sample(issues, 1)

    # Test acceptance strategy
    # Action: None, Env: None, Train: True
    # env is None, so the respond is always ResponseType.REJECT_OFFER
    drl_negotiator.train = True
    respond = drl_negotiator.respond(mechanism.state, offer[0])
    assert  type(respond) == ResponseType
    if respond == ResponseType.REJECT_OFFER:
        proposal_offer = drl_negotiator.propose(mechanism.state)
        if proposal_offer is not None:
            assert proposal_offer in drl_negotiator.ami.outcomes
    else:
        assert False, "Reponse must be ResponseType.Reject_OFFER!"

    # Action: None, Env: not None, Train: True
    # Env will automatically set the issues, games and so on attributes.
    drl_negotiator.train = True
    name = "drl_negotiation_env"
    n_env = DRLNegotiationEnv(
        name=name
    )

    drl_negotiator.reset(env=n_env)
    assert isinstance(drl_negotiator.env, NEnv)
    respond = drl_negotiator.respond(mechanism.state, offer[0])
    assert type(respond) == ResponseType, "Type of response must be ResponseType!"

    # Action: not None, Env: not None
    drl_negotiator.reset(env=n_env)
    action = drl_negotiator.env.action_space.sample()

    drl_negotiator.set_current_action(action)
    assert type(drl_negotiator.action) == ResponseType
    assert type(respond) == ResponseType, "Type of response must be ResponseType!"

    respond = drl_negotiator.respond(mechanism.state, offer[0])
    assert respond == drl_negotiator.action

    # Test Offer/bidding strategy


def test_opponent_negotiator():
    from drl_negotiation.agents.negotiator import MyOpponentNegotiator
    from negmas import Issue, SAOMechanism
    from drl_negotiation.core.utils.utility_functions import MyUtilityFunction

    name = "test_opponent_negotiator"
    # seller
    weights = (0, 0.25, 1)
    issues = [
        Issue(values=10, name="quantity"),
        Issue(values=100, name="delivery_time"),
        Issue(values=100, name="unit_price")
    ]
    mechanism = SAOMechanism(
        issues=issues, n_steps=100
    )
    ufun = MyUtilityFunction(
        weights=weights,
        ami=mechanism.ami
    )
    opponent_negotiator = MyOpponentNegotiator(
        name=name,
        ufun=ufun
    )
    mechanism.add(opponent_negotiator)

if __name__ == "__main__":
    test_drl_negotiator()
    test_opponent_negotiator()