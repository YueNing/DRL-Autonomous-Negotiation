import sys
import os
sys.path.append(r'/home/nauen/ANLearning/lab_notebook') 


def test_drl_negotiator():
    from mynegotiator import MyDRLNegotiator, DRLNegotiator
    from myutilityfunction import MyUtilityFunction
    from negmas import SAOMechanism, Issue, ResponseType, Outcome
    from scml_env import DRLNegotiationEnv, NEnv
    
    name = "drl_negotiator"
    issues = [
        Issue(values=10, name="quantity"), 
        Issue(values=100, name="delivery_time"), 
        Issue(values=100, name="unit_price")
    ]

    weights = (0, -0.5, -0.8)

    # set the utility function with MyUtilityFunction
    mechanism = SAOMechanism(
        issues=issues, n_steps=100
    )
    ufun = MyUtilityFunction(
        weights=weights,
        ami=mechanism.ami
    )
    
    drl_negotiator = DRLNegotiator(
        name=name,
        ufun=ufun
    )
    
    mechanism.add(
        drl_negotiator
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

    # Action: None, Env: None
    # env is None, so the respond is always ResponseType.REJECT_OFFER
    respond = drl_negotiator.respond(mechanism.state, offer[0])
    assert  type(respond) == ResponseType
    if respond == ResponseType.REJECT_OFFER:
        proposal_offer = drl_negotiator.propose(mechanism.state)
        if proposal_offer is not None:
            assert proposal_offer in drl_negotiator.ami.outcomes
    else:
        assert False, "Reponse must be ResponseType.Reject_OFFER!"

    # Action: None, Env: not None
    # Env will automatically set the issues, games and so on attributes.
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

    drl_negotiator.set_current_action(1)
    assert type(drl_negotiator.action) == ResponseType
    assert type(respond) == ResponseType, "Type of response must be ResponseType!"

    respond = drl_negotiator.respond(mechanism.state, offer[0])
    assert respond == drl_negotiator.action



if __name__ == "__main__":
    test_drl_negotiator()