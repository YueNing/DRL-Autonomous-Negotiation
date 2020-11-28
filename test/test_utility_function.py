from negmas import AgentMechanismInterface, SAOMechanism, Issue, AspirationNegotiator, MappingUtilityFunction
import random # for generating random ufuns

def test_my_utility_function():
    from drl_negotiation.utility_functions import MyUtilityFunction

    issues = [
        Issue(values=10, name="quantity"), 
        Issue(values=100, name="delivery_time"), 
        Issue(values=100, name="unit_price")
    ]
    
    # do not have the ami
    weights = [(0, 0.25, 1), (0, -0.5, -0.8)]
    offer = [10 , 12, 3]

    for _ in weights:
        ufun = MyUtilityFunction(
            weights=_
        )
        assert ufun(offer) == sum(w * v for w, v in zip(_, offer))
        assert ufun.issues is None

    # give the ami
    mechanism = SAOMechanism(
        issues=issues, n_steps=100
    )
    ami = mechanism.ami
    for _ in weights:
        ufun = MyUtilityFunction(
            weights=_, 
            ami=ami
        )
        offers = Issue.sample(issues=issues, n_outcomes=1, astype=tuple)
        assert ufun(offers[0]) == sum(w * v for w, v in zip(_, offers[0]))
        assert ufun.issues == issues

        # Test the utility of outcomes
        new_offer = ufun.outcome_with_utility(rng=(ufun(offers[0])+0.1, ufun(offers[0]) + 0.5), outcomes=ufun.ami.outcomes)
        if new_offer is not None:
            result = ufun.is_better(first=new_offer, second=offers[0])
            assert result
        else:
            pass
        # import pdb;pdb.set_trace()
    # assert ufun((1, 0, 1))
    # import pdb;pdb.set_trace()

def test_anegma_utility_function():
    from drl_negotiation.utility_functions import ANegmaUtilityFunction
    from negmas import Issue

    issues = [Issue((300, 550))]

    ip = Issue.sample(issues=issues, n_outcomes=1, astype=tuple)[0][0]
    rp = Issue.sample(issues=issues, n_outcomes=1, astype=tuple)[0][0]
    max_t = 0.8
    delta = 0.6

    mechanism = SAOMechanism(
        issues=issues, n_steps=100
    )
    ami = mechanism.ami

    anegma_utility_function = ANegmaUtilityFunction(
        delta=0.6,
        rp=rp,
        ip=ip,
        max_t=max_t,
        ami=ami
    )
    negotiators = [AspirationNegotiator(name=f'a{_}') for _ in range(5)]
    for negotiator in negotiators:
        mechanism.add(negotiator, ufun=MappingUtilityFunction(lambda x: random.random() * x[0]))

    for _ in range(10):
        offers = Issue.sample(issues=issues, n_outcomes=1, astype=tuple)
        mechanism.step()
        ufun = anegma_utility_function(offers[0])
        result = ((float(rp) - float(offers[0][0])) / (float(rp) - float(ip))) * (float(getattr(ami.state, anegma_utility_function.factor)+1.0) / float(max_t)) ** delta
        assert ufun ==result
        print(ufun)

if __name__ == '__main__':
    test_my_utility_function()
    test_anegma_utility_function()