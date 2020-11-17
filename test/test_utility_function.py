import sys
import os
sys.path.append(r'/home/nauen/PycharmProjects/tn_source_code')

def test_my_utility_function():
    from myutilityfunction import MyUtilityFunction
    from negmas import AgentMechanismInterface, SAOMechanism, Issue
    
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
        assert ufun(offers[0]) == ufun.delta**(getattr(ami.state, ufun.factor) - 1) * sum(w * v for w, v in zip(_, offers[0]))
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

if __name__ == '__main__':
    test_my_utility_function()