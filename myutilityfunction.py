from typing import Optional, Type
from negmas import Action #  An action that an `Agent` can execute in a `World` through the `Simulator`
from negmas import (
                        Issue, 
                        UtilityFunction,
                        UtilityValue,
                        AgentMechanismInterface,
                        Outcome,
                        outcome_for,
                        outcome_as_tuple,
                    )
from negmas.generics import (
    ikeys,
    iget
)

import random
from typing import List, Optional, Type, Sequence

class MyUtilityFunction(UtilityFunction):
    r""" Model My utility function, linear utility function with discount factor based on relative time
    
    Info:
        issues = [
            Issue(qvalues, name="quantity"),
            Issue(tvalues, name="time"),
            Issue(uvalues, name="uvalues"),
        ]

    Doc Test:
        
        >>>ufun = MyUtilityFunction(issues=[Issue((10, 100), name="quantity"), Issue((0, 1), name="time"), Issue((0, 10), name="uvalues")])
    
    Args:
        weights: weights for combining `issue utilities`
        delta: discount rate
        factor: the factor used by delta, default is `relative time`
    
    Notes:

        The utility value is calculated as:

        .. math::

            u = \sum_{i=0}^{n_{outcomes}-1} {w_i * \omega_i}*delta**(t-1)

    """
    def __init__(
        self, 
        weights: Optional[Sequence[float]] = None,
        delta: Optional[float] = 0.6,
        factor: Optional[str] = 'relative_time',
        name: Optional[str] = None,
        reserved_value: UtilityValue = float("-inf"),
        ami: AgentMechanismInterface = None, 
        outcome_type: Optional[Type]= None,

) -> None:
        super().__init__(
            name=name, outcome_type=outcome_type, reserved_value=reserved_value, ami=ami
        )
        self.weights = weights
        self.delta = delta
        self.factor = factor
    
    def eval(self, offer: Optional[Outcome]) -> Optional[UtilityValue]:
        '''
        called by magic function __call__, calculate the utility value of inputed offer

        '''
        if offer is None:
            return self.reserved_value
        offer = outcome_for(offer, self.ami) if self.ami is not None else offer
        offer = outcome_as_tuple(offer)

        if self.ami:
            # the ami is not None, means, can use the factor defined by user!
            return self.delta**(getattr(self.ami.state, self.factor) - 1) * sum(w * v for w, v in zip(self.weights, offer))
        else:
            return sum(w * v for w, v in zip(self.weights, offer))
    
    def xml(self, issues: List[Issue]) -> str:
        output = ""
        keys = list(ikeys(issues))
        for i, k in enumerate(keys):
            issue_name = iget(issues, k).name
            output += f'<issue index="{i+1}" etype="discrete" type="discrete" vtype="discrete" name="{issue_name}">\n'
            vals = iget(issues, k).all
            for indx, u in enumerate(vals):
                output += (
                    f'    <item index="{indx+1}" value="{u}" evaluation="{u}" />\n'
                )
            output += "</issue>\n"
        for i, k in enumerate(keys):
            output += (
                f'<weight index="{i+1}" value="{iget(self.weights, k)}">\n</weight>\n'
            )
        return output

    def __str__(self):
        return f"w: {self.weights}, delta: {self.delta}, factor: {self.factor}"