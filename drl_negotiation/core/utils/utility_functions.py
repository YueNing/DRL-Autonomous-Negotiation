import random
from negmas import (
                        Issue, 
                        UtilityFunction,
                        AgentMechanismInterface,
                        Outcome,
                        outcome_for,
                        outcome_as_tuple,
                        MappingUtilityFunction,
                    )
from negmas.generics import (
    ikeys,
    iget
)
from negmas.utilities import UtilityValue
from typing import List, Optional, Type, Sequence, Union, Tuple


class ANegmaUtilityFunction(UtilityFunction):
    """
        Idea comes from ANegma, single issue
    """
    def __init__(
            self,
            name = None,
            delta: Optional[float] = 0.6,
            rp: Union[Tuple, None] = None,
            ip=None,
            max_t=None,
            factor="step",
            reserved_value: UtilityValue=float("-inf"),
            ami= None
    ):
        """

        Args:
            name:
            delta: the hyper parameter of influence of time for utility
            rp: reserved proposal
            ip: intial proposal, set by user when instance utility function or set it by negotiator
            max_t: the maximum end time for this negotiator
            factor: default is "relative time" when max_t is also relative time,
                    absolute time when max_t is also absolute time.
            reserved_value: when offer is none, return reserved value
            ami: agent mechanism interface
        """
        super(ANegmaUtilityFunction, self).__init__(
            name=name,
            ami=ami,
            reserved_value=reserved_value
        )

        self.delta = delta
        self.rp = rp
        self.ip = ip
        self.max_t = max_t
        self.factor = factor

    def eval(self, offer: "Outcome") -> UtilityValue:
        if self.ami:
            return ((float(self.rp) - float(offer[0])) / (float(self.rp) - float(self.ip))) * \
                    ((getattr(self.ami.state, self.factor)+1.0) / float(self.max_t)) ** self.delta
        return self.reserved_value

    def xml(self, issues: List[Issue]) -> str:
        pass
    
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
        ip = None,
        rp = None,

) -> None:
        super().__init__(
            name=name, outcome_type=outcome_type, reserved_value=reserved_value, ami=ami
        )
        self.weights = weights
        self.delta = delta
        self.factor = factor
        self.ip = ip
        self.rp = rp
    
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
            # return self.delta**(getattr(self.ami.state, self.factor) - 1) * sum(w * v for w, v in zip(self.weights, offer))
            return sum(w * v for w, v in zip(self.weights, offer))
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

MyOpponentUtilityFunction = MappingUtilityFunction(lambda x: random.random() * x[0])