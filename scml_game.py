from abc import ABC, abstractmethod, abstractproperty
import gym 
from negmas import (
                        Mechanism, 
                        SAOMechanism, 
                        Issue, 
                        Agent, 
                        MechanismState, 
                        AspirationNegotiator,
                        Negotiator,
                        ResponseType
                    )
from typing import Optional, Union, Dict, List
from myagent import MyComponentsBasedAgent
from mynegotiator import MyDRLNegotiator, MyOpponentNegotiator, DRLNegotiator
from myutilityfunction import  MyUtilityFunction
from scml.scml2020.agents import DecentralizingAgent, BuyCheapSellExpensiveAgent
from scml.scml2020 import anac2020_std, anac2020_collusion
from dataclasses import dataclass, field, fields

# extra reward for agreement when negotiator get a agreement
EXTRA_REWARD = 0.1

__all__ = [
    "Game",
    "NegotiationGame",
    "SCMLGame",
    "DRLGameMixIn",
    "DRLNegotiationGame",
    "DRLSCMLGame",
    "MyDRLNegotiationGame",
    "MyDRLSCMLGame",

]

@dataclass
class AgentGameInterface():
    pass

####################################################################################################
# For Game
#
#
####################################################################################################
class Game(ABC):
    '''
    Model the Game

    Attribute: 
        name:
        game_type: Negotiation, SCML, DRLNegotiation, DRLSCML
        session:  
        session_data: 
        issues: 
        n_steps:
        reset: 
    '''
    def __init__(
        self,
        name:str = None,
        game_type: str = 'Negotiation',
        session: Union[Mechanism, anac2020_std, anac2020_collusion, None] = SAOMechanism,
        session_data: Optional[Dict] = None,
        issues: List[Issue] = None,
        n_steps: int = 10, 
        competitors: Optional[List[Negotiator]] = None,
        env = None,
        reset: bool = True,
    ):
        self._name = name
        self._game_type = game_type
        self._session = session
        self._session_data = session_data
        self._issues = issues
        self._n_steps = n_steps
        self._competitors = competitors
        self._env = env

        # # TODO: set the agent game interface
        # self.agi = AgentGameInterface(
        # )
        # self.agi._env = self

        if reset:
            self.reset()
    

    def __str__(self):
        return f'The name of Game is {self.name}'

    # def add(self):
    #     pass
    @property
    def env(self):
        return self._env

    @env.setter
    def env(self, env):
        self._env = env

    def set_env(self, env):
        self._env = env
        for _ in self.competitors:
            _.set_env(env=self.env)

    def add_competitors(self, competitors: Optional[List[Agent]]):
        if self.game_type == "Negotiation" or self.game_type == "DRLNegotiation":
            for _ in competitors:
                if _.has_ufun:
                    self.add_competitor(_)
                else:
                    raise ValueError("Please set the utility function!")

    @abstractmethod
    def add_competitor(self, competitor: Union[MyDRLNegotiator, MyOpponentNegotiator]):
        raise  NotImplementedError("")

    def init_game(self):
        # check the inputed game type valid or not
        self.check()

        # reset the competitors
        for _ in self.competitors:
            _.reset()

        # create a session, negotiation mechanism or scml anac
        self._session = self.create_session()
        self.add_competitors(competitors=self.competitors)

        # init all argumente
        self._step = 0

    def check(self):
        assert (
            self.game_type == 'Negotiation' or \
                self.game_type == 'SCML' or \
                    self.game_type == 'DRLNegotiation' or \
                        self.game_type == 'DRLSCML' 
        
        ), "the game_type is illegal!" 

    @property
    def game_type(self):
        return self._game_type
    
    @property
    def format_issues(self):
        '''
            Return the issues that used by SAOMechanism 
            in simple designed negmas negotiation or 
            in scml negotiation!
        '''
        low = []
        high = []
        for _ in self._issues:
            low.append(_.values[0])
            high.append(_.values[1])
        return low, high
    
    @property
    def issues(self):
        return self._issues

    @property
    def session_data(self):
        return self._session_data
    
    @property
    def session(self):
        return self._session
    
    @property
    def n_steps(self):
        return self._n_steps
    
    @property
    def ami(self):
        return self.session.ami
    
    @property
    def time(self):
        '''
        Return the time
        '''
        return self.session.relative_time

    @property
    def name(self):
        return self._name
    
    @property
    def competitors(self):
        return self._competitors

    def set_session_data(self, data: Optional[Dict] = None):
        self._session_data = data if data is not None else {}

    def run(self) -> MechanismState:
        return self._session.run()
    
    def step(self, action: "gym.spaces.Discrete.sample()" = None) -> Optional[MechanismState]:
        '''
        Run the game step by step, run the game one step forward, 
        and return the reward or state base on the game type
        '''
        
        self.check()

        return self.step_forward(action=action, competitor=self.competitors[0])

    @abstractmethod       
    def step_forward(self, action=None, competitor: Optional[DRLNegotiator] = None):
        raise NotImplementedError(" ")

    def create_session(self)-> Union[Mechanism, anac2020_std, anac2020_collusion]:

        # self.check()
        if self.game_type == 'Negotiation' or self.game_type == 'DRLNegotiation':
            if self.session_data is None:
                self.set_session_data(data={})
            
            if self._issues is not None:
                return SAOMechanism(issues=self.issues, n_steps=self.n_steps, **self.session_data)
            else:
                # Default issues
                return SAOMechanism(issues=[Issue(300, 550)], n_steps=self.n_steps, **self.session_data)
        
        if self.game_type == 'SCML' or self.game_type == 'DRLSCML':
            if self.session_data is None:
                self.session_data = {}

            # TODO: SCML session
    
    @classmethod
    def load_session(cls, session_data: dict = None):
        pass

    def reset(self):
        self.init_game()
    
    def seed(self, np_random=None):
        
        self._np_random = np_random
        
        # TODO: set the np random, 
        
    
    
    @abstractmethod
    def get_life(self):
        raise NotImplementedError('')

    @abstractmethod
    def get_state(self):
        raise NotImplementedError('')




class DRLGameMixIn:

    def get_observation(self):
        """
        Returns:
            obs: relates to the design of observation space
            1. [offer of my negotiator, time], obs[0]
            2. [offer of all negotiators, time], obs
            3. ?
        """
        obs = []

        assert self.competitors is not None, "The competitors is None!, please firstly let the competitors join into the game!"

        for _ in self.competitors:
            obs.append(_.get_obs())

        return obs[0]



####################################################################################################
# For Negotiation
#
#
####################################################################################################

class NegotiationGame(DRLGameMixIn, Game):
    """
    Game designed for ANegma
    """
    def __init__(
        self, 
        name: str = 'negotiation_game',
        game_type: str = 'Negotiation',
        issues: Optional[List[Issue]] = None, 
        n_steps: int = 10, 
        competitors: Optional[List[Negotiator]] = None,
        env = None
    ):
        # Default ANegma Issue
        if issues is None:
            issues = [Issue((300, 550))]

        # Default competitors
        if competitors is None:
            competitors = [
                MyDRLNegotiator(is_seller=False),
                MyOpponentNegotiator(is_seller=True)
            ]

        super().__init__(
            name=name,
            game_type=game_type,
            issues=issues,
            n_steps=n_steps,
            competitors=competitors,
            env = env
        )


    def add_competitor(self, competitor: Union[MyDRLNegotiator, MyOpponentNegotiator]):
        competitor.reset(env=self.env)

        self.session.add(competitor)
        # competitor.set_env(env=self.env)


    def step_forward(self, action=None, competitor: Optional[DRLNegotiator] = None):
        
        if self.game_type == 'Negotiation':
            # Normal game types 
            state = self.session.step()
            self._step += 1
            return state
        
        if self.game_type == 'DRLNegotiation':

            # competitors go forward one step
            # session/simulator/environment go one step
            self.step_competitors(action=action, competitor=competitor)
            self.session.step()

            # result 
            result = self.session.state

            # reward
            if self.env.strategy == "ac_s":
                # reward design for acceptance strategy
                reward = 0
                if competitor.time < competitor.maximum_time:

                    if competitor.action == ResponseType.ACCEPT_OFFER:
                        if result.agreement:
                            if "_rp" in competitor.__dict__:
                                if competitor.ufun(result.agreement) >= competitor.ufun(competitor._rp):
                                    reward = competitor.get_ufun(result.agreement)
                                    reward += EXTRA_REWARD
                                else:
                                    reward = -1
                            else:
                                reward = competitor.get_ufun(result.agreement)
                        else:
                            reward = 0

                    elif competitor.action == ResponseType.REJECT_OFFER:
                        # when competitor reject the offer
                        if competitor.proposal_offer:
                            # proposal a meaningful offer
                            # calculate the reward
                            if "_rp" in competitor.__dict__:
                                if competitor.ufun(competitor.proposal_offer) >= competitor.ufun(competitor._rp):
                                    reward = competitor.get_ufun(competitor.proposal_offer)
                                else:
                                    reward = -1
                            else:
                                reward = competitor.get_ufun(competitor.proposal_offer)
                            # competitor.set_proposal_offer(offer=None)
                        else:
                            # reject the offer, but not proposal a meanful offer
                            reward = -1

                    elif competitor.action == ResponseType.END_NEGOTIATION or \
                        competitor.action == ResponseType.NO_RESPONSE:

                        reward = -1

                else:
                    reward = 0

                return reward
            elif self.env.strategy == "of_s":
                if competitor.time < competitor.maximum_time:
                    reward = competitor.ufun(action)
                    if result.agreement:
                        reward += EXTRA_REWARD
                else:
                    reward = -1

                return reward

            elif self.env.strategy == "hybrid":
                raise NotImplementedError(f"Design of reward based on {self.env.strategy} is not Implemented!")
            else:
                raise ValueError(f"Design of reward based on {self.env.strategy} is error!")

    def step_competitors(self, action=None, competitor: Optional[DRLNegotiator] = None):
        """
        set the action of competitors
        """
        # for _ in self.competitors:
        #     if _.id == competitor.id:
        # just set the current action when t
        competitor.set_current_action(action=action)
            # else:
            #     _.set_current_action(action=None)

    def get_life(self):
        return self.session.running
    
    def get_state(self):
        return self.session.state

class DRLNegotiationGame(NegotiationGame):
    '''
    Game design for more general negotiation, based on self designed issues.
    Implemente the functions inherit from NegotiationGame and DRLGameMixIn,
    '''
    def __init__(
        self,
        name: str = 'dlr_negotiation_game',
        game_type: str = "DRLNegotiation",
        issues: Optional[List[Issue]] = None,
        n_steps: int = 100,
        competitors: Optional[List[Negotiator]] = None,
        env = None
    ):
        if competitors is None:
            competitors = [
                MyDRLNegotiator(name="c1", is_seller=False, env=env, init_proposal=False),
                MyOpponentNegotiator(name="c2", env=env)
            ]
        
        # Default issues used for negotiation game
        if not issues:
            issues = [
                Issue(values=10, name="quantity"), 
                Issue(values=n_steps, name="delivery_time"), 
                Issue(values=100, name="unit_price")
            ]


        super().__init__(
            name=name,
            game_type = game_type,
            issues = issues,
            n_steps = n_steps,
            competitors = competitors,
            env = env
        )


class MyDRLNegotiationGame(DRLNegotiationGame):
    '''
    '''
    def __init__(
        self,
        name: str = "test_my_drl_negotiation_game",
        env = None
    ):
        super().__init__(
            name=name,
            env=env
        )

####################################################################################################
# For SCML
#
#
####################################################################################################

class SCMLGame(Game):

    def __init__(
        self,
        competition: str = "std",
        reveal_names = True, 
        n_steps = 20, 
        n_configs = 2, 
        max_n_worlds_per_config = None,
        n_runs_per_world = 1,
        competitors: List[Agent] = [MyComponentsBasedAgent, DecentralizingAgent, BuyCheapSellExpensiveAgent],
        env = None,
        *args,
        **kwargs
    ):
        self.competition = competition
        self.reveal_names = reveal_names
        self.n_steps = n_steps,
        self.n_configs = n_configs
        self.max_n_worlds_per_config = max_n_worlds_per_config
        self.n_runs_per_world = n_runs_per_world
        self.competitors = competitors
    

    def step_forward(self, action=None):

        if self.game_type == 'DRLNegotiation' or self.game_type == 'DRLSCML':
            self._action = action
            
            # give the action passed through algorithm to negotiator

class DRLSCMLGame(DRLGameMixIn, SCMLGame):
    pass

class MyDRLSCMLGame(DRLSCMLGame):
    pass
