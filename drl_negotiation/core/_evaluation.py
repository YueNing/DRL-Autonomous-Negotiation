import numpy as np
from dataclasses import dataclass

class Complexity:
    """
    Complexity of the agent and of the MAS environments
    Inaccessibility:The Inaccessibility parameter expresses the difficulty in gaining
                    complete access at any instant to the resources in its environment. Such resources
                    include the environment components (e.g. web services, DBMS, etc.) or data (e.g.
                    metadata, ontologies, etc.).using the metrics: comp_inacc and res_inacc, that assess
                    the inaccessibility of the agent environment components and data;  ag_inacc and max_inacc,
                    that represent the inaccessibility of the agent and the MAS environment, respectively.
    Instability: The Instability parameter expresses the way the environment evolves, and
                how fast. In other words, the difficulty in perceiving changes in the environment. The
                faster and more unpredictably the environment changes, the more complex it is.
    Complexity of the Interaction:
    """
    def __init__(self, agent, world):
        self._agent = agent
        self._world = world

    def comp_inacc(self, agent):
        """
        assesses the inaccessibility of the agent environment components
        (DBMS, other MAS agents with which it interacts, etc.). For each component
        Comp_inacc is 1 if the inaccessibility is high, 0.5 if it is medium, 0 if low. The
        agent overall value is the mean of the measured values.
        Returns:
            np.ndarray[float]
        """
        pass

    def res_inacc(self, agent):
        """
        assesses the inaccessibility of the agent environment data (metadata,
        ontologies, etc). For each type of datum, ResInacc is 1 if the inaccessibility is
        high, 0.5 if it is medium, 0 if low. The overall value is the mean of the measured
        values.
        Returns:
            float
        """
        pass

    def time(self):
        pass

    def dynam(self):
        pass

    def num_effect_act(self):
        pass

    def comp_grad(self):
        pass

    def coop_grad(self):
        pass

    def tr_rep_mod(self):
        pass

    def ag_inacc(self, agent):
        return np.mean(self.comp_inacc(agent) + self.res_inacc(agent))

    def mas_inacc(self):
        pass

    def ag_instab(self):
        pass

    def mas_instab(self):
        pass

    def ag_compl_int(self):
        pass

    def mas_compl_int(self):
        pass

    @property
    def ag_env_compl(self):
        pass

    @property
    def mas_env_compl(self):
        pass


class Rationality:
    pass


class Autonomy:
    pass


class Reactivity:
    pass


class Adaptability:
    pass


@dataclass
class Eva:
    r"""A tuple representing a single step Eva derived by the environment.
    Attributes:
        env_spec (dict):
    """
    complexity: Complexity
    rationality: Rationality
    autonomy: Autonomy
    reactivity: Reactivity
    adaptability: Adaptability





