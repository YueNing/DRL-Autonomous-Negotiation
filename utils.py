import random
from  negmas import Issue
from typing import List, Tuple
# from scml_env import NegotiationEnv
# from mynegotiator import DRLNegotiator

def generate_config_anegma():
    """
    Single issue
    For Anegma settings, generate random settings for game
    Returns:
        dict
    >>> generate_config_anegma()

    """
    issues = [Issue((300, 550))]
    rp_range = (500, 550)
    ip_range = (300, 350)
    t_range = (0, 100)
    return {
        "issues": issues,
        "rp_range": rp_range,
        "ip_range": ip_range,
        "max_t": random.randint(t_range[0] + 1, t_range[1])
    }

def observation_space_anegma(config=None, normalize: bool=True):
    if config:
        if normalize:
            return [
                [-1 for _ in config.get("issues")] + [-1],
                [1 for _ in config.get("issues")] + [1],
            ]
        # single issue
        return [
            [
                config.get("issues")[0].values[0],
                0,
                # config.get("ip_range")[0],
                # config.get("rp_range")[0]
            ],
            [   config.get("issues")[0].values[1],
                config.get("max_t"),
                # config.get("ip_range")[1],
                # config.get("rp_range")[1]
             ],
        ]

def action_space_anegma(config=None, normalize=True):
    if config:
        if normalize:
            return [
                [-1 for _ in config.get("issues")],
                [1 for _ in config.get("issues")]
            ]
        # single issue
        return [[config.get("issues")[0].values[0], ], [config.get("issues")[0].values[1], ]]

def normalize_observation(obs =None, negotiator:"DRLNegotiator" = None, rng=(-1, 1)) -> List:
    """

    Args: [(300, 0), (550, 1)]
        obs: [offer, time] -> [quantity, delivery_time, unit_price, time] -> (350, 0.10)
        issues:
    Returns: between -1 and 1

    """
    _obs = []
    for index, x_in in enumerate(obs[:-1]):
        x_min = negotiator.ami.issues[index].values[0]
        x_max = negotiator.ami.issues[index].values[1]

        result = (rng[1]-rng[0])*(
            (x_in - x_min) / (x_max-x_min)
        ) + rng[0]

        _obs.append(
            result
        )

    result = (rng[1]-rng[0])*(
        (obs[-1] - 0) / (negotiator.maximum_time - 0)
    ) + rng[0]

    _obs.append(result)

    return _obs

def reverse_normalize_action(action: Tuple=None, negotiator:"DRLNegotiator" = None, rng=(-1, 1)):

    _action = []
    for index, _ in enumerate(action):
        x_min = negotiator.ami.issues[index].values[0]
        x_max = negotiator.ami.issues[index].values[1]
        result = ((_ - rng[0]) / (rng[1] - rng[0]))*(x_max - x_min) + x_min
        _action.append(result)

    return _action