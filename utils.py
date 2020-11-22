import random
from  negmas import Issue

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

def observation_space_anegma(config=None):
    if config:
        return [
            [
                config.get("issues")[0].values[0],
                0,
                config.get("ip_range")[0],
                config.get("rp_range")[0]
            ],
            [   config.get("issues")[0].values[1],
                config.get("max_t"),
                config.get("ip_range")[1],
                config.get("rp_range")[1]
             ],
        ]