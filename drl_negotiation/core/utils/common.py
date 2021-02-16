import os
import logging
from drl_negotiation.core.config.hyperparameters import (LOGGING_LEVEL, LOGGING_FILE_LEVEL,
                                                         FILENAME,
                                                         DISABLE_LOGGING_COLOR
                                                         )
from drl_negotiation.third_party.ansistrm.ansistrm import ColorizingStreamHandler

#########################################################################
# inputs
#########################################################################
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        "Reinforcement Learning experiments for multiagent supply chain managerment environments")

    # env
    parser.add_argument('--scenario', type=str, default="scml", help="name of the scenario script")
    parser.add_argument('--num-episodes', type=int, default=60000, help="number of episodes")
    parser.add_argument('--max-episode-len', type=int, default=100, help="maximum episode length")
    parser.add_argument('--num-adversaries', type=int, default=0, help="number of adversaries")
    parser.add_argument('--good-policy', type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="heuristic", help="policy of adversaries")

    # Training
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=128, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    parser.add_argument("--exp-name", type=str, default=None, help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="/tmp/policy/",
                        help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000,
                        help="save model once every time this many episodes are compeleted")
    parser.add_argument("--load-dir", type=str, default='',
                        help="directory in which training state and model are loaded")

    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/",
                        help="directory where plot data is saved")

    # Log
    parser.add_argument("--logging-level", type=str, default=logging.DEBUG, help="level of stream logging")

    return parser.parse_args()


#####################################################################################
# logging
#####################################################################################
def logging_setup(level=None, filename=None):
    """
    logging setup
    Args:
        level: logging level for stream handler, When None load it from hyperparamters.py
        filename: logging saved path, When None, load it from hyperparamters.py

    Returns:
        None
    """
    level = level if level is not None else LOGGING_LEVEL
    filename = filename if filename is not None else FILENAME

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        fmt='%(asctime)s  %(message)s',
        datefmt='%a, %d %b %Y %H:%M:%S +0000',
    )

    # FileHandler
    try:
        fh = logging.FileHandler(filename=filename)
    except FileNotFoundError:
        path = '/'.join(filename.split("/")[:-1])
        os.makedirs(path)
        fh = logging.FileHandler(filename=filename)
    fh.setLevel(LOGGING_FILE_LEVEL)
    fh.setFormatter(formatter)

    # StreamHandler
    if DISABLE_LOGGING_COLOR:
        ch = logging.StreamHandler()
    else:
        ch = ColorizingStreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)

    # add two handler
    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.info(f"log file saved in {filename}")