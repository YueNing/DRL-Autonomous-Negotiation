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


def get_common_args():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--difficulty', type=str, default='7', help='the difficulty of the game')
    parser.add_argument('--game_version', type=str, default='latest', help='the version of the game')
    parser.add_argument('--map', type=str, default='3m', help='the map of the game')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--step_mul', type=int, default=8, help='how many steps to make an action')
    parser.add_argument('--replay_dir', type=str, default='', help='absolute path to save the replay')
    # The alternative algorithms are vdn, coma, central_v, qmix, qtran_base,
    # qtran_alt, reinforce, coma+commnet, central_v+commnet, reinforce+commnet，
    # coma+g2anet, central_v+g2anet, reinforce+g2anet, maven
    parser.add_argument('--alg', type=str, default='qmix', help='the algorithm to train the agent')
    parser.add_argument('--n_steps', type=int, default=2000000, help='total time steps')
    parser.add_argument('--n_episodes', type=int, default=1, help='the number of episodes before once training')
    parser.add_argument('--last_action', type=bool, default=True,
                        help='whether to use the last action to choose action')
    parser.add_argument('--reuse_network', type=bool, default=True, help='whether to use one network for all agents')
    parser.add_argument('--gamma', type=float, default=0.95, help='discount factor')
    parser.add_argument('--optimizer', type=str, default="RMS", help='optimizer')
    parser.add_argument('--evaluate_cycle', type=int, default=500, help='how often to evaluate the model')
    parser.add_argument('--evaluate_epoch', type=int, default=64, help='number of the epoch to evaluate the agent')
    parser.add_argument('--model_dir', type=str, default='./model', help='model directory of the policy')
    parser.add_argument('--result_dir', type=str, default='./result', help='result directory of the policy')
    parser.add_argument('--load_model', type=bool, default=False, help='whether to load the pretrained model')
    parser.add_argument('--evaluate', type=bool, default=False, help='whether to evaluate the model')
    parser.add_argument('--cuda', type=bool, default=True, help='whether to use the GPU')
    args = parser.parse_args()
    return args

def get_mixer_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.mixing_embed_dim = 32
    args.two_hyper_layers = False
    args.hyper_hidden_dim = 64
    args.qtran_hidden_dim = 64
    args.lr = 5e-4

    # epsilon greedy
    args.epsilon = 1
    args.min_epsilon = 0.005
    anneal_steps = 500
    args.anneal_epsilon = (args.epsilon - args.min_epsilon) / anneal_steps
    args.epsilon_anneal_scale = 'step'

    # the number of the train steps in one epoch
    args.train_steps = 1

    # experience replay
    args.batch_size = 64
    args.buffer_size = int(2e3)

    # how often to save the model
    args.save_cycle = 5000

    # how often to update the target_net
    args.target_update_cycle = 200

    # QTRAN lambda
    args.lambda_opt = 1
    args.lambda_nopt = 1

    # prevent gradient explosion
    args.grad_norm_clip = 10

    # MAVEN
    args.noise_dim = 16
    args.lambda_mi = 0.001
    args.lambda_ql = 1
    args.entropy_coefficient = 0.001
    return args


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