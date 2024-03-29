import os
import numpy as np
import collections
import random
import tensorflow.compat.v1 as tf
from tensorflow.python import pywrap_tensorflow
from gym import spaces
import pickle
from  negmas import Issue
from negmas.helpers import get_class
from typing import List, Tuple
from drl_negotiation.hyperparameters import *

# from scml_env import NegotiationEnv
# from mynegotiator import DRLNegotiator

def generate_config(n_issues=1):
    """
    Single issue
    For Anegma settings, generate random settings for game
    Returns:
        dict
    >>> generate_config()

    """
    issues = []
    n_steps = 100
    if n_issues == 1:
        issues.append(Issue((300, 550)))
        rp_range = (500, 550)
        ip_range = (300, 350)
        weights = [(-0.35, ), (0.25,)]
    if n_issues == 2:
        issues.append(Issue((0, 10))) # quantity
        issues.append(Issue((10, 100))) # unit price
        rp_range = None
        ip_range = None
        weights = [(0, -0.25), (0, 0.25)]
    if n_issues == 3:
        issues.append(Issue((0, 10))) # quantity
        issues.append(Issue((0, 100))) # delivery time
        issues.append(Issue((10, 100))) # unit price
        rp_range = None
        ip_range = None
        weights = [(0, -0.25, -0.6), (0, 0.25, 1)]

    t_range = (0, 100)
    return {
        "issues": issues,
        "rp_range": rp_range,
        "ip_range": ip_range,
        "max_t": random.randint(t_range[0] + 1, t_range[1]),
        "weights": weights,
        "n_steps": n_steps
    }

def genearate_observation_space(config=None, normalize: bool=True):
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

def generate_action_space(config=None, normalize=True):
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

def reverse_normalize(action: Tuple=None, source_rng: Tuple=None, rng=(-1, 1)):
    """
    used for SCMLAgent,
    action: (-1, 1)
    Args:
        action:
        agent:
        source_rng: ((cprice, 3/2 * cprice), (3/2 * cprice, 2 * cprice))
        rng:

    Returns: true action

    >>> reverse_normalize((-1, 1), ((10, 15), (15, 20)), rng=(-1, 1))
    (10, 20)
    """
    _action = []
    for index, _ in enumerate(action):
        x_min = source_rng[index][0]
        x_max = source_rng[index][1]
        result = ((_ - rng[0]) / (rng[1] - rng[0]))*(x_max - x_min) + x_min
        _action.append(int(result))

    return tuple(_action)

# Global session
def get_session():
    '''
        get the default tensorflow session
    '''
    return tf.get_default_session()

def make_session(num_cpu):
    """
        returns a session that will use num_cpu CPU's only
    """
    tf_config = tf.ConfigProto(
            inter_op_parallelism_threads=num_cpu,
            intra_op_parallelism_threads=num_cpu,
            )
    return tf.Session(config=tf_config)

def single_threaded_session():
    """
        Returns a session which will only use a single CPU
    """
    return make_session(1)

ALREADY_INITIALIZED = set()

def  initialize():
    """
        Initialize all uninitalized variables in the global scope
    """
    new_variables = set(tf.global_variables()) - ALREADY_INITIALIZED
    get_session().run(tf.variables_initializer(new_variables))
    ALREADY_INITIALIZED.update(new_variables)

# tf utils
def function(inputs, outputs, updates=None):
    '''
    like Theano function.
    Example:
        x = tf.placeholder(tf.int32, (), name="x")
        y = tf.placeholder(tf.int32, (), name="y")
        z = 3 * x + 2 * y
        lin = function([x, y], z, givens={y: 0})
        with single_threaded_session():
            initialize()
            assert lin(2) == 6
            assert lin(x=3) == 9
            assert lin(2, 2) == 10
            assert lin(x=2, y=3) == 12
    '''
    if isinstance(outputs, list):
        _function = _Function(inputs, outputs, updates)
    elif isinstance(outputs, (dict, collections.OrderedDict)):
        f = _Function(inputs, outputs.values(), updates)
        _function = lambda *args, **kwargs: type(outputs)(zip(outputs.keys(), f(*args, **kwargs)))
    else:
        f = _Function(inputs, [outputs], updates)
        _function = lambda *args, **kwargs: f(*args, **kwargs)[0]
    return _function

class _Function:
    '''
        Capsules functions
    '''
    def __init__(self, inputs, outputs, updates, check_nan=False):
        for inp in inputs:
            if not issubclass(type(inp), TfInput):
                assert len(inp.op.inputs) == 0,\
                    "inputs should all be placeholders of rl_algs.common.IfInput"
        self.inputs =inputs
        updates = updates or []
        self.update_group = tf.group(*updates)
        self.outputs_update = list(outputs) + [self.update_group]
        self.givens = {}
        self.check_nan = check_nan

    @staticmethod
    def _feed_input(feed_dict, inpt, value):
        if issubclass(type(inpt), TfInput):
            feed_dict.update(inpt.make_feed_dict(value))
        elif is_placeholder(inpt):
            feed_dict[inpt] = value

    def __call__(self, *args, **kwargs):
        assert len(args) <= len(self.inputs), "Too many arguments provided"
        feed_dict = {}
        
        # args
        for inpt, value in zip(self.inputs, args):
            self._feed_input(feed_dict, inpt, value)

        # kwargs
        kwargs_passed_inpt_names = set()
        for inpt in self.inputs[len(args):]:
            inpt_name = inpt.name.split(':')[0]
            inpt_name = inpt_name.split("/")[-1]
            assert inpt_name not in kwargs_passed_inpt_names, f"this function has two arguments with the same name {inpt_name}"
            if inpt_name in kwargs:
                kwargs_passed_inpt_names.add(inpt_name)
                self._feed_input(feed_dict, inpt, kwargs.pop(inpt_name))
            else:
                assert inpt in self.givens, "Missing argument" + inpt_name

        assert len(kwargs) == 0, f"Function got extra arguments {str(list(kwargs.keys()))}"

        # update feed dict with givens
        for inpt in self.givens:
            feed_dict[inpt] = feed_dict.get(inpt, self.givens[inpt])
        
        results = get_session().run(self.outputs_update, feed_dict=feed_dict)[:-1]
        if self.check_nan:
            if any(np.isnan(r).any() for r in results):
                raise RuntimeError("Nan detected")

        return results

# tf inputs
def is_placeholder(x):
    return type(x) is tf.Tensor and len(x.op.inputs) == 0

class TfInput(object):
    def __init__(self, name="unnamed"):
        self.name = name

    def get(self):
        raise NotImplementedError

    def make_feed_dict(self):
        raise NotImplementedError

class PlaceholderTfInput(TfInput):
    def __init__(self, placeholder):
        self._placeholder = placeholder

    def get(self):
        return self._placeholder

    def make_feed_dict(self, data):
        return {self._placeholder: data}

class BatchInput(PlaceholderTfInput):
    def __init__(self, shape, dtype=tf.float32, name=None):
        super().__init__(tf.placeholder(dtype, [None]+list(shape), name=name))

class Unit8Input(PlaceholderTfInput):
    def __init__(self, shape, name=None):

        super().__init__(tf.placeholder(tf.uint8, [None]+list(shape), name=name))
        self._shape= shape
        self._output = tf.cast(super().get(), tf.float32) / 255.0

    def get(self):
        return self._output

# scope
def scope_name():
    return tf.get_variable_scope().name

def scope_vars(scope, trainable_only=False):
    """
        get the paramters inside a scope
    """
    return tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES if trainable_only else tf.GraphKeys.GLOBAL_VARIABLES,
            scope=scope if isinstance(scope, str) else scope.name
            )
def absolute_scope_name(relative_scope_name):
    return scope_name() + "/" + relative_scope_name

# optimizer 
def minimize_and_clip(optimizer, objective, var_list, clip_val=10):
    if clip_val is None:
        return optimizer.minimize(objective, var_list=var_list)
    else:
        gradients = optimizer.compute_gradients(objective, var_list=var_list)
        for i, (grad, var) in enumerate(gradients):
            if grad is not None:
                gradients[i] = (tf.clip_by_norm(grad, clip_val), var)
        return optimizer.apply_gradients(gradients)

# ================================================================
# Saving variables
# ================================================================
def get_saver():
    return tf.train.Saver()

def load_state(fname, saver=None):
    """Load all the variables to the current session from the location <fname>"""
    if saver is None:
        saver = tf.train.Saver()
    saver.restore(get_session(), fname)
    return saver

def load_states(fnames, saver=None):
    if saver is None:
        saver = tf.train.Saver()

    saver.restore(get_session(), )

def traversal_dir_first_dir(path):
    list = []
    if (os.path.exists(path)):
        files = os.listdir(path)
        for file in files:
            m = os.path.join(path,file)
            if (os.path.isdir(m)):
                list.append(m)
    else:
        os.mkdir(path)

    return list

def load_weights(ckpt_path, prefix_list):
    vars_weights = {}
    reader = pywrap_tensorflow.NewCheckpointReader(ckpt_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in sorted(var_to_shape_map):
        for _pref in prefix_list:
            if key.startswith(_pref):
                vars_weights[key+':0'] = reader.get_tensor(key)
    return vars_weights

def save_as_scope(scope_prefix: "MADDPGAgentTrainer", save_dir=None, model_name=None, extra="/p_func"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope_prefix+extra)[-6:])
    # dirs = traversalDir_FirstDir(save_dir+scope_prefix)
    # if not dirs:
    #     sub_save_dir = '/'+'0001'+'/'
    # else:
    #     sub_save_dir = '/'+str(int(dirs[-1]) + 1).zfill(4)+'/'
    if not os.path.exists(save_dir+scope_prefix):
        os.mkdir(save_dir+scope_prefix)
    saver.save(get_session(), save_dir+scope_prefix+'/'+model_name)
    return saver

def save_state(fname, saver=None, global_step=None):
    """Save all the variables in the current session to the location <fname>"""
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    if saver is None:
        saver = tf.train.Saver()
    saver.save(get_session(), fname, global_step=global_step)
    return saver

def summary(filename):
    g = tf.Graph()
    with g.as_default() as g:
        tf.train.import_meta_graph(filename)
    with tf.Session(graph=g) as sess:
        tf.summary.FileWriter(logdir=SAVE_DIR, graph=g)

# operations

def _sum(x, axis=None, keepdims=False):
    return tf.reduce_sum(x, axis=None if axis is None else [axis], keep_dims=keepdims)

def _mean(x, axis=None, keepdims=False):
    return tf.reduce_mean(x, axis=None if axis is None else [axis], keep_dims=keepdims)

def _var(x, axis=None, keepdims=False):
    meanx = _mean(x, axis=axis, keepdims=keepdims)
    return _mean(tf.square(x-meanx), axis=axis, keepdims=keepdims)

def _std(x, axis=None, keepdims=False):
    return tf.sqrt(_var(x, axis=axis, keepdims=keepdims))

def _max(x, axis=None, keepdims=False):
    return tf.reduce_max(x, axis=None if axis is None else [axis], keep_dims=keepdims)

def _min(x, axis=None, keepdims=False):
    return tf.reduce_min(x, axis=None if axis is None else [axis], keep_dims=keepdims)

def _concatenate(arrs, axis=0):
    return tf.concat(axis=axis, values=arrs)

def _argmax(x, axis=None):
    return tf.argmax(x, axis=axis)

def _softmax(x, axis=None):
    return tf.nn.softmax(x, axis=axis)


########################################################
# negotiation model
########################################################
from drl_negotiation.a2c.trainer import MADDPGAgentTrainer

def load_seller_neg_model(path="NEG_SELL_PATH") ->MADDPGAgentTrainer:
    """

    Returns:
        model of seller, to decide the next step seller's action
    """
    pass

def load_buyer_neg_model(path="NEG_BUY_PATH"):
    """

    Returns:
        model of buyer, to decide the next step buyer's action
    """
    pass

###########################################################
# env
###########################################################
def make_env(scenario_name, arglist=None, save_config=False, load_config=False, save_dir=None, load_dir=None):
    from drl_negotiation.env import SCMLEnv
    import drl_negotiation.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + '.py').Scenario()

    # create world/game
    if load_config:
        try:
            with open(load_dir+'.pkl', 'rb') as file:
                config = pickle.load(file)
                agent_types = config['agent_types']
                config['agent_types'] = []
                for _ in agent_types:
                    config['agent_types'].append(get_class(_))
                logging.info(f"load world config successfully from {load_dir}")
        except FileNotFoundError as e:
            logging.error(f"Error when Try to load the file from {load_dir}, "
                          f"please ensure world config file in the path {load_dir}")
            logging.error(str(e))
            logging.info("Finished, will not load world config!")
            config = None
    else:
        config = None

    world = scenario.make_world(config)

    if save_config:
        world.save_config(file_name=save_dir)

    # create multi-agent supply chain management environment
    env = SCMLEnv(
        world,
        reset_callback=scenario.reset_world,
        reward_callback=scenario.reward,
        observation_callback=scenario.observation,
        info_callback=scenario.benchmark_data,
        done_callback=scenario.done,
        shared_viewer=False
    )

    return env


#####################################################################
# trainer
#####################################################################
from drl_negotiation.a2c.policy import mlp_model

def get_trainers(env, num_adversaries=0, obs_shape_n=None, arglist=None):
    #TODO: train seller and buyer together, env.action_space?

    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer

    action_space = env.action_space

    # if not only_seller:
    #     obs_shape_n = obs_shape_n * 2
    #     action_space = action_space * 2
    #     assert len(obs_shape_n)==env.n * 2, "Error, length of obs_shape_n is not same as 2*policy agents"
    #     assert len(action_space)==len(obs_shape_n), "Error, length of act_space_n and obs_space_n are not equal!"

    # first set up the adversaries, default num_adversaries is 0
    for i in range(num_adversaries):
        trainers.append(trainer(
            env.agents[i].name.replace("@", '-')+"_seller", model, obs_shape_n, action_space, i, arglist,
            local_q_func=(arglist.adv_policy == 'ddpg')
        ))
        if not ONLY_SELLER:
            trainers.append(
                trainer(
                    env.agents[i].name.replace("@", '-') + "_buyer", model, obs_shape_n, action_space,
                    i + 1, arglist,
                    local_q_func=(arglist.adv_policy == 'ddpg')
                )
            )
    # if not only_seller:
    #     for i in range(num_adversaries):
    #         trainers.append(
    #             trainer(
    #                 env.agents[i].name.replace("@", '-')+"_buyer", model, obs_shape_n, action_space, i+ int(len(obs_shape_n) / 2), arglist,
    #                 local_q_func=(arglist.adv_policy == 'ddpg')
    #             )
    #         )

    # set up the good agent
    for i in range(num_adversaries, env.n):
        trainers.append(trainer(
            env.agents[i].name.replace("@", '-')+"_seller", model, obs_shape_n, action_space, i, arglist,
            local_q_func=(arglist.good_policy == "ddpg")
        )
        )
        if not ONLY_SELLER:
            trainers.append(trainer(
                env.agents[i].name.replace("@", '-') + "_buyer", model, obs_shape_n, action_space,
                i + 1, arglist,
                local_q_func=(arglist.good_policy == 'ddpg')
            ))

    # if not only_seller:
    #     for i in range(num_adversaries, env.n):
    #         trainers.append(trainer(
    #             env.agents[i].name.replace("@", '-')+"_buyer", model, obs_shape_n, action_space, i+int(len(obs_shape_n) / 2), arglist,
    #             local_q_func=(arglist.good_policy == 'ddpg')
    #         ))

    return trainers


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

    return parser.parse_args()

#####################################################################################
# logging
#####################################################################################
def init_setup():
    if TRAIN and not RESTORE:
        try:
            os.makedirs(SAVE_DIR)
        except Exception as e:
            print(e)

def logging_setup():
        logging.basicConfig(level=LOGGING_LEVEL,
                            format='%(asctime)s  %(message)s',
                            datefmt='%a, %d %b %Y %H:%M:%S +0000',
                            filename=FILENAME if FILENAME != '' and TRAIN else './my.log')

#######################################################################################
# Visualize
#######################################################################################
try:
    import plotly.express as px
except Exception as e:
    print(e)

def show(filename, labels=None):
    with open(filename, 'rb') as fb:
        data = pickle.load(fb)

    fig = px.line(data, labels=labels)
    fig.show()
