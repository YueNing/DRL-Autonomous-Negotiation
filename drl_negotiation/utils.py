import numpy as np
import collections
import random
import tensorflow as tf
from gym import spaces
from  negmas import Issue
from typing import List, Tuple
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

# Global session
def get_session():
    '''
        get the default tensorflow session
    '''
    return tf.compat.v1.get_default_session()

def make_session(num_cpu):
    """
        returns a session that will use num_cpu CPU's only
    """
    tf_config = tf.compat.v1.ConfigProto(
            inter_op_parallelism_threads=num_cpu,
            intra_op_parallelism_threads=num_cpu,
            )
    return tf.compat.v1.Session(config=tf_config)

def single_threaded_session():
    """
        Returns a session which will only use a single CPU
    """
    return make_session(1)

ALREADY_INITIALIZED = set()

def initialize():
    """
        Initialize all uninitalized variables in the global scope
    """
    new_variables = set(tf.compat.v1.global_variables()) - ALREADY_INITIALIZED
    get_session().run(tf.compat.v1.variables_initializer(new_variables))
    ALREADY_INITIALIZED.update(new_variables)

# Distribution

class Pd:
    """
    Probality distribution, for action
    """
    def flatparam(self):
        '''
            flat the params
        '''
        raise NotImplementedError()

    def mode(self):
        '''
            value that appears most often in a set of data values.
        '''
        raise NotImplementedError()

    def logp(self, x):
        '''
            log probability
        '''
        raise NotImplementedError()

    def kl(self, other):
        '''
            kullback-Leibler Divergence
        '''

        raise NotImplementedError()

    def entropy(self):
        '''
            entropy
        '''
        raise NotImplementedError()

    def sample(self):
        '''
            sample data from this probability distribution
        '''
        raise NotImplementedError()

class PdType:
    """
    parametirzed family of pd
    """
    def param_shape(self):
        '''
            the shape of params
        '''
        raise NotImplementedError
    def sample_shape(self):
        '''
            the shape of sample data
        '''
        raise NotImplementedError
    def sample_dtype(self):
        '''
            the type of the sampled data
        '''
        raise NotImplementedError
    def pdfromflat(self, flat):
        '''
            create probability distribution from flat params
        '''
        return self.pdclass()(flat)
    def pdclass(self):
        '''
            return the class of probability distribution
        '''
        raise NotImplementedError

    def param_placeholder(self, prepend_shape, name=None):
        '''
            parameters placeholder with tensorflow
        '''
        return tf.placeholder(dtype=tf.float32, shape=prepend_shape+self.param_shape(), name=name)
    
    def sample_placeholder(self, prepend_shape, name=None):
        '''
            sample data placeholder with tensorflow
        '''
        return tf.compat.v1.placeholder(dtype=self.sample_dtype(),
                shape=prepend_shape+self.sample_shape(),
                name=name
                )

class DiagGaussianPdType(PdType):
    '''
        Gaussian Distribution With a Diagonal Covariance Matrix,
        multivariate gaussian distribution type
    '''
    def __init__(self, size):
        self.size = size

    def pdclass(self):
        return DiagGaussianPd

    def param_shape(self):
        return [2*self.size]

    def sample_shape(self):
        return [self.size]

    def sample_dtype(self):
        return tf.float32

class DiagGaussianPd(Pd):
    '''
        Gaussian distribution with a diagonal covariance matrix,
        multivariate guassian distribution,
        off-diagonals of the covariance matrix only play a minor
        role, an alternative representation of a multivariate Gaussian
        distribution.
    '''
    def __init__(self, flat):
        self.flat = flat
        _mean, _logstd = tf.split(flat, 2, axis=1)
        self.mean = _mean
        self.logstd = _logstd
        self.std = tf.exp(_logstd)

    def flatparam(self):
        return self.flat

    def mode(self):
        return self.mean

    def logp(self, x):
        result = -0.5 * _sum(tf.square((x-self.mean) / self.std), axis=1)\
                -0.5 * np.log(2.0 * np.pi) * tf.cast(tf.shape(x)[1], dtype=tf.float32)\
                -_sum(self.logstd, axis=1)
        return result 

    def kl(self, other):
        assert isinstance(other, DiagGaussianPd)
        #TODO
        result = None
        return result

    def entropy(self):
        return None

    def sample(self):
        return self.mean + self.std * tf.random_normal(tf.shape(self.mean))

    @classmethod
    def fromflat(cls, flat):
        '''
        return the class from flat parameters
        '''
        return cls(flat)

class SoftCategoricalPdType(PdType):
    def __init__(self, ncat):
        self.ncat = ncat

    def pdclass(self):
        return SoftCategoricalPd

    def param_shape(self):
        return [self.ncat]

    def sample_shape(self):
        return [self.ncat]

    def sample_dtype(self):
        return tf.float32

class SoftCategoricalPd(Pd):
    """
       Soft Categorical probability distribution 
    """

    def __init__(self, logits):
        self.logits = logits

    def flatparam(self):
        return self.logits

    def mode(self):
        return _softmax(self.logits, axis=-1)
    
    def logp(self, x):
        return -tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=x)

    def kl(self, other):
        pass

    def entropy(self):
        pass

    def sample(self):
        u = tf.random.uniform(tf.shape(self.logits))
        return _softmax(self.logits - tf.math.log(-tf.math.log(u)), axis=-1)

    @classmethod
    def fromflat(cls, flat):
        return cls(flat)

class BernoulliPdType(PdType):
    def __init__(self, size):
        self.size = size

    def pdclass(self):
        return BernoulliPd

    def param_shape(self):
        return [self.size]

    def sample_shape(self):
        return [self.size]

    def sample_dtype(self):
        return tf.int32

class BernoulliPd(Pd):
    """
        0 and 1, bernoullipd probability distribution
    """
    def __init__(self, logits):
        self.logits = logits
        self.ps = tf.sigmoid(logits)

    def flatparam(self):
        return self.logits

    def mode(self):
        return tf.round(self.ps)

    def logp(self, x):
        pass

    def kl(self, other):
        pass

    def entropy(self):
        pass

    def sample(self):
        pass

    @classmethod
    def fromflat(cls, flat):
        return cls(flat)


def make_pdtype(ac_space):
    '''
        create probability distribution type based on the type of action space
    '''
    if isinstance(ac_space, spaces.Box):
        assert len(ac_space.shape) == 1
        pd_type = DiagGaussianPdType(ac_space.shape[0])
    elif isinstance(ac_space, spaces.Discrete):
        pd_type = SoftCategoricalPdType(ac_space.n)
    else: raise NotImplementedError()
    return pd_type


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
        super().__init__(tf.compat.v1.placeholder(dtype, [None]+list(shape), name=name))

class Unit8Input(PlaceholderTfInput):
    def __init__(self, shape, name=None):

        super().__init__(tf.placeholder(tf.uint8, [None]+list(shape), name=name))
        self._shape= shape
        self._output = tf.cast(super().get(), tf.float32) / 255.0

    def get(self):
        return self._output

# scope
def scope_name():
    return tf.compat.v1.get_variable_scope().name

def scope_vars(scope, trainable_only=False):
    """
        get the paramters inside a scope
    """
    return tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES if trainable_only else tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,
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
