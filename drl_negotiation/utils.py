import random
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


# Distribution
class Pd(object):
    """
    Probality distribution
    """
    def flatparam(self):
        raise NotImplementedError()

    def mode(self):
        raise NotImplementedError()

    def logp(self):
        raise NotImplementedError()

    def kl(self, other):
        raise NotImplementedError()

    def entropy(self):
        raise NotImplementedError()

    def sample(self):
        raie NotImplementedError()

class PdType(object):
    """
    parametirzed family of pd
    """
    
    def param_placeholder(self, prepend_shape, name=None):
        return tf.placeholder(dtype=tf.float32, shape=prepend_shape+self.param_shape(), name=name)
    def sample_paceholder(self, prepend_shape, name=None):
        return tf.placeholder(dtype=self.sample_dtype(), shape=prepend_shape+self.sample_shape(), name=name)

class DiaGaussianPdType(PdType):
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
    def __init__(self, flat):
        self.flat = flat
        mean, logstd = tf.split(flat, 2, axis=1)
        self.mean = mean
        self.logstd = logstd
        self.std = tf.exp(logstd)

    def flatparam(self):
        return self.flat

    def mode(self):
        return self.mean

    def logp(self, x):
        return 

    def kl(self, other):
        assert isinstance(other, DiagGuassianPd)
        return

    def entropy(self):
        return 

    def sample(self):
        return self.mean + self.std * tf.random_normal(tf.shape(self.mean))

    @classmethod
    def fromflat(cls, flat):
        return cls(flat)

def make_pdtype(ac_space):
    from gym import spaces
    if isinstance(ac_space, spaces.Box):
        assert len(ac_space.shape) == 1
        return DiagGaussianPdType(ac_space.shape[0])
    elif isinstance(ac_space, spaces.Discrete):
        return SoftCategoricalPdType(ac_space.n)
    else:
        raise NotImplementedError


# tf utils
def function(inputs, outputs, updates=None):

    if isinstance(outputs, list):
        return _Function(inputs, outputs, updates)
    elif isinstance(outputs, (dict, collections.OrderedDict)):
        f = _Function(inputs, output.values(), updates)
        return lambda *args, **kwargs: type(outputs)(zip(outputs.keys(), f(*args, **kwargs)))
    else:
        f = _Function(inputs, [outputs], updates)
        return lambda *args, **kwargs: f(*args, **kwargs)[0]

class _Function(object):
    def __init__(self, inputs, outputs, updates, check_nan=False):
        for inp in inputs:
            if not issubclass(type(inp), TfInput):
                assert len(inp.op.inputs) == 0, "inputs should all be placeholders of rl_algs.common.IfInput"
        self.inputs =inputs
        updates = updates or []
        self.update_group = tf.group(*updates)
        self.outputs_update = list(outputs) + [self.update_group]
        self.givens = {}
        self.check_nan = check_nan

    def _feed_input(self, feed_dict, inpt, value):
        if issubclass(type(inpt), TfInput):
            feed_dict.update(inpt.make_feed_dict(value))
        elif is_placehodler(inpt):
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

        assert len(kwargs) == 0, f"Function got extra arguments {str(list(kwargs.keys()))}")

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

class PlaceholderInput(TfInput):
    def __init__(self, placeholder):
        self._placeholder = placeholder

    def get(self):
        return self._placeholder

    def make_feed_dict(self, data):
        return (self._placeholder: data)

class BatchInput(Placehholder):
    def __init__(self, shape, dtype=tf.float32, name=None):
        super().__init__(tf.placeholder(dtype, [None]+list(shape), name=name))

class Unit8Input(PlaceholderTfInput):
    def __init__(self, shape, name=None):

        super().__init__(tf.placeholder(tf.unit8, [None]+list(shape), name=name))
        self._shape= shape
        self._output = tf.cast(super().get(), tf.float32) / 255.0

    def get(self):
        return self._output

# operations

def sum(x, axis=None, keepdims=False):
    return tf.reduce_sum(x, axis=None if axis is None else [axis], keep_dims=keepdims)

def mean(x, axis=None, keepdims=False):
    return tf.reduce_mean(x, axis=None if axis is None else [axis], keep_dims=keepdims)

def var(x, axis=None, keepdims=False):
    meanx = mean(x, axis=axis, keepdims=keepdims)
    return mean(tf.square(x-means), axis=axis, keepdims=keepdims)

def std(x, axis=None, keepdims=False):
    return tf.sqrt(var(x, axis=axis, keepdims=keepdims))

def max(x, axis=None, keepdims=False):
    return tf.reduce_max(x, axis=None if axis is None else [axis], keep_dims=keepdims)

def min(x, axis=None, keepdims=False):
    return tf.reduce_min(x, axis=None if axis is None else [axis], keep_dims=keepdims)

def concatenate(arrs, axis=0):
    return tf.concat(axis=axis, values=arrs)

def argmax(x, axis=None):
    return tf.argmax(x, axis=axis)

def softmax(x, axis=None):
    return tf.nn.softmax(x, axis=axis)













