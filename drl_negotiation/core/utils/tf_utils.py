import os
import numpy as np
import collections
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

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
                assert len(inp.op.inputs) == 0, \
                    "inputs should all be placeholders of rl_algs.common.IfInput"
        self.inputs = inputs
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
        super().__init__(tf.compat.v1.placeholder(dtype, [None] + list(shape), name=name))


class Unit8Input(PlaceholderTfInput):
    def __init__(self, shape, name=None):
        super().__init__(tf.placeholder(tf.uint8, [None] + list(shape), name=name))
        self._shape = shape
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
# ================================================================
# Saving variables
# ================================================================
def get_saver():
    return tf.compat.v1.train.Saver()


def load_state(fname, saver=None):
    """Load all the variables to the current session from the location <fname>"""
    if saver is None:
        saver = tf.train.Saver()
    saver.restore(get_session(), fname)
    return saver


def traversalDir_FirstDir(path):
    list = []
    if (os.path.exists(path)):
        files = os.listdir(path)
        for file in files:
            m = os.path.join(path, file)
            if (os.path.isdir(m)):
                list.append(m)
    else:
        os.mkdir(path)

    return list


def save_as_scope(scope_prefix: "MADDPGAgentTrainer", save_dir=None, model_name=None, extra="/p_func"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope_prefix + extra)[-6:])
    # dirs = traversalDir_FirstDir(save_dir+scope_prefix)
    # if not dirs:
    #     sub_save_dir = '/'+'0001'+'/'
    # else:
    #     sub_save_dir = '/'+str(int(dirs[-1]) + 1).zfill(4)+'/'
    if not os.path.exists(save_dir + scope_prefix):
        os.mkdir(save_dir + scope_prefix)
    saver.save(get_session(), save_dir + scope_prefix + '/' + model_name)
    return saver


def save_state(fname, saver=None, global_step=None):
    """Save all the variables in the current session to the location <fname>"""
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    if saver is None:
        saver = tf.train.Saver()
    saver.save(get_session(), fname, global_step=global_step)
    return saver

# operations
def _sum(x, axis=None, keepdims=False):
    return tf.reduce_sum(x, axis=None if axis is None else [axis], keep_dims=keepdims)


def _mean(x, axis=None, keepdims=False):
    return tf.reduce_mean(x, axis=None if axis is None else [axis], keep_dims=keepdims)


def _var(x, axis=None, keepdims=False):
    meanx = _mean(x, axis=axis, keepdims=keepdims)
    return _mean(tf.square(x - meanx), axis=axis, keepdims=keepdims)


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