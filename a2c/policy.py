from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np
from stable_baselines.common.input import observation_input
from stable_baselines.common.distributions import make_proba_dist_type, CategoricalProbabilityDistribution, \
    MultiCategoricalProbabilityDistribution, DiagGaussianProbabilityDistribution, BernoulliProbabilityDistribution
from stable_baselines.common.tf_layers import conv, linear, conv_to_fc, lstm
from gym.spaces import Discrete

class BasePolicy(ABC):
    '''
        The base policy object
        sess: Tensorflow session, 
        ob_space: The observation space of the environment
        ac_space: The action space of the environment
        n_env: The number of environments to run
        n_steps: The number of steps to run for each environment
        n_batch: The number of batches to run (n_envs * n_steps)
        reuse: If the policy is reusable or not
        scale: whether or not to scale the input
        obs_phs: Tensorflow tensor, observation placeholder
        add_action_ph: whether or not to create an action placeholder
    '''
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, scale=False,
                obs_phs=None, add_action_ph=False):
        self.n_env = e_env
        self.n_steps = n_steps
        self.n_batch = n_batch 
        with tf.variable_scope("input", reuse=False):
            if obs_phs is None:
                self._obs_phs, self._processed_obs = observation_input(ob_space, n_batch, scale=scale)
            else:
                self._obs_ph, self._processed_obs = obs_phs
            
            self._action_ph = None
            if add_action_ph:
                self._action_ph = tf.placeholder(dtype=ac_space.dtype, 
                                                 shape=(n_batch, ) + ac_space.shape, 
                                                 name='action_ph')
        
        self.sess = sess
        self.reuse = reuse
        self.ob_space = ob_space
        self.ac_space = ac_space
    
    @property
    def is_discrete(self):
        '''
        is action space discrete.
        '''
        return isinstance(self.ac_space, Discrete)
    
    @property
    def initial_state(self):
        assert not self.recurrent, "when using recurrent policies, you must overwrite `initial_state()` method"
        return None
    
    @property
    def obs_ph(self) -> tf.Tensor:
        return self._obs_ph
    
    @property
    def processed_obs(self) ->tf.Tensor:
        return self._processed_obs
    
    @property
    def action_ph(self) -> tf.Tensor:
        return self._action_ph
    
    @staticmethod
    def _kwargs_check(feature_extraction, kwargs):
        if feature_extraction == 'mlp' and len(kwargs) > 0:
            raise ValueError("Unknow keywords for policy: {}".format(kwargs))
    
    @abstractmethod
    def step(self, obs, state=None, mask=None):
        raise NotImplementedError
    
    @abstractmethod
    def proba_step(self, obs, state=None, mask=None):
        raise NotImplementedError

def make_proba_dist_type(ac_space):
    pass

class ActorCriticPolicy(BasePolicy):
    
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, scale=False):
        super(ActorCriticPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, resue=reuse, scale=scale)
        self._pdtype = make_proba_dist_type(ac_space)
        self._policy = None
        self._proba_distribution = None
        self._value_fn = None
        self._action = None
        self._deterministic_action = None
        
    def _setup_init(self):
        """Sets up the distributions, actions, and value."""
        with tf.variable_scope("output", reuse=True):
            assert self.policy is not None and self.proba_distribution is not None and self.value_fn is not None
            self._action = self.proba_distribution.sample()
            self._deterministic_action = self.proba_distribution.mode() # ?
            self._neglogp = self.proba_distribution.neglogp(self.action) # ?
            if isinstance(self.proba_distribution, CategoricalProbabilityDistribution):
                self._policy_proba = tf.nn.softmax(self.policy)
            elif isinstance(self.proba_distribution, DiagGaussianProbabilityDistribution):
                self._policy_proba = [self.proba_distribution.mean, self.proba_distribution.std]
            elif isinstance(self.proba_distribution, BernoulliProbabilityDistribution):
                self._policy_proba = tf.nn.sigmoid(self.policy)
            elif isinstance(self.proba_distribution, MultiCategoricalProbabilityDistribution):
                self._policy_proba = [tf.nn.softmax(categorical.flatparam()) 
                                      for categorical in self.proba_distribution.categoricals]
            else:
                self._policy_proba = []
            self._value_flat = self.value_fn[:,0] # x[:,n]表示在全部数组（维）中取第n个数据，直观来说，x[:,n]就是取所有集合的第n个数据,
    
    @property
    def pdtype(self):
        return self._pdtype
    
    @property
    def policy(self):
        return self._policy
    
    @property
    def proba_distribution(self):
        return self._proba_distribution
    
    @property
    def value_fn(self):
        return self._value_fn
    
    @property
    def value_flat(self):
        return self._value_flat
    
    @property
    def action(self):
        return self._action
    
    @property
    def deterministic_action(self):
        return self._deterministic_action
    
    @property
    def neglogp(self):
        return self._neglogp
    
    @property
    def policy_proba(self):
        return self._policy_proba
    
    @abstractmethod
    def step(self, obs, state=None, mask=None, deterministic=False):
        raise NotImplementedError
    
    @abstractmethod
    def value(self, obs, state=None, mask=None):
        raise NotImplementedError

def mlp_extractor(flat_observations, net_arch, act_fun):
    """
    TODO: tf.flatten, 
    Architecture of the Acceptance Strategy [128, dict(vf=[512, 512], pi=[512])], 
    activation function for shared layer: relu6
    """
    latent = flat_observations
    policy_only_layers = []
    value_only_layers = []
    
    for idx, layer in enumerate(net_arch):
        if isinstance(layer, int): # base layer, shared layer
            layer_size = layer
            latent = act_fun(linear(latent, "shared_fc".format(idx), layer_size, init_scale=np.sqrt(2)))
        else:
            assert isinstance(layer, dict), "Error: the net_arch list can only contain ints and dicts"
            if 'pi' in layer:
                assert isinstance(layer['pi'], list), "Error: net_arch[-1]['pi'] must contain a list of integers."
                policy_only_layers = []
            if 'vf' in layer:
                assert isinstance(layer['vf'], list), "Error: net_arch[-1]['vf'] must contain a list of integers."
                value_only_layers = layer['vf']
    
    latent_policy = latent
    latent_value = latent
    
    for idx, (pi_layer_size, vf_layer_size) in enumerate(zip_longest(policy_only_layers, value_only_layers)):
        if pi_layer_size is not None:
            assert isinstance(pi_layer_size, int), "Error: net_arch[-1]['pi'] must only contain integers."
            latent_policy = act_fun(linear(latent_policy, "pi_fc{}".format(idx), pi_layer_size, init_scale=np.sqrt(2)))
        if vf_layer_size is not None:
            assert isinstance(vf_layer_size, int), "Error: net_arch[-1]['vf'] must only contain integers."
            latent_value = act_fun(linear(latent_value, "vf_fc{}".format(idx), vf_layer_size, init_scale=np.sqrt(2)))
    
    return latent_policy, latent_value
            

class FeedForwardPolicy(ActorCriticPolicy):
    
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, net_arch=None, 
                act_fun=tf.tanh, **kwargs):
        super(FeedForwardPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, 
                                               scale=(feature_extraction=="cnn"))
        self._kwargs_check(feature_extraction, kwargs) # kwargs, mlp, 
        
        if net_arch is None:
            # vf: value network, pi: policy network
            net_arch = [dict(vf=[64, 64], pi=[64, 64])]
        
        with tf.variable_scope("model", reuse=reuse):
            pi_latent, vf_latent = mlp_extractor(tf.layers.flatten(self.processed_obs), net_arch, act_fun)
            
            self._value_fn = linear(vf_latent, 'vf', 1)
            
            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)
        
        self._setup_init()
    
    def step(self, obs, state=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp], {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp], {self.obs_ph: obs})
        
        return action, value, self.initial_state, neglogp
    
    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})
    
    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})         


class CustomPolicy(FeedForwardPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, *args, **kwargs):
        super(CustomPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse, 
                                           net_arch=[128, dict(vf=[512, 512], pi=[512])], 
                                           feature_extraction="mlp", 
                                           act_fun=tf.nn.relu6,
                                            *args, **kwargs, 
                                          )
        
            