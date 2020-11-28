"""
A2C model
"""
import warnings
import tensorflow as tf
import time
from abc import ABC, abstractmethod
from ..helpers.base_class import AbstractEnvRunner, TensorboardWriter, Scheduler, A2CRunner
from policy import ActorCriticPolicy
from ..helpers.helper import make_session, outer_scope_getter, mse, get_trainable_vars

class BaseRLModel(ABC):
    '''
    Base RL model
    *: python函数中间有一个（）分隔，星号后面为*命名关键字参数，星号本身不是参数**。命名关键字参数，在函数调用时必须带参数名字进行调用。
    '''
    def __init__(self, policy, env, verbose=0, *, 
                 policy_base, policy_kwargs=None, seed=None, n_cpu_tf_sess=None):
            
        self.policy = policy
        
        self.env = env
        self.verbose = verbose
        self.policy_kwargs = {} if policy_kwargs is None else policy_kwargs
        self.observation_space = None
        self.action_space = None
        self.n_envs = None
        self._vectorize_action = False
        self.num_timesteps = 0
        self.graph = None
        self.sess = None
        self.params = None
        self.seed = seed
        self._param_load_ops = None
        self.n_cpu_tf_sess = n_cpu_tf_sess
        self.episode_reward = None
        self.ep_info_buf = None
        
        if env is not None:
            self.observation_space = env.observation_space
            self.action_space = env.action_space
    
    def get_env(self):
        return self.env
    
    def set_env(self, env):
        if env is None and self.env is None:
            if self.verbose >= 1:
                print("Loading a model without an environment, "
                      "this model cannot be trained until it has a valid environment.")
            return
        elif env is None:
            raise ValueError("Error: trying to replace the current environment with None")
        
        self.n_envs = 1 # requires_vec_env is False
        self.env = env
        self.episode_reward = None
        self.ep_info_buf = None
    
    def _init_num_timesteps(self, reset_num_timesteps=True) -> bool:
        
        if reset_num_timesteps:
            self.num_timesteps = 0
        
        new_tb_log = self.num_timesteps == 0
        return new_tb_log
    
    @abstractmethod
    def setup_model(self):
        pass
    
    def set_random_seed(self, seed):
        if seed is None:
            return
        if self.env is not None:
            self.env.seed(seed)
            self.env.action_space.seed(seed)
        self.action_space.seed(seed)
    
    def _setup_learn(self):
        if self.env is None:
            raise ValueError("Error: cannot train the model without a valid environment, please set an environment with"
                             "set_env(self, env) method.")
        if self.episode_reward is None:
            self.episode_reward = np.zeros((self.n_envs,))
        if self.ep_info_buf is None:
            self.ep_info_buf = deque(maxlen=100)
    
    @abstractmethod
    def get_parameter_list(self):
        pass
    
    def get_parameters(self):
        parameters = self.get_parameter_list()
        paramete_values = self.sess.run(parameters)
        return_dictionary = OrderedDict((param.name, value) for param, value in zip(parameters, parameter_values))
        return return_dictionary
    
    def _setup_load_operations(self):
        pass
    
    @abstractmethod
    def _get_pretrain_placeholders(self):
        pass
    
    def pretrain(self, dataset, n_epochs=10, learning_rate=1e-4,
                adam_epsilon=1e-8, val_interval=None):
        
        '''supervised learning given an expert dataset'''
        
        continuous_actions = isinstance(self.action_space, gym.spaces.Box)
        discrete_actions = isinstance(self.action_space, gym.spaces.Discrete)
        
        assert discrete_actions or continuous_actions, 'Only Discrete and Box action spaces are supported'
        
        """TODO: """
    
    @abstractmethod
    def learn(self, total_timesteps, callback=None, log_interval=100, tb_log_name="run", reset_num_timesteps=True):
        pass
    
    @abstractmethod
    def predict(self, observation, state=None, mask=None, deterministic=False):
        pass
    
    @abstractmethod
    def action_probability(self, observation, state=None, mask=None, actions=None, logp=False):
        pass
    
    def load_parameters(self, load_path_or_dict, exact_match=True):
        """
        TODO: load parameters from file or dictionary
        """
        pass
    
    @abstractmethod
    def save(self, save_path, cloudpickle=False):
        raise NotImplementedError()
    
    @classmethod
    @abstractmethod
    def load(cls, load_path, env=None, custom_objects=None, **kwargs):
        raise NotImplementedError()
    
    @staticmethod
    def _save_to_file_cloudpickle(save_path, data=None, params=None):
        """
        TODO: legacy code for saving models with cloudpickle
        """
        pass
    
    @staticmethod
    def _save_to_file_zip(save_path, data=None, params=None):
        """
        TODO: Save model to a .zip archive
        """
        pass
    
    @staticmethod
    def _save_to_file(save_path, data=None, params=None, cloudpickle=False):
        """
        TODO: Save model to a zip archive or cloudpickle file.
        """
        if cloudpickle:
            BaseRLModel._save_to_file_cloudpickle(save_path, data, params)
        else:
            BaseRLModel._save_to_file_zip(save_path, data, params)
    
    @staticmethod
    def _load_from_file_cloudpickle(load_path):
        pass
    
    @staticmethod
    def _load_from_file(load_path, load_data=True, custom_objects=None):
        pass
    
    def _init_callback(self, callback):
        pass


class ActorCriticRLModel(BaseRLModel):
    def __init__(self, policy, env, verbose=0, _init_setup_model=True, policy_base=ActorCriticPolicy, 
                 policy_kwargs=None, seed=None, n_cpu_tf_sess=None):
        super(ActorCriticRLModel, self).__init__(policy, env, verbose=verbose, 
                                                 policy_base=policy_base, seed=seed, 
                                                 n_cpu_tf_sess=n_cpu_tf_sess
                                                )
        self.sess = None
        self.initial_state = None
        self.step = None
        self.proba_step = None
        self.params = None
        self._runner = None
    
    @abstractmethod
    def _make_runner(self) -> AbstractRunner:
        raise NotImplementedError("This model is not configured to use a Runner!")
    
    @property
    def runner(self) -> AbstractRunner:
        if self._runner is None:
            self._runner = self._make_runner()
        return self._runner
    
    def set_env(self, env):
        self._runner = None
        super().set_env(env)
    
    @abstractmethod
    def setup_model(self):
        raise NotImplementedError

    @abstractmethod
    def learn(self, total_timesteps, callback=None, 
             log_interval=100, tb_log_name="run", reset_num_timesteps=True):
        raise NotImplementedError

    def predict(self, obvservation, state=None, deterministic=False):
        if state is None:
            state = self.initial_state
        observation = np.array(observation)
        # array([1, 2, 3, 4]) ->
        # array([[1],
        #        [2],
        #        [3],
        #        [4]])
        observation = observation.reshape((-1, ) + self.observation_space.shape)
        actions, _, states = self.step(observation, state, deterministic=deterministic)
        
        if isinstance(self.action_space, gym.sapces.Box):
            clipped_actions = self.clipped_actions(actions)
        
        return clipped_actions, states
    
    def clipped_actions(self, actions):
        return np.clip(actions, self.action_space.low, self.action_space.high)
    
    def action_probability(self, observation, state=None, actions=None, logp=False):
        if state is None:
            state = self.initial_state
        observation = np.array(observation)
        observation = observation.reshape((-1,) + self.observation_space.shape)
        actions_proba = self.proba_step(observation, state)
        
        if len(actions_proba) == 0:
            warnings.warn(f"Warning: action probability is not implemented for {type(self.action_space).__name__} action space Returning None.")
            return None
        
        if actions is not None:
            prob = None
            logprob = None
            actions = np.array([actions])
            if isinstance(self.action_space, gym.spaces.Discrete):
                actions = actions.reshape((-1, ))
                prob = actions_proba[np.arange(actions.shape[0]), actions]
            
            elif isinstance(self.action_space, gym.spaces.Box):
                actions = actions.reshape((-1, ) + self.action_space.shape)
                mean, logstd = actions_proba
                std = np.exp(logstd)
                
                n_elts = np.prod(mean.shape[1:])
                log_normalizer = n_elts / 2 * np.log(2 * np.i) + 0.5 * np.sum(logstd, axis=1)
                
                logprob = -np.sum(np.square(actions - mean) / (2 * std), axis=1) - log_normalizer
            else:
                warnings.warn(f"Warning: action_probability not implemented for {type(self.action_space).__name__} actions space. Returning None.")
                return None
            
            if logp:
                if logprob is None:
                    logprob = np.log(prob)
                ret = logprob
            else:
                if prob is None:
                    prob = np.exp(logprob)
                ret = prob
            
            ret = ret.reshape((-1, 1))
        else:
            ret = actions_proba
        
        return ret
        
    
class A2C(ActorCriticRLModel):
    '''
        policy: ActorCriticPolicy, the policy model to use(MlpPolicy)
        env: Gym env, the environment to learn from
        gamma: Discount factor, for reward
        n_steps: The number of steps to run for each environment per update
        vf_coef: value function coefficient for the loss calculation
        ent_coef: entropy coefficient for the loss calculation, add the entropy of the policy pi to the objective function improved exploration
        max_grad_norm: The maximum value for the gradient clipping
        learning_rate: The learning rate 
        alpha: RMSProp decay parameter, from paper default 0.99
        momentum: RMSProp momentum parameter, from paper default 0
        epsilon: RMSProp epsilon
        lr_schedule: The type of scheduler for the learning rate update
        verbose: the verbosity level
        tensorboard_log = the log location for tensorboard
        _init_setup_model: whether or not to build the network at the creation of the instance(used only for loading)
        policy_kwargs: additional arguments to be passed to the policy on creation
        full_tensorboard_log: enable additional logging when using tensorboard
        seed: seed for the pseudo-random generators (python, numpy, tensorflow)
        n_cpu_tf_sess: The number of threads for TensorFlow operations
    '''
    def __init__(self, policy, env, gamma=0.09, n_steps=5, vf_coef=0.25, ent_coef=0.01, max_grad_norm=0.5,
                learning_rate=7e-4, alpha=0.99, momentum=0.0, epsilon=1e-5, lr_schedule='constant',
                verbose=0, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None,
                full_tensorboard_log=False, seed=None, n_cpu_tf_sess=None):
        
        # basic
        self.n_steps = n_steps
        self.gamma = gamma
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm

        # Optimization RMSprop, root mean squard backpropagation
        self.alpha = alpha
        self.momentum = momentum
        self.epsilon = epsilon
        self.lr_schedule = lr_schedule
        self.learning_rate = learning_rate

        # tensorboard log
        self.tensorboard_log = tensorboard_log
        self.full_tensorboard_log = full_tensorboard_log

        # tensforflow placeholder
        self.learning_rate_ph = None
        self.n_batch = None
        self.actions_ph = None
        self.advs_ph = None
        self.rewards_ph = None
        
        # train
        self.pg_loss = None # policy gradient loss
        self.vf_loss = None # value function loss
        self.entropy = None 
        self.apply_backprop = None
        self.train_model = None
        self.step_model = None
        self.proba_step = None
        self.value = None
        self.initial_state = None
        self.learning_rate_schedule = None
        
        # tensorflow summary
        self.summary = None
        
        super(A2C, self).__init__(policy=policy, env=env, verbose=verbose,
                                 _init_setup_model=_init_setup_model, policy_kwargs=policy_kwargs,
                                 seed=seed, n_cpu_tf_sess=n_cpu_tf_sess)
        if _init_setup_model:
            self.setup_model()
    
    def _make_runner(self):
        return A2CRunner(self.env, self, n_steps=self.n_steps, gamma=self.gamma)
    
    def _get_pretrain_placeholders(self):
        pass
    
    def setup_model(self):
        
        assert issubclass(self.policy, ActorCriticPolicy), "Error: the input policy for the A2C model must be an isntance of ActorCriticPolicy."
        
        self.graph = tf.Graph()
        with self.graph.as_default(): # context manager, which overrides the current default graph for lifetime of the context
            self.set_random_seed(self.seed)
            self.sess = make_session(num_cpu=self.n_cpu_tf_sess, graph=self.graph)

            self.n_batch = self.n_envs * self.n_steps

            n_batch_step = None
            n_batch_train = None

            step_model = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs, 1, n_batch_step, reuse=False) # one step

            with tf.variable_scope("train_model", reuse=True, custom_getter=outer_scope_getter("train_model")):
                train_model = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs, self.n_steps, n_batch_train, reuse=True)
            
            with tf.variable_scope("loss", reuse=False):
                self.actions_ph = train_model.pdtype.sample_placeholder([None], name="action_ph")
                self.advs_ph = tf.placeholder(tf.float32, [None], name="advs_ph")
                self.rewards_ph = tf.placeholder(tf.float32, [None], name="rewards_ph")
                self.learning_rate_ph = tf.placeholder(tf.float32, [], name="learning_rate_ph")
                
                neglogpac = train_model.proba_distribution.neglogp(self.actions_ph)

                self.entropy = tf.reduce_mean(train_model.proba_distribution.entropy())
                self.pg_loss = tf.reduce_mean(self.advs_ph * neglogpac)
                self.vf_loss = mse(tf.squeeze(train_model.value_flat), self.rewards_ph)

                loss = self.pg_loss - self.entropy * self.ent_coef + self.vf_loss * self.vf_coef

                tf.summary.scalar('entropy_loss', self.entropy)
                tf.summary.scalar('policy_gradient_loss', self.pg_loss)
                tf.summary.scalar('value_function_loss', self.vf_loss)
                tf.summary.scalar('loss', loss)

                self.params = get_trainable_vars("model")
                grads = tf.gradients(loss, self.params)

                if self.max_grad_norm is not None:
                    grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
                grads = list(zip(grads, self.params))
            
            with tf.variable_scope("input_info", reuse=False):
                tf.summary.scalar('discounted_rewards', tf.reduce_mean(self.rewards_ph))
                tf.summary.scalar('learning_rate', tf.reduce_mean(self.learning_rate_ph))
                pass

            trainer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate_ph, decay=self.alpha,
                                                epsilon=self.epsilon, mementum=self.mementum)
            self.apply_backprop = trainer.apply_gradients(grads)
            self.train_model = train_model
            self.step_model =step_model
            self.step = step_model.step 
            self.proba_step = step_model.proba_step
            self.value = step_model.value 
            self.initial_state = step_model.initial_state
            tf.global_variables_initializer().run(session=self.sess)
            self.summary = tf.summary.merge_all()

                
    def _train_step(self, obs, states, rewards, masks, actions, values, update, writer=None):
        # advantage
        advs = rewards - values
        cur_lr = None
        for _ in range(len(obs)):
            cur_lr = self.learning_rate_schedule.value()
        
        assert cur_lr is not None, "Error, "
        
        td_map = {self.train_model.obs_ph: obs, self.actions_ph: actions, self.advs_ph:advs, 
                 self.rewards_ph: rewards, self.learning_rate_ph: cur_lr}
        
        if states is not None:
            td_map[self.train_model.states_ph] = states
            td_map[self.train_model.dones_ph] = masks
        
        if writer is not None:
            summary, policy_loss, value_loss, policy_entropy, _ = self.sess.run([self.summary, self.pg_loss, self.vf_loss, self.entropy, self.apply_backprop], td_map)
            writer.add_summary(summary, update*self.n_batch)
        else:
            policy_loss, value_loss, policy_entropy, _ = self.sess.run([self.pg_loss, self.vf_loss, self.entropy, self.apply_backprop], td_map)
        
        return policy_loss, value_loss, policy_entropy
    
    def learn(self, total_timesteps, callback=None, log_interval=100, tb_log_name="A2C",
             reset_num_timesteps=True):
        new_tb_log = self._init_num_timesteps(reset_num_timesteps)
        with TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) as writer:
            self._setup_learn()
            # change the learning rate based on the current step and special learning rate schedule
            self.learning_rate_schedule = Scheduler(initial_value=self.learning_rate, n_values=total_timesteps, 
                                                   schedule=self.lr_schedule)
            
            t_start = time.time()
            
            for update in range(1, total_timesteps // self.n_batch + 1):
                # run the env and get the relative results of rl,
                rollout = self.runner.run() # observations, states, rewards, masks, actions, values
                
                obs, states, rewards, masks, actions, values, ep_infos, true_reward = rollout
                
                self.ep_info_buf.extend(ep_infos)
                # train one step
                _, value_loss, policy_entropy = self._train_step(obs, states, rewards, masks, actions, values, 
                                                                self.num_timesteps // self.n_batch, writer)
                
                n_seconds = time.time() - t_start
                fps = int((update * self.n_batch) / n_seconds)
                
                # logger and verbose
        
        return self
    
    def save(self, save_path, cloudpickle=False):
        data = {
            "gamma": self.gamma,
            "n_steps": self.n_steps,
            "vf_coef": self.vf_coef,
            "ent_coef": self.ent_coef,
            "max_grad_norm": self.max_grad_norm,
            "learning_rate": self.learning_rate,
            "alpha": self.alpha,
            "epsilon": self.epsilon,
            "lr_schedule": self.lr_schedule,
            "verbose": self.verbose,
            "policy": self.policy,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "n_envs": self.n_envs,
            "n_cpu_tf_sess": self.n_cpu_tf_sess,
            "seed": self.seed,
            "policy_kwargs": self.policy_kwargs
        }
        
        params_to_save = self.get_parameters()
        self._save_to_file(save_path, data=data, params=params_to_save, cloudpickle=cloudpickle)