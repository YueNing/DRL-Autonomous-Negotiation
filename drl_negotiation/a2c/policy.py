import tensorflow as tf
import numpy as np
from pyglet.window import key

########################################################################
# Interactive policy for scml's agent
########################################################################
class Policy(object):
    def __init__(self):
        pass

    def action(self, obs):
        raise NotImplementedError


class InteractivePolicy(Policy):
    def __init__(self, env, agent_index):
        super(InteractivePolicy, self).__init__()
        self.env = env
        self.agent_index = agent_index
        # hard-coded keyboard events
        # self.management = [False for i in range(12)]
        # manage the uvalues of buyer and seller, seller 4, buyer 4
        self.management = [False for i in range(8)]
        self.comm = [False for i in range(env.world.dim_c)]
        # register keyboard events with this environment's window
        env.viewers[int(agent_index % env.n)].on_key_press = self.key_press
        env.viewers[int(agent_index % env.n)].on_key_release = self.key_release
        self.pressed = False

    def action(self, obs):
        if self.env.discrete_action_input:
            m = 0
            if self.management[0]: m = 1
            if self.management[1]: m = 2
            if self.management[2]: m = 3
            if self.management[3]: m = 4
            if self.management[4]: m = 5
            if self.management[5]: m = 6
            if self.management[6]: m = 7
            if self.management[7]: m = 8
        else:
            m = np.zeros(9)  # 7-d because of no-change action
            if self.management[0]: m[1] += 1.0
            if self.management[1]: m[2] += 1.0
            if self.management[2]: m[3] += 1.0
            if self.management[3]: m[4] += 1.0
            if self.management[4]: m[5] += 1.0
            if self.management[5]: m[6] += 1.0
            if self.management[6]: m[7] += 1.0
            if self.management[7]: m[8] += 1.0
            if True not in self.management:
                m[0] += 1.0
            else:
                print(f'{self.env.agents[self.agent_index]}:{m}')

        return np.concatenate([m, np.zeros(self.env.world.dim_c)])

    def key_press(self, k, mod):
        if k == key._1: self.management[0] = True
        if k == key._2: self.management[1] = True
        if k == key._3: self.management[2] = True
        if k == key._4: self.management[3] = True
        if k == key._5: self.management[4] = True
        if k == key._6: self.management[5] = True
        if k == key._7: self.management[6] = True
        if k == key._8: self.management[7] = True

    def key_release(self, k, mod):
        if k == key._1: self.management[0] = False
        if k == key._2: self.management[1] = False
        if k == key._3: self.management[2] = False
        if k == key._4: self.management[3] = False
        if k == key._5: self.management[4] = False
        if k == key._6: self.management[5] = False
        if k == key._7: self.management[6] = False
        if k == key._8: self.management[7] = False

##########################################################
# a2c trainer policy network
##########################################################
def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    """
    multi layers perceptron
    Args:
        input:
        num_outputs:
        scope:
        reuse:
        num_units:
        rnn_cell:

    Returns:

    """
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        out = input
        output = tf.layers.dense(out, num_units, activation=tf.nn.relu)
        out = tf.layers.dense(out, num_units, activation=tf.nn.relu)
        out = tf.layers.dense(out, num_outputs, activation=None)
        return out

from drl_negotiation.a2c.trainer import p_predict

def create_actor(make_obs_ph, act_space, scope):
    p_func = mlp_model
    act = p_predict(make_obs_ph=make_obs_ph,
                    act_space=act_space,
                    p_func=p_func,
                    scope=scope,
                    )
    return act