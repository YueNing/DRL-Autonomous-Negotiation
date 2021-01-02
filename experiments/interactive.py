import os, sys
sys.path.append("/home/nauen/PycharmProjects/tn_source_code")
from drl_negotiation.a2c.policy import InteractivePolicy
import time
from drl_negotiation.utils import make_env
from drl_negotiation.hyperparameters import *

if __name__ == '__main__':
    # parse parameters
    env = make_env("scml")
    
    # render call to create window
    env.render()
    # create interactive policies for each agent
    if not ONLY_SELLER:
        policies = []
        for i in range(env.n):
            policies.append(InteractivePolicy(env, i))
            policies.append(InteractivePolicy(env, i+1))
    else:
        policies = [InteractivePolicy(env, i) for i in range(env.n)]
    # if not ONLY_SELLER:
    #     policies = policies * 2

    # execution loop
    obs_n = env.reset()
    # if not ONLY_SELLER:
    #     obs_n = obs_n * 2

    done = False
    print(f'{env.world.n_steps}')
    #while not done:
    while True:
        print(env.world.current_step)
        act_n = []
        for i, policy in enumerate(policies):
            act_n.append(policy.action(obs_n[i]))
        
        #print(act_n)
        # step env
        obs_n, reward_n, done_n, _ = env.step(act_n)
        print(reward_n)
        done = all(done_n)
        # render all agent views
        env.render()
        time.sleep(1)

