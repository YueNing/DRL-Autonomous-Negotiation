import os, sys
sys.path.append("/home/nauen/PycharmProjects/tn_source_code")

import argparse

from drl_negotiation.env import SCMLEnv
from drl_negotiation.policy import InteractivePolicy
import drl_negotiation.scenarios as scenarios
import ipdb
import time

if __name__ == '__main__':
    # parse parameters
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-s', '--scenario', default='scml.py', help="Path of the scenario Python script.")
    args = parser.parse_args()

    # load scenario from script
    scenario = scenarios.load(args.scenario).Scenario()
    
    # create world
    world = scenario.make_world()
    
    # create scml  environment
    env = SCMLEnv(
            world, 
            scenario.reset_world, 
            scenario.reward, 
            scenario.observation, 
            info_callback=None, 
            done_callback=scenario.done, 
            shared_viewer=False
            )
    
    # render call to create window
    env.render()
    # create interactive policies for each agent
    policies = [InteractivePolicy(env, i) for i in range(env.n)]
    
    # execution loop
    obs_n = env.reset()
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
        done = all(done_n)
        # render all agent views
        env.render()
        time.sleep(1)

