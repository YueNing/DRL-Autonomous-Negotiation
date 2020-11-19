import sys
import os
sys.path.append(r'/home/nauen/PycharmProjects/tn_source_code')

from train import train_negotiation
from scml_env import DRLNegotiationEnv, NEnv

def test_train_negotiation():
    #  train based on NegotiationEnv 
    
    env = DRLNegotiationEnv(
        name="my_negotiation_env"
    )
    
    plot = True

    assert isinstance(env, NEnv)

    model = "DQN"
    # import pdb;pdb.set_trace()
    done, _ = train_negotiation(plot=plot, model=model, env=env, monitor=False)
    assert done,  f'train false by the model {model}'

    # model = "PPO1"
    # done, _ = train_negotiation(plot=plot, model=model, env=env)
    # assert done,  f'train false by the model {model}'
    #
    # model = "Test"
    # done, _ = train_negotiation(plot=plot, model=model, env=env)
    # assert done,  f'train false by the model {model}'


if __name__ == '__main__':
    test_train_negotiation()