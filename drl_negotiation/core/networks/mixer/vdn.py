import torch as th
import torch.nn as nn


class VDNMixerNet(nn.Module):
    def __init__(self):
        super(VDNMixerNet, self).__init__()

    def forward(self, agent_qs, batch):
        """VDN, sum local q values as global q value"""
        return th.sum(agent_qs, dim=2, keepdim=True)
