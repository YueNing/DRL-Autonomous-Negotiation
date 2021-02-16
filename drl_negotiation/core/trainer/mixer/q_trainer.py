import copy
from drl_negotiation.core.trainer._trainer import AgentTrainer
from drl_negotiation.core.modules.mixer.vdn import VDNMixer
from drl_negotiation.core.modules.mixer.qmixer import QMixer
from torch.optim import RMSprop


class QTrainer(AgentTrainer):
    def __init__(self, args):
        super(QTrainer, self).__init__()
        self.args = args

        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise NotImplementedError
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

    def update(self, agent, t):
        # get episode batch, replay buffer
        batch = None

        rewards = batch["rewards"][:,:-1]
        actions = batch["actions"][:,:-1]
        terminated = batch["terminated"][:,:-1].float()

        avail_actions = batch["avail_actions"]




