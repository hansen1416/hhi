import torch.nn as nn
from rl_games.algos_torch.models import ModelA2CContinuousLogStd

class ModelPHCContinuous(ModelA2CContinuousLogStd):
    def __init__(self, network):
        super().__init__(network)
        return

    def build(self, config):
        net = self.network_builder.build('amp', **config)
        for name, _ in net.named_parameters():
            print(name)
        return ModelPHCContinuous.Network(net)

    class Network(ModelA2CContinuousLogStd.Network):
        def __init__(self, a2c_network):
            super().__init__(a2c_network)
            return

        def forward(self, input_dict):
            is_train = input_dict.get('is_train', True)
            result = super().forward(input_dict)

            if (is_train):
                # we should first figure out the math, like actor/discriminator/critic
                # amp_obs/amp_obs_replay/amp_obs_demo
                # maybe not really need to condition everything on the same shape
                # disc-shape-condition

                # the current fake branch from the latest rollout.
                # [amp_minibatch_size, 1960], [amp_minibatch_size, 11] 
                disc_agent_logit = self.a2c_network.eval_disc(input_dict['amp_obs'], input_dict['amp_shape'])
                result["disc_agent_logit"] = disc_agent_logit

                # the older fake branch: 
                # past agent-generated AMP observations kept in _amp_replay_buffer 
                # so the discriminator does not train only against the latest rollout, 
                # which improves stability and reduces oscillation.
                # [amp_minibatch_size, 1960], [amp_minibatch_size, 11]
                disc_agent_replay_logit = self.a2c_network.eval_disc(input_dict['amp_obs_replay'], input_dict['amp_shape_replay'])
                result["disc_agent_replay_logit"] = disc_agent_replay_logit

                # the real discriminator branch: 
                # real motion windows sampled from the motion library and stored in _amp_obs_demo_buffer
                # [amp_minibatch_size, 1960], [amp_minibatch_size, 11]
                disc_demo_logit = self.a2c_network.eval_disc(input_dict['amp_obs_demo'], input_dict['amp_shape_demo'])
                result["disc_demo_logit"] = disc_demo_logit

            return result