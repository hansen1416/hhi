"""
HHI network builder using plain MLP (PHC/AMP style).

This version matches the original AMP/PHC networks from:
- https://github.com/nv-tlabs/ASE (Adversarial Motion Priors)
- https://github.com/ZhengyiLuo/PHC (Perpetual Humanoid Control)

Shape parameters (gender + 10 betas = 11-D total from HUMOS) are now
concatenated directly into the observation vector for the actor/critic
and into the AMP features for the discriminator. No FiLM modulation,
no separate conditioning modules.

Observation layout (environment):
    obs: [B, state_dim + 11]   ← last 11 dims = [gender, beta_1, ..., beta_10]

AMP layout:
    amp_obs: [B, amp_dim] or [T, N, amp_dim]
    amp_shape: [B, 11] or [T, N, 11]   ← concatenated inside eval_disc

This is the clean baseline you requested for the HHI project.
It will make training on the 128 HUMOS-generated body-shape variations
(64 shapes × 2 genders per AMASS motion) much simpler and closer to the
original non-physical → physical motion conversion goal.
"""

from rl_games.common import object_factory
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import layers
import learning.rl_games_network_builder as network_builder

import torch
import torch.nn as nn

DISC_LOGIT_INIT_SCALE = 1.0


class HHIBuilder(network_builder.A2CBuilder):
    """Wrapper for rl_games to build the plain-MLP PHC actor-critic + discriminator."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    class Network(network_builder.A2CBuilder.Network):
        """
        Plain MLP Actor-Critic + Discriminator (no FiLM).

        - Actor / Critic receive the FULL observation (state + 11 shape dims).
        - Discriminator receives AMP features concatenated with the 11 shape dims.
        - This is exactly how the original AMP/PHC networks handled extra inputs.
        """

        def __init__(self, params, **kwargs):
            """
            Build the full model using the base rl_games MLP construction.

            The base class already builds actor_mlp and critic_mlp with the
            full observation size (state_dim + 11 shape dims) coming from
            the environment config.
            """
            # Let the base rl_games class build the default actor/critic parts.
            # This automatically uses the full input size that now includes shape.
            super().__init__(params, **kwargs)

            # Fixed sigma handling (exactly as in original PHC/AMP).
            if self.is_continuous and not self.space_config["learn_sigma"]:
                actions_num = kwargs.get("actions_num")
                sigma_init = self.init_factory.create(**self.space_config["sigma_init"])
                self.sigma = nn.Parameter(
                    torch.zeros(actions_num, requires_grad=False, dtype=torch.float32),
                    requires_grad=False,
                )
                sigma_init(self.sigma)

            self._shape_dim = 11

            # Build discriminator (now expects AMP obs + concatenated shape).
            amp_input_shape = kwargs.get("amp_input_shape")
            self._build_disc(amp_input_shape)

        def load(self, params):
            """
            Read hyperparameters from config (kept exactly as before).
            """
            super().load(params)

            # Discriminator hyperparameters.
            self._disc_hidden_units = params["disc"]["units"]
            self._disc_activation = params["disc"]["activation"]
            self._disc_initializer = params["disc"]["initializer"]

            # Actor/Critic hyperparameters (still needed for _build_mlp calls
            # if the base class uses them, and for any future extensions).
            self._actor_hidden_units = params["mlp"]["units"]
            self._actor_activation = params["mlp"]["activation"]
            self._actor_initializer = params["mlp"]["initializer"]

            self._critic_hidden_units = params["mlp"]["units"]
            self._critic_activation = params["mlp"]["activation"]
            self._critic_initializer = params["mlp"]["initializer"]

        def _build_disc(self, input_shape):
            """
            Build discriminator as plain MLP.

            Input = AMP motion features + concatenated shape (11-D).
            """
            # input_shape[0] is the original AMP motion dim (e.g. 2920).
            # We add the shape dim here.
            disc_input_dim = input_shape[0] + self._shape_dim

            mlp_args = {
                "input_size": disc_input_dim,
                "units": self._disc_hidden_units,
                "activation": self._disc_activation,
                "dense_func": nn.Linear,
            }
            self._disc_mlp = self._build_mlp(**mlp_args)

            out_size = self._disc_hidden_units[-1]
            self._disc_logits = torch.nn.Linear(out_size, 1)

            # Same initialization as original PHC/AMP discriminator.
            mlp_init = self.init_factory.create(**self._disc_initializer)
            for m in self._disc_mlp.modules():
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)

            torch.nn.init.uniform_(self._disc_logits.weight, -DISC_LOGIT_INIT_SCALE, DISC_LOGIT_INIT_SCALE)
            torch.nn.init.zeros_(self._disc_logits.bias)

        def forward(self, obs_dict):
            """
            Standard rl_games entry point (unchanged).
            """
            obs = obs_dict["obs"]
            states = obs_dict.get("rnn_states", None)

            actor_outputs = self.eval_actor(obs)
            value = self.eval_critic(obs)

            return actor_outputs + (value, states)

        def eval_actor(self, obs):
            """
            Plain MLP actor on full observation (state + 11 shape dims).
            """
            a_out = self.actor_cnn(obs)
            a_out = a_out.contiguous().view(a_out.size(0), -1)

            # Standard rl_games action head (exactly like original PHC/AMP).
            if self.is_discrete:
                return self.logits(a_out)
            if self.is_multi_discrete:
                return [logit(a_out) for logit in self.logits]
            if self.is_continuous:
                mu = self.mu_act(self.mu(a_out))
                if self.space_config.get("fixed_sigma", False):
                    sigma = mu * 0.0 + self.sigma_act(self.sigma)
                else:
                    sigma = self.sigma_act(self.sigma(a_out))
                return mu, sigma

        def eval_critic(self, obs):
            """
            Plain MLP critic on full observation (state + 11 shape dims).
            """
            c_out = self.critic_cnn(obs)
            c_out = c_out.contiguous().view(c_out.size(0), -1)
            return self.value_act(self.value(c_out))

        def eval_disc(self, amp_obs, amp_shape):
            """
            Plain MLP discriminator.

            amp_shape is concatenated to amp_obs (works for both [B, D]
            and [T, N, D] shaped tensors).
            """
            # Concatenate along the feature dimension.
            disc_input = torch.cat([amp_obs, amp_shape], dim=-1)

            disc_out = self._disc_mlp(disc_input)
            return self._disc_logits(disc_out)

        def get_disc_logit_weights(self):
            """
            Return flattened final discriminator head weights (unchanged).
            """
            return torch.flatten(self._disc_logits.weight)

        def get_disc_weights(self):
            """
            Return flattened discriminator weights (unchanged).
            """
            weights = [torch.flatten(m.weight) for m in self._disc_mlp.modules() if isinstance(m, nn.Linear)]
            weights.append(torch.flatten(self._disc_logits.weight))
            return weights

    def build(self, name, **kwargs):
        """
        rl_games entry point (unchanged).
        """
        return HHIBuilder.Network(self.params, **kwargs)