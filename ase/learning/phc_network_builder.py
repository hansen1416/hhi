"""
PHC network builder with morphology-conditioned FiLM modulation (actor only).

Key idea
--------
We keep the actor policy as a standard MLP, but condition a subset of its hidden
computations using FiLM-style modulation:

    h <- h * gamma(cond) + beta(cond)

where `cond` is the last 11 dims of the observation (gender + 10 betas).

Implementation notes
--------------------
- The environment produces a 585-D observation.
- Actor trunk consumes only the first 574 dims (state/task features).
- Last 11 dims are reserved for conditioning (gender + betas).
- Critic remains unchanged (consumes full obs).
- Discriminator is added for AMP-style tasks.
"""

from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import layers
from rl_games.algos_torch import network_builder

import torch
import torch.nn as nn
import numpy as np

DISC_LOGIT_INIT_SCALE = 1.0


class PHCBuilder(network_builder.A2CBuilder):
    """Wrapper for rl_games to build the PHC actor-critic + discriminator."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    class Network(network_builder.A2CBuilder.Network):
        """
        Actor-Critic network with FiLM-conditioned actor and auxiliary discriminator.

        Observation layout:
        - obs: [B, 585]
          - obs[:, :574]: state/task features (actor trunk input)
          - obs[:, 574:]: 11-D condition = [gender (1), betas (10)] (FiLM conditioner input)

        FiLM Conditioning:
        - A small MLP maps the 11-D condition to gamma/beta params for each actor hidden layer.

        Discriminator:
        - Separate MLP + linear head for AMP-style discrimination.
        """

        def __init__(self, params, **kwargs):
            # Calls self.load(params) internally, making actor/critic configs available.
            super().__init__(params, **kwargs)

            # Handle fixed sigma for continuous actions if not learning it.
            if self.is_continuous and not self.space_config["learn_sigma"]:
                actions_num = kwargs.get("actions_num")
                sigma_init = self.init_factory.create(**self.space_config["sigma_init"])
                self.sigma = nn.Parameter(
                    torch.zeros(actions_num, requires_grad=False, dtype=torch.float32),
                    requires_grad=False,
                )
                sigma_init(self.sigma)

            # Build discriminator on top of standard actor-critic.
            amp_input_shape = kwargs.get("amp_input_shape")
            self._build_disc(amp_input_shape)

            # Rebuild actor to consume only state/task dims (574).
            self._rebuild_actor_trunk(actor_in_dim=574)

            # Build FiLM conditioner.
            self._build_film_cond()

        def load(self, params):
            """Load hyperparameters from config (called by rl_games during init)."""
            super().load(params)

            # Discriminator config.
            self._disc_hidden_units = params["disc"]["units"]
            self._disc_activation = params["disc"]["activation"]
            self._disc_initializer = params["disc"]["initializer"]

            # Actor MLP config (shared for trunk and conditioner).
            self._actor_hidden_units = params["mlp"]["units"]
            self._actor_activation = params["mlp"]["activation"]
            self._actor_initializer = params["mlp"]["initializer"]

        def _rebuild_actor_trunk(self, actor_in_dim):
            """Rebuild actor MLP to take reduced input dim (state/task only)."""
            self._actor_in_dim = actor_in_dim  # Store for eval_actor slicing.

            # No CNN typically used in PHC/ASE; use identity.
            self.actor_cnn = nn.Identity()

            # Build MLP with original hidden units/activation but new input size.
            mlp_args = {
                "input_size": actor_in_dim,
                "units": self._actor_hidden_units,
                "activation": self._actor_activation,
                "dense_func": nn.Linear,
            }
            self.actor_mlp = self._build_mlp(**mlp_args)

            # Initialize weights like original MLP.
            mlp_init = self.init_factory.create(**self._actor_initializer)
            for m in self.actor_mlp.modules():
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)

        def _build_film_cond(self):
            """Build conditioner MLP that outputs FiLM params (gamma, beta) for each actor hidden layer."""
            cond_mlp_args = {
                "input_size": 11,  # gender + 10 betas
                "units": [64, 64],  # Simple two-layer conditioner
                "activation": self._actor_activation,
                "dense_func": torch.nn.Linear,
            }
            self.cond_mlp = self._build_mlp(**cond_mlp_args)

            # Output size: 2 * sum(hidden_units) for all gammas + betas.
            film_out_size = sum(2 * u for u in self._actor_hidden_units)
            self.cond_linear = torch.nn.Linear(cond_mlp_args["units"][-1], film_out_size)

            # Initialize conditioner like actor MLP.
            mlp_init = self.init_factory.create(**self._actor_initializer)
            for m in list(self.cond_mlp.modules()) + [self.cond_linear]:
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)

        def _split_film_params(self, cond_out):
            """Split flat cond_out into per-layer (gamma, beta) pairs."""
            film_params = []
            pos = 0
            for h in self._actor_hidden_units:
                h = int(h)
                gamma = cond_out[:, pos : pos + h]
                beta = cond_out[:, pos + h : pos + 2 * h]
                film_params.append((gamma, beta))
                pos += 2 * h
            return film_params

        def _forward_mlp_with_film(self, mlp, x, film_params):
            """Forward MLP with FiLM applied once per Linear-block (after non-linears)."""
            modules = list(mlp)
            film_idx = 0  # Tracks current FiLM param pair.

            for i, layer in enumerate(modules):
                x = layer(x)

                # Start pending FiLM after each Linear.
                if isinstance(layer, nn.Linear):
                    pending_film = True
                else:
                    pending_film = pending_film if 'pending_film' in locals() else False

                # Apply FiLM at block end: either end of MLP or before next Linear.
                is_end_of_block = (i == len(modules) - 1) or isinstance(modules[i + 1], nn.Linear)
                if pending_film and is_end_of_block:
                    gamma, beta = film_params[film_idx]
                    x = x * gamma + beta
                    film_idx += 1
                    pending_film = False

            return x

        def forward(self, obs_dict):
            """rl_games forward: returns policy outputs, value, and RNN states."""
            obs = obs_dict["obs"]
            states = obs_dict.get("rnn_states", None)

            actor_outputs = self.eval_actor(obs)
            value = self.eval_critic(obs)

            return actor_outputs + (value, states)

        def eval_actor(self, obs):
            """Actor forward with FiLM conditioning on gender/betas.
            Use film to condition on the gender-betas
            """
            state_obs = obs[:, :574]  # State/task features.
            gender_betas = obs[:, 574:]  # Condition (11 dims).

            a_out = self.actor_cnn(state_obs)
            a_out = a_out.contiguous().view(a_out.size(0), -1)

            # Compute FiLM params from condition.
            cond_out = self.cond_linear(self.cond_mlp(gender_betas))
            film_params = self._split_film_params(cond_out)

            # Forward actor MLP with FiLM.
            a_out = self._forward_mlp_with_film(self.actor_mlp, a_out, film_params)

            if self.is_discrete:
                return self.logits(a_out)

            if self.is_multi_discrete:
                return [logit(a_out) for logit in self.logits]

            if self.is_continuous:
                mu = self.mu_act(self.mu(a_out))
                if self.space_config["fixed_sigma"]:
                    sigma = mu * 0.0 + self.sigma_act(self.sigma)
                else:
                    sigma = self.sigma_act(self.sigma(a_out))
                return mu, sigma

        def eval_critic(self, obs):
            """Critic forward (uses full obs).
            critic: estimates the value of the current state, V(s), for PPO. 
            It is used to compute next_values, advantages, returns, and the PPO value loss.

            # todo0310, maybe we should condition the critic on gender,betas too

            state_obs = obs[:, :574]
            gender_betas = obs[:, 574:]

            c_out = self.critic_cnn(state_obs)
            c_out = c_out.contiguous().view(c_out.size(0), -1)

            cond_out = self.critic_cond_linear(self.critic_cond_mlp(gender_betas))
            film_params = self._split_critic_film_params(cond_out)
            c_out = self._forward_mlp_with_film(self.critic_mlp, c_out, film_params)

            value = self.value_act(self.value(c_out))
            """
            c_out = self.critic_cnn(obs)
            c_out = c_out.contiguous().view(c_out.size(0), -1)
            c_out = self.critic_mlp(c_out)
            return self.value_act(self.value(c_out))

        def eval_disc(self, amp_obs):
            """Discriminator forward for AMP observations.
            # todo0310:
            motion = amp_obs[:, :-11]
            shape  = amp_obs[:, -11:]
            x = torch.cat([motion, shape], dim=-1)
            d = self._disc_mlp(x)
            return self._disc_logits(d)

            append the same 11-D shape code to:
                - rollout amp_obs
                - replay amp_obs
                - demo amp_obs_demo.
            """
            disc_out = self._disc_mlp(amp_obs)
            return self._disc_logits(disc_out)

        def get_disc_logit_weights(self):
            """Flattened discriminator logit head weights (for regularization)."""
            return torch.flatten(self._disc_logits.weight)

        def get_disc_weights(self):
            """Flattened discriminator MLP + logit head weights."""
            weights = [torch.flatten(m.weight) for m in self._disc_mlp.modules() if isinstance(m, nn.Linear)]
            weights.append(torch.flatten(self._disc_logits.weight))
            return weights

        def _build_disc(self, input_shape):
            """Build discriminator MLP and logit head.

            # todo0310, condition the descriminator on gender,betas too
            """
            self._disc_mlp = nn.Sequential()

            mlp_args = {
                "input_size": input_shape[0],
                "units": self._disc_hidden_units,
                "activation": self._disc_activation,
                "dense_func": torch.nn.Linear,
            }
            self._disc_mlp = self._build_mlp(**mlp_args)

            out_size = self._disc_hidden_units[-1]
            self._disc_logits = torch.nn.Linear(out_size, 1)

            # Initialize MLP.
            mlp_init = self.init_factory.create(**self._disc_initializer)
            for m in self._disc_mlp.modules():
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)

            # Initialize logit head.
            torch.nn.init.uniform_(self._disc_logits.weight, -DISC_LOGIT_INIT_SCALE, DISC_LOGIT_INIT_SCALE)
            torch.nn.init.zeros_(self._disc_logits.bias)

    def build(self, name, **kwargs):
        """rl_games entry point: build and return Network."""
        return PHCBuilder.Network(self.params, **kwargs)