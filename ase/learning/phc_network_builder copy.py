"""
PHC network builder with morphology-conditioned FiLM modulation (actor only).

Key idea
--------
We keep the actor policy as a standard MLP, but *condition* a subset of its hidden
computations using FiLM-style modulation:

    h <- h * gamma(cond) + beta(cond)

where `cond` is the last 11 dims of the observation (gender + 10 betas).

Implementation notes
--------------------
- The environment still produces a 585-D observation.
- We *rebuild the actor trunk* to consume only the first 574 dims (state/task),
  while the last 11 dims are reserved for conditioning.
- The critic remains unchanged (still consumes the full observation by default).
"""

from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import layers
from rl_games.algos_torch import network_builder

import torch
import torch.nn as nn
import numpy as np

DISC_LOGIT_INIT_SCALE = 1.0

class PHCBuilder(network_builder.A2CBuilder):
    """Network builder wrapper used by rl_games to construct the PHC actor-critic + discriminator."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    class Network(network_builder.A2CBuilder.Network):
        """
        Actor-Critic network with an auxiliary discriminator and FiLM-conditioned actor.

        Observation layout assumption (important)
        ----------------------------------------
        obs: [B, 585]
          - obs[:, :574]   : state/task features (used by actor trunk)
          - obs[:, 574:]   : 11-D condition = [gender, betas(10)] (used by FiLM conditioner)

        Conditioning
        ------------
        A small conditioner MLP maps the 11-D condition to a vector of FiLM parameters
        (gamma, beta) for *each* hidden layer of the actor MLP.

        Discriminator
        -------------
        A separate MLP + linear head for AMP-style discrimination.
        """
        
        def __init__(self, params, **kwargs):
            # `super().__init__` will call `self.load(params)` (rl_games convention),
            # so _actor_units/_actor_activation/_actor_initializer, etc. become available.
            super().__init__(params, **kwargs)
            
            # For continuous control: optionally create a fixed sigma parameter (when learn_sigma=False).
            if self.is_continuous:
                if (not self.space_config['learn_sigma']):
                    actions_num = kwargs.get('actions_num')
                    sigma_init = self.init_factory.create(**self.space_config['sigma_init'])
                    self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=False, dtype=torch.float32), requires_grad=False)
                    sigma_init(self.sigma)
            
            amp_input_shape = kwargs.get('amp_input_shape')
            # This is the place that adds the discriminator on top of the standard actor–critic.
            self._build_disc(amp_input_shape)

            # reduce actor input: 574 = state/task only
            self._rebuild_actor_trunk(actor_in_dim=574)

            self._build_film_cond()

            return

        def _rebuild_actor_trunk(self, actor_in_dim: int):
            """
            Replace the actor trunk (actor_cnn + actor_mlp) such that the actor consumes
            only `actor_in_dim` features.

            This keeps the original MLP hidden units/activation/initializer defined in config,
            but changes only the input dimensionality.
            """
            # remember the actor input dim (so eval_actor can slice consistently)
            self._actor_in_dim = actor_in_dim

            # PHC/ASE typically uses no CNN; treat actor_cnn as identity
            self.actor_cnn = nn.Identity()

            mlp_args = {
                "input_size": actor_in_dim,
                "units": self._actor_units,
                "activation": self._actor_activation,
                "dense_func": nn.Linear,
            }
            self.actor_mlp = self._build_mlp(**mlp_args)

            # init actor_mlp weights the same way as original MLP
            mlp_init = self.init_factory.create(**self._actor_initializer)
            for m in self.actor_mlp.modules():
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)

        def load(self, params):
            """
            Load hyperparameters from the config dict.

            rl_games calls this during initialization (inside super().__init__).
            """
            super().load(params)

            # Discriminator config
            self._disc_units = params['disc']['units']
            self._disc_activation = params['disc']['activation']
            self._disc_initializer = params['disc']['initializer']

            # Actor MLP config (used for both actor trunk and conditioner activation/init)
            self._actor_units = params['mlp']['units']
            self._actor_activation = params['mlp']['activation']
            self._actor_initializer = params['mlp']['initializer']
            return

        def _build_film_cond(self):
            """
            Build a conditioner network that outputs FiLM parameters for all actor hidden layers.

            Conditioner pipeline:
                cond (11) -> cond_mlp -> cond_linear -> cond_out

            cond_out is shaped [B, sum_i 2*h_i], where h_i are actor hidden layer widths.
            For each hidden layer i:
                gamma_i = cond_out[..., pos:pos+h_i]
                beta_i  = cond_out[..., pos+h_i:pos+2*h_i]
            """
            cond_mlp_args = {
                'input_size' : 11, 
                'units' : [64, 64], 
                'activation' : self._actor_activation, 
                'dense_func' : torch.nn.Linear
            }
            self.cond_mlp = self._build_mlp(**cond_mlp_args)
            
            film_out_size = sum(2 * u for u in self._actor_units)
            self.cond_linear = torch.nn.Linear(cond_mlp_args['units'][-1], film_out_size)

            mlp_init = self.init_factory.create(**self._actor_initializer)
            for m in list(self.cond_mlp.modules()) + [self.cond_linear]:
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias) 

            return

        def _split_film_params(self, cond_out: torch.Tensor):
            """
            Split concatenated FiLM parameters into per-layer (gamma, beta) tuples.

            Parameters
            ----------
            cond_out : torch.Tensor
                Shape [B, sum_i 2*h_i], where h_i are widths of actor hidden layers.

            Returns
            -------
            list[tuple[torch.Tensor, torch.Tensor]]
                A list of (gamma_i, beta_i), each shaped [B, h_i].
            """
            film = []
            pos = 0
            for h in self._actor_units:
                h = int(h)
                gamma = cond_out[:, pos:pos + h]
                beta  = cond_out[:, pos + h:pos + 2 * h]
                film.append((gamma, beta))
                pos += 2 * h
            return film


        def _forward_mlp_with_film(self, mlp: nn.Sequential, x: torch.Tensor, film_params):
            """
            Forward an MLP while applying FiLM exactly once per Linear-block.

            Definition of "Linear-block"
            ----------------------------
            A block begins at a Linear layer and includes subsequent non-linear modules
            (e.g., activation, dropout, layernorm) up to (but excluding) the next Linear.

            We apply FiLM at the *end* of each block to avoid repeated modulation when
            there are multiple non-linear modules after a Linear.

            Parameters
            ----------
            mlp : nn.Sequential
                The actor MLP as a Sequential container.
            x : torch.Tensor
                Input features [B, D].
            film_params : list[(gamma, beta)]
                Per-hidden-layer FiLM params. Length must match number of Linear layers in `mlp`.

            Returns
            -------
            torch.Tensor
                The modulated output features after the MLP.
            """
            mods = list(mlp)
            lin_idx = -1
            pending_film = False

            for i, layer in enumerate(mods):
                x = layer(x)

                if isinstance(layer, nn.Linear):
                    lin_idx += 1
                    pending_film = True

                # apply FiLM at the end of this Linear-block:
                # - end of Sequential, or
                # - next layer starts a new block (next is Linear)
                next_is_linear = (i + 1 < len(mods)) and isinstance(mods[i + 1], nn.Linear)
                if pending_film and (i == len(mods) - 1 or next_is_linear):
                    gamma, beta = film_params[lin_idx]
                    x = x * gamma + beta
                    pending_film = False

            return x

        def forward(self, obs_dict):
            """
            rl_games forward: returns (policy outputs..., value, rnn_states).

            obs_dict keys:
              - 'obs': [B, obs_dim]
              - optionally 'rnn_states'
            """
            obs = obs_dict['obs']
            states = obs_dict.get('rnn_states', None)

            actor_outputs = self.eval_actor(obs)
            value = self.eval_critic(obs)

            output = actor_outputs + (value, states)

            return output

        def eval_actor(self, obs):
            """
            Actor forward pass with FiLM conditioning.

            - Actor trunk consumes only obs[:, :574]
            - Conditioner consumes obs[:, 574:] (11 dims)
            """

            # humanoid_obs = obs[:, :358]
            # task_obs = obs[:, 358:574]
            state_obs = obs[:, :574]
            gender_betas = obs[:, 574:]

            a_out = self.actor_cnn(state_obs)
            a_out = a_out.contiguous().view(a_out.size(0), -1)

            cond_out = self.cond_linear(self.cond_mlp(gender_betas))   # [B, sum 2*h_i]
            film_params = self._split_film_params(cond_out)           # [(gamma_i, beta_i), ...]
            a_out = self._forward_mlp_with_film(self.actor_mlp, a_out, film_params)

            if self.is_discrete:
                logits = self.logits(a_out)
                return logits

            if self.is_multi_discrete:
                logits = [logit(a_out) for logit in self.logits]
                return logits

            if self.is_continuous:
                mu = self.mu_act(self.mu(a_out))
                if self.space_config['fixed_sigma']:
                    sigma = mu * 0.0 + self.sigma_act(self.sigma)
                else:
                    sigma = self.sigma_act(self.sigma(a_out))

                return mu, sigma
            return

        def eval_critic(self, obs):
            """
            Critic forward pass.

            Note: currently uses the original critic path, which (by default) consumes
            the full observation. If you want symmetry (critic also excludes cond dims),
            you can rebuild critic trunk similarly.
            """
            c_out = self.critic_cnn(obs)
            c_out = c_out.contiguous().view(c_out.size(0), -1)
            c_out = self.critic_mlp(c_out)              
            value = self.value_act(self.value(c_out))
            return value

        def eval_disc(self, amp_obs):
            """Compute discriminator logits for AMP observations."""
            disc_mlp_out = self._disc_mlp(amp_obs)
            disc_logits = self._disc_logits(disc_mlp_out)
            return disc_logits

        def get_disc_logit_weights(self):
            """Return flattened weights of the discriminator logit head (useful for regularization/monitoring)."""
            return torch.flatten(self._disc_logits.weight)

        def get_disc_weights(self):
            """Return flattened weights of discriminator MLP + logit head."""
            weights = []
            for m in self._disc_mlp.modules():
                if isinstance(m, nn.Linear):
                    weights.append(torch.flatten(m.weight))

            weights.append(torch.flatten(self._disc_logits.weight))
            return weights

        def _build_disc(self, input_shape):
            """
            Build discriminator MLP and a single-unit logit head.

            Parameters
            ----------
            input_shape : tuple
                Expected discriminator input shape; we use input_shape[0] as feature dim.
            """
            self._disc_mlp = nn.Sequential()

            mlp_args = {
                'input_size' : input_shape[0], 
                'units' : self._disc_units, 
                'activation' : self._disc_activation, 
                'dense_func' : torch.nn.Linear
            }
            self._disc_mlp = self._build_mlp(**mlp_args)
            
            mlp_out_size = self._disc_units[-1]
            self._disc_logits = torch.nn.Linear(mlp_out_size, 1)

            mlp_init = self.init_factory.create(**self._disc_initializer)
            for m in self._disc_mlp.modules():
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias) 

            torch.nn.init.uniform_(self._disc_logits.weight, -DISC_LOGIT_INIT_SCALE, DISC_LOGIT_INIT_SCALE)
            torch.nn.init.zeros_(self._disc_logits.bias) 

            return

    def build(self, name, **kwargs):
        """rl_games entry point: construct and return the Network instance."""
        net = PHCBuilder.Network(self.params, **kwargs)
        return net