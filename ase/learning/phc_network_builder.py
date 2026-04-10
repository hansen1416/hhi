"""
PHC network builder with morphology-conditioned FiLM modulation.

High-level structure
--------------------
This file defines the neural network used by rl_games for PHC training.

The model has three logical parts:

1. Actor (policy)
   - Takes the environment observation.
   - Uses only the first 934 dims as the main motion/task/state input.
   - Uses the last 11 dims (gender + 10 betas) as a morphology condition.
   - Applies FiLM modulation to the actor hidden features:
         h <- h * gamma(cond) + beta(cond)

2. Critic (value function)
   - Estimates V(s) for PPO.
   - Currently unchanged: it consumes the full observation directly.
   - No morphology conditioning is applied yet.

3. Discriminator
   - Used by AMP / ASE-style adversarial training.
   - Input is AMP observation features (motion window features).
   - Also conditioned on the same 11-D morphology code via FiLM.
   - Outputs one scalar logit: "real/demo-like" vs "fake/agent-like".

Important observation layout
----------------------------
Environment observation shape:
    obs: [B, 945]

Split:
    obs[:, :934]   -> state/task features for actor trunk
    obs[:, 934:]   -> morphology condition = [gender, beta_1, ..., beta_10]

Important AMP layout
--------------------
AMP observation:
    amp_obs: [B, amp_dim] or [T, N, amp_dim]

Morphology condition for AMP:
    amp_shape: [B, 11] or [T, N, 11]

The discriminator FiLM implementation is written to work on the LAST dimension,
so it supports both 2D [B, D] and 3D [T, N, D] inputs.
"""
from rl_games.common import object_factory
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import layers
# from rl_games.algos_torch import network_builder
import learning.rl_games_network_builder as network_builder

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
        Actor-Critic + Discriminator network.

        Summary of components
        ---------------------
        Actor:
            - input: obs[:, :934] (obs: [B, 945])
            - condition: obs[:, 934:] (11 dims)
            - structure: actor MLP + FiLM modulation from morphology

        Critic:
            - input: full observation
            - structure: standard critic MLP from rl_games

        Discriminator:
            - input: AMP motion features
            - condition: 11-D morphology code
            - structure: discriminator MLP + FiLM modulation + scalar logit head

        Why FiLM?
        ---------
        FiLM lets morphology influence hidden features multiplicatively/additively,
        without simply concatenating shape to the raw motion/state input.
        This keeps the main representation motion-centric while allowing the
        network to adapt its internal processing according to body shape.
        """

        def __init__(self, params, **kwargs):
            """
            Build the full model.

            Steps:
            1. Let the base rl_games class build the default actor/critic parts.
            2. Optionally create fixed sigma for continuous policies.
            3. Build the discriminator for AMP.
            4. Rebuild the actor trunk so it consumes only 934 dims, not full obs.
            5. Build the actor FiLM conditioner.
            """
            # Calls self.load(params) internally, making actor/critic configs available.
            super().__init__(params, **kwargs)

            # For continuous control, rl_games may either:
            # - learn sigma from a network head, or
            # - keep sigma fixed as a parameter.
            #
            # This branch handles the fixed-sigma case.
            if self.is_continuous and not self.space_config["learn_sigma"]:
                actions_num = kwargs.get("actions_num")
                sigma_init = self.init_factory.create(**self.space_config["sigma_init"])
                self.sigma = nn.Parameter(
                    torch.zeros(actions_num, requires_grad=False, dtype=torch.float32),
                    requires_grad=False,
                )
                sigma_init(self.sigma)

            self._shape_dim = 11

            # (2920,) This is the motion feature dimension seen by the discriminator.
            amp_input_shape = kwargs.get("amp_input_shape")
            # Build discriminator:
            #   amp_obs -> disc_mlp -> disc_logits
            # and also its FiLM conditioner:
            #   amp_shape -> disc_cond_mlp -> disc_cond_linear -> gamma/beta
            self._build_disc(amp_input_shape)

            # Replace the default actor MLP with a new actor MLP that takes only
            # the first 934 dims of the observation. The last 11 dims are handled
            # separately by the FiLM conditioner.
            self._rebuild_actor_trunk(actor_in_dim=934)

            # Build actor-side morphology conditioner:
            #   shape(11) -> cond_mlp -> cond_linear -> gamma/beta for actor layers
            self._build_film_cond()

            self._rebuild_critic_trunk(critic_in_dim=934)
            self._build_critic_film_cond()

        def load(self, params):
            """
            Read hyperparameters from config.

            The base rl_games class loads common settings. Here we additionally
            cache the actor/discriminator-specific settings for later network
            construction.
            """
            super().load(params)

            # Discriminator hyperparameters.
            self._disc_hidden_units = params["disc"]["units"]
            self._disc_activation = params["disc"]["activation"]
            self._disc_initializer = params["disc"]["initializer"]

            # Actor hyperparameters.
            # These are used for:
            # - the actor MLP trunk
            # - the actor FiLM conditioner
            self._actor_hidden_units = params["mlp"]["units"]
            self._actor_activation = params["mlp"]["activation"]
            self._actor_initializer = params["mlp"]["initializer"]

            self._critic_hidden_units = params["mlp"]["units"]
            self._critic_activation = params["mlp"]["activation"]
            self._critic_initializer = params["mlp"]["initializer"]

        def _rebuild_actor_trunk(self, actor_in_dim):
            """
            Rebuild actor MLP so that it only consumes the motion/task/state part
            of the observation, not the shape condition.

            Input:
                actor_in_dim = 934

            Output:
                actor_mlp: standard MLP with hidden sizes from config

            Why rebuild?
            ------------
            The default rl_games actor would consume the full observation.
            Here we deliberately exclude the 11 morphology dims from the actor
            trunk, because we want morphology to enter only through FiLM.
            """
            # Save input size so the actor forward knows where to split obs.
            self._actor_in_dim = actor_in_dim  # Store for eval_actor slicing.

            # PHC typically uses vector observations, not images, so CNN is identity.
            self.actor_cnn = nn.Identity()

            # Build MLP with original hidden units/activation but new input size.
            mlp_args = {
                "input_size": actor_in_dim,
                "units": self._actor_hidden_units,
                "activation": self._actor_activation,
                "dense_func": nn.Linear,
            }
            self.actor_mlp = self._build_mlp(**mlp_args)

            # Initialize actor trunk weights with the configured initializer.
            mlp_init = self.init_factory.create(**self._actor_initializer)
            for m in self.actor_mlp.modules():
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)

        def _build_film_cond(self):
            """
            Build actor-side FiLM conditioner.

            Network:
                shape(11)
                  -> cond_mlp (64, 64)
                  -> cond_linear
                  -> flat FiLM parameter vector

            The output stores all gamma/beta values for all actor hidden layers.

            Example:
                actor hidden units = [1024, 512]

                film_out_size =
                    2*1024 + 2*512
                  = 3072

                This 3072-D vector is later split into:
                    layer1 gamma [1024]
                    layer1 beta  [1024]
                    layer2 gamma [512]
                    layer2 beta  [512]
            """
            cond_mlp_args = {
                "input_size": 11,  # gender + 10 betas
                "units": [64, 64],  # Simple two-layer conditioner
                "activation": self._actor_activation,
                "dense_func": torch.nn.Linear,
            }
            self.cond_mlp = self._build_mlp(**cond_mlp_args)

            # One gamma and one beta per hidden unit in each actor hidden layer.
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
            """
            Split actor FiLM output into per-layer (gamma, beta) pairs.

            Input:
                cond_out: [B, total_film_dim]
                    Example: [B, 3072]

            Output:
                film_params = [
                    (gamma_1, beta_1),   # for actor hidden layer 1
                    (gamma_2, beta_2),   # for actor hidden layer 2
                    ...
                ]

            Notes
            -----
            This version is for the actor path, which currently operates on
            2D tensors [B, D].

            If you later want actor FiLM to support sequence-shaped input like
            [T, N, D], you would change `cond_out[:, ...]` to `cond_out[..., ...]`,
            similar to the discriminator helper below.
            """
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
            """
            Forward an MLP while applying FiLM once per Linear block.

            Conceptually:
                x -> Linear -> Activation -> FiLM -> Linear -> Activation -> FiLM ...

            Why "block" instead of "every layer"?
            ------------------------------------
            The MLP returned by `_build_mlp(...)` is typically a sequence like:
                Linear -> Activation -> Linear -> Activation -> ...

            We apply one (gamma, beta) pair after the nonlinearity of each hidden
            block, not after every module object individually.

            Inputs
            ------
            mlp:
                A Sequential MLP built by `_build_mlp(...)`

            x:
                Input feature tensor.
                Can be:
                    [B, D]     during standard training
                    [T, N, D]  during rollout-time AMP reward computation

            film_params:
                List of (gamma, beta), one pair for each hidden block.
                Each gamma/beta matches the feature size of that hidden block.

            Important detail
            ----------------
            Broadcasting works because gamma/beta are shaped on the LAST dimension.
            So this function naturally supports both 2D and 3D inputs.
            """
            modules = list(mlp)
            film_idx = 0  # Tracks current FiLM param pair.

            for i, layer in enumerate(modules):
                x = layer(x)

                # Start pending FiLM after each Linear.
                if isinstance(layer, nn.Linear):
                    pending_film = True
                else:
                    pending_film = pending_film if 'pending_film' in locals() else False

                # We consider the block finished when:
                # - this is the last module, or
                # - the next module starts a new block (another Linear)
                is_end_of_block = (i == len(modules) - 1) or isinstance(modules[i + 1], nn.Linear)
                if pending_film and is_end_of_block:
                    gamma, beta = film_params[film_idx]
                    x = x * gamma + beta
                    film_idx += 1
                    pending_film = False

            return x

        def _rebuild_critic_trunk(self, critic_in_dim):
            self._critic_in_dim = critic_in_dim

            self.critic_cnn = nn.Identity()

            mlp_args = {
                "input_size": critic_in_dim,
                "units": self._critic_hidden_units,
                "activation": self._critic_activation,
                "dense_func": nn.Linear,
            }
            self.critic_mlp = self._build_mlp(**mlp_args)

            mlp_init = self.init_factory.create(**self._critic_initializer)
            for m in self.critic_mlp.modules():
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)


        def _build_critic_film_cond(self):
            cond_mlp_args = {
                "input_size": self._shape_dim,
                "units": [64, 64],
                "activation": self._critic_activation,
                "dense_func": torch.nn.Linear,
            }
            self.critic_cond_mlp = self._build_mlp(**cond_mlp_args)

            film_out_size = sum(2 * u for u in self._critic_hidden_units)
            self.critic_cond_linear = torch.nn.Linear(
                cond_mlp_args["units"][-1], film_out_size
            )

            mlp_init = self.init_factory.create(**self._critic_initializer)
            for m in list(self.critic_cond_mlp.modules()) + [self.critic_cond_linear]:
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)

        def forward(self, obs_dict):
            """
            Standard rl_games entry point.

            This method is used for normal actor-critic inference:
                obs -> actor outputs + value + rnn_states

            It does NOT compute discriminator outputs here.
            The discriminator is called separately through `eval_disc(...)`
            in the AMP training code.
            """
            obs = obs_dict["obs"]
            states = obs_dict.get("rnn_states", None)

            actor_outputs = self.eval_actor(obs)
            value = self.eval_critic(obs)

            # rl_games expects:
            #   discrete: logits, value, rnn_states
            #   continuous: mu, sigma, value, rnn_states
            return actor_outputs + (value, states)

        def eval_actor(self, obs):
            """
            Actor forward pass with morphology-conditioned FiLM.

            Steps
            -----
            1. Split observation into:
                - state/task input      [B, 934]
                - morphology condition  [B, 11]

            2. Run state/task input through actor trunk.

            3. Run morphology condition through conditioner network to get
               FiLM parameters (gamma, beta) for each actor hidden layer.

            4. Apply FiLM-modulated forward through actor MLP.

            5. Produce action distribution parameters:
                - discrete: logits
                - continuous: mu and sigma

            Output
            ------
            For continuous control:
                mu, sigma
            """
            state_obs = obs[:, :934]  # State/task features.
            gender_betas = obs[:, 934:]  # Condition (11 dims).

            a_out = self.actor_cnn(state_obs)
            a_out = a_out.contiguous().view(a_out.size(0), -1)

            # Convert morphology code into FiLM parameters.
            cond_out = self.cond_linear(self.cond_mlp(gender_betas))
            film_params = self._split_film_params(cond_out)

            # Forward actor trunk with layer-wise FiLM modulation.
            a_out = self._forward_mlp_with_film(self.actor_mlp, a_out, film_params)

            # Standard rl_games action heads.
            if self.is_discrete:
                return self.logits(a_out)

            if self.is_multi_discrete:
                return [logit(a_out) for logit in self.logits]

            if self.is_continuous:
                mu = self.mu_act(self.mu(a_out))
                # Fixed sigma: one global learned/fixed parameter vector
                # independent of current observation.
                if self.space_config["fixed_sigma"]:
                    sigma = mu * 0.0 + self.sigma_act(self.sigma)
                else:
                    sigma = self.sigma_act(self.sigma(a_out))
                return mu, sigma

        def eval_critic(self, obs):
            """
            Critic forward pass.

            Current design
            --------------
            The critic remains unchanged from the standard rl_games pattern:
                full obs -> critic_cnn -> critic_mlp -> value head

            So unlike the actor, the critic currently:
            - consumes all 945 dims directly
            - does not explicitly separate morphology
            - does not use FiLM conditioning

            Purpose
            -------
            The critic estimates V(s), which PPO uses for:
            - bootstrap targets
            - advantage estimation
            - value loss

            Future extension
            ----------------
            You could later make the critic morphology-aware too, by applying
            the same split-and-FiLM pattern as the actor.
            """
            state_obs = obs[:, :934]
            gender_betas = obs[:, 934:]

            c_out = self.critic_cnn(state_obs)
            c_out = c_out.contiguous().view(c_out.size(0), -1)

            cond_out = self.critic_cond_linear(self.critic_cond_mlp(gender_betas))
            film_params = self._split_film_params_by_units(cond_out, self._critic_hidden_units)

            c_out = self._forward_mlp_with_film(self.critic_mlp, c_out, film_params)

            return self.value_act(self.value(c_out))

        def get_disc_logit_weights(self):
            """
            Return flattened final discriminator head weights.

            Useful when the training code wants to regularize or inspect only
            the final classifier layer of the discriminator.
            """
            return torch.flatten(self._disc_logits.weight)

        def get_disc_weights(self):
            """
            Return flattened discriminator weights.

            Includes:
            - all Linear weights in discriminator MLP
            - final discriminator logit head weight

            This is commonly used for weight regularization terms.
            """
            weights = [torch.flatten(m.weight) for m in self._disc_mlp.modules() if isinstance(m, nn.Linear)]
            weights.append(torch.flatten(self._disc_logits.weight))
            return weights
        
        def _split_film_params_by_units(self, cond_out, hidden_units):
            """
            Generic FiLM splitter for arbitrary hidden layer sizes.

            This version is used by the discriminator path.

            Why separate from `_split_film_params(...)`?
            --------------------------------------------
            The actor uses `self._actor_hidden_units`, while the discriminator
            uses `self._disc_hidden_units`. This helper works for any list of
            hidden sizes.

            Very important implementation detail
            ------------------------------------
            We slice on the LAST dimension using `...`:
                cond_out[..., start:end]

            This is necessary because discriminator evaluation may happen on:
                [B, D]     during minibatch training
                [T, N, D]  during rollout reward computation

            Example
            -------
            If:
                cond_out.shape = [4, 16, 3072]
                hidden_units = [1024, 512]

            Then output is:
                layer1 gamma: [4, 16, 1024]
                layer1 beta : [4, 16, 1024]
                layer2 gamma: [4, 16, 512]
                layer2 beta : [4, 16, 512]
            """

            film_params = []
            pos = 0
            for h in hidden_units:
                h = int(h)
                # gamma = cond_out[:, pos : pos + h]
                # beta  = cond_out[:, pos + h : pos + 2 * h]
                gamma = cond_out[..., pos : pos + h]
                beta  = cond_out[..., pos + h : pos + 2 * h]
                film_params.append((gamma, beta))
                pos += 2 * h

            # print(film_params[0][0].shape)
            # # torch.Size([4, 16, 1024])
            # print(film_params[0][1].shape)
            # # torch.Size([4, 16, 1024])
            # print(pos)
            # # 3072
            # exit()
            
            return film_params

        def _build_disc_film_cond(self):
            """
            Build discriminator-side FiLM conditioner.

            Network:
                amp_shape(11)
                  -> disc_cond_mlp (64, 64)
                  -> disc_cond_linear
                  -> flat discriminator gamma/beta vector

            This is analogous to `_build_film_cond()` for the actor, but uses
            discriminator hyperparameters and outputs FiLM parameters sized for
            discriminator hidden layers.
            """
            cond_mlp_args = {
                "input_size": self._shape_dim,
                "units": [64, 64],
                "activation": self._disc_activation,
                "dense_func": torch.nn.Linear,
            }
            self.disc_cond_mlp = self._build_mlp(**cond_mlp_args)

            film_out_size = sum(2 * u for u in self._disc_hidden_units)
            self.disc_cond_linear = torch.nn.Linear(
                cond_mlp_args["units"][-1], film_out_size
            )
            # Initialize discriminator conditioner.
            mlp_init = self.init_factory.create(**self._disc_initializer)
            for m in list(self.disc_cond_mlp.modules()) + [self.disc_cond_linear]:
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)

        def _build_disc(self, input_shape):
            """
            Build discriminator network.

            Input
            -----
            input_shape[0] = AMP feature dimension, e.g. 2920

            Structure
            ---------
            amp_obs
              -> disc_mlp
              -> disc_logits (scalar)

            Plus a separate morphology conditioner:
            amp_shape
              -> disc_cond_mlp
              -> disc_cond_linear
              -> gamma/beta for disc hidden layers

            Important design choice
            -----------------------
            We do NOT concatenate amp_shape to amp_obs directly.
            Instead:
                - amp_obs remains the main motion feature input
                - amp_shape modulates hidden activations via FiLM
            """
            # keep discriminator input as motion-only AMP features
            mlp_args = {
                "input_size": input_shape[0],
                "units": self._disc_hidden_units,
                "activation": self._disc_activation,
                "dense_func": torch.nn.Linear,
            }
            self._disc_mlp = self._build_mlp(**mlp_args)

            out_size = self._disc_hidden_units[-1]
            self._disc_logits = torch.nn.Linear(out_size, 1)

            mlp_init = self.init_factory.create(**self._disc_initializer)
            for m in self._disc_mlp.modules():
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)

            # Initialize logit layer with small uniform range.
            torch.nn.init.uniform_(self._disc_logits.weight, -DISC_LOGIT_INIT_SCALE, DISC_LOGIT_INIT_SCALE)
            torch.nn.init.zeros_(self._disc_logits.bias)

            # Build discriminator morphology conditioner.
            self._build_disc_film_cond()

        def eval_disc(self, amp_obs, amp_shape):
            """
            Discriminator forward pass with morphology-conditioned FiLM.

            Inputs
            ------
            amp_obs:
                AMP motion features
                shape:
                    [B, amp_dim]    during minibatch training
                    [T, N, amp_dim] during rollout reward computation

            amp_shape:
                morphology condition
                shape:
                    [B, 11]    or
                    [T, N, 11]

            Steps
            -----
            1. Convert morphology code into discriminator FiLM parameters.
            2. Run AMP motion features through discriminator trunk with FiLM.
            3. Project final hidden features to a single discriminator logit.

            Output
            ------
            disc_logits:
                [B, 1] or [T, N, 1]

            Interpretation
            --------------
            Higher logit means "more demo-like / more real".
            Lower logit means "more agent-like / more fake".
            """

            cond_out = self.disc_cond_linear(self.disc_cond_mlp(amp_shape))
            disc_film_params = self._split_film_params_by_units(cond_out, self._disc_hidden_units)

            disc_out = self._forward_mlp_with_film(self._disc_mlp, amp_obs, disc_film_params)

            return self._disc_logits(disc_out)

    def build(self, name, **kwargs):
        """
        rl_games entry point.

        rl_games calls this builder to construct the actual neural network object.
        """
        return PHCBuilder.Network(self.params, **kwargs)