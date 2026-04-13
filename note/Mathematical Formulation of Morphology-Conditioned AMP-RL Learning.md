Below is the **mathematical formulation of the learning process** corresponding to your pipeline.
I separate it into **environment dynamics, policy learning, value learning, and adversarial motion learning**.

---

# 1. Morphology-Conditioned MDP

Your system is a **conditional Markov Decision Process (CMDP)**.

State
[
s_t \in \mathcal{S}
]

Morphology condition
[
m \in \mathcal{M}
]

where

[
m = [\text{gender}, \beta_1,\dots,\beta_{10}]
]

Action

[
a_t \in \mathcal{A}
]

Transition dynamics

[
s_{t+1} \sim P(s_{t+1} \mid s_t, a_t, m)
]

The policy is conditioned on morphology:

[
\pi_\theta(a_t \mid s_t, m)
]

The trajectory distribution induced by the policy is

[
\tau = (s_0,a_0,s_1,a_1,\dots)
]

[
p_\theta(\tau|m)
================

p(s_0|m)
\prod_{t=0}^{T}
\pi_\theta(a_t|s_t,m)
P(s_{t+1}|s_t,a_t,m)
]

---

# 2. Learning Objective

The objective is to maximize the expected discounted return:

[
J(\theta)
=========

\mathbb{E}*{m\sim p(m)}
\mathbb{E}*{\tau\sim p_\theta(\tau|m)}
\left[
\sum_{t=0}^{\infty}\gamma^t r_t
\right]
]

The reward contains two parts:

[
r_t = r_{\text{task}}(s_t,a_t,m) + r_{\text{AMP}}(x_t,m)
]

where

(x_t) = AMP motion features.

---

# 3. Value Function Learning

The critic learns the morphology-conditioned value function:

[
V_\phi(s_t,m)
=============

\mathbb{E}*{\pi*\theta}
\left[
\sum_{k=0}^{\infty}\gamma^k r_{t+k}
\mid s_t,m
\right]
]

---

## Bellman equation

[
V_\phi(s_t,m)
=============

\mathbb{E}*{a_t\sim\pi*\theta}
\left[
r_t
+
\gamma
\mathbb{E}*{s*{t+1}}V_\phi(s_{t+1},m)
\right]
]

---

## Critic training target

One-step bootstrap:

[
y_t
===

r_t
+
\gamma V_\phi(s_{t+1},m)
]

Critic loss

[
L_V(\phi)
=========

\mathbb{E}
\left[
(V_\phi(s_t,m) - y_t)^2
\right]
]

---

# 4. Advantage Estimation

Advantage measures whether an action was better than expected.

[
A_t
===

## R_t

V_\phi(s_t,m)
]

where

[
R_t
===

\sum_{k=0}^{K-1}\gamma^k r_{t+k}
+
\gamma^K V_\phi(s_{t+K},m)
]

In practice PPO uses **Generalized Advantage Estimation (GAE)**:

[
\delta_t
========

r_t
+
\gamma V_\phi(s_{t+1},m)
------------------------

V_\phi(s_t,m)
]

[
A_t
===

\sum_{l=0}^{\infty}
(\gamma\lambda)^l
\delta_{t+l}
]

---

# 5. PPO Actor Objective

Policy ratio:

[
r_t(\theta)
===========

\frac{\pi_\theta(a_t|s_t,m)}
{\pi_{\theta_{\text{old}}}(a_t|s_t,m)}
]

Clipped objective:

[
L_{\text{PPO}}(\theta)
======================

\mathbb{E}
\left[
\min
\left(
r_t(\theta)A_t,
\text{clip}(r_t(\theta),1-\epsilon,1+\epsilon)A_t
\right)
\right]
]

This objective updates the actor.

---

# 6. Adversarial Motion Learning (AMP)

AMP introduces a discriminator that learns to distinguish **real vs generated motion windows**.

Motion feature window

[
x \in \mathcal{X}
]

---

## Discriminator

[
D_\psi(x,m)
]

It outputs the probability that motion (x) comes from the **real dataset**.

---

### Real motion distribution

[
x \sim p_{\text{data}}(x|m)
]

### Policy motion distribution

[
x \sim p_\theta(x|m)
]

---

## Discriminator objective

Binary classification loss:

[
L_D(\psi)
=========

*

\mathbb{E}*{x\sim p*{\text{data}}(x|m)}
[\log D_\psi(x,m)]
------------------

\mathbb{E}*{x\sim p*\theta(x|m)}
[\log(1-D_\psi(x,m))]
]

---

# 7. AMP Reward

The discriminator output is converted into a **style reward**.

Typical AMP reward:

[
r_{\text{AMP}}(x,m)
===================

-\log(1-D_\psi(x,m))
]

or equivalently

[
r_{\text{AMP}}(x,m)
===================

\log D_\psi(x,m)
]

depending on implementation.

This encourages policy motions that look **indistinguishable from real motion**.

---

# 8. Replay Buffer Distribution

Policy motion samples come from two sources.

Current rollout:

[
x \sim p_{\pi}(x|m)
]

Replay buffer:

[
x \sim p_{\text{replay}}(x|m)
]

The discriminator is trained on the mixture:

[
p_{\text{fake}}(x|m)
====================

\alpha p_\pi(x|m)
+
(1-\alpha)p_{\text{replay}}(x|m)
]

---

# 9. Full Training Objective

The final learning problem is a **three-player optimization**.

Actor

[
\max_\theta
\mathbb{E}*{\pi*\theta}
\left[
\sum_t \gamma^t
(r_{\text{task}} + r_{\text{AMP}})
\right]
]

Critic

[
\min_\phi L_V(\phi)
]

Discriminator

[
\min_\psi L_D(\psi)
]

---

# 10. What the Model Ultimately Learns

The system learns a **shape-conditioned physical motion distribution**

[
p_\theta(x|m)
]

that matches the real motion distribution

[
p_{\text{data}}(x|m)
]

while satisfying physics constraints through the environment dynamics.

So the learning goal can be interpreted as:

[
p_\theta(x|m)
\approx
p_{\text{data}}(x|m)
]

subject to

[
x \text{ is physically valid motion}.
]

---

# Final Interpretation

Your project is learning a **conditional physical motion generator**:

[
\pi_\theta(a|s,m)
]

such that the induced motion distribution

[
p_\theta(x|m)
]

matches the real motion distribution

[
p_{\text{data}}(x|m)
]

under physics simulation.

---

If you want, I can also write the **single compact equation that summarizes the entire algorithm**, which is often how **AMP/ASE papers present the method in one line**.
