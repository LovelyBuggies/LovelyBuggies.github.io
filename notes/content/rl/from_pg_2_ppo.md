---
date: 2025-03-10
title: "From PG to PPO"
math: true
weight: 1
postType: notes
readingTime: 25
linkTitle: "From Policy Gradient to PPO"
---

{{< katex />}}

# From Policy Gradient to PPO
{{< postbadges >}}

## Policy Gradient

Compared with value-based methods (Q-learning), Policy-based methods aim directly at learning the parameterized policy that can select actions without consulting a value function. Policy gradient (PG) methods seek to maximize a performance measure {{< katex >}}J(\theta){{< /katex >}} with the policy’s parameter {{< katex >}}\theta{{< /katex >}}, where the updates approximate gradient ascent in {{< katex >}}J{{< /katex >}}.

{{< katex display=true >}}
\label{eq:pg}
\theta^{(i+1)} \leftarrow \theta^{(i)} + \alpha\nabla J(\theta^{(i)}).
{{< /katex >}}

There are 2 main advantages of PG methods,

- Approximating policy can approach a deterministic policy, whereas {{< katex >}}\epsilon{{< /katex >}}-greedy always has probability of selecting a random action;

- With continuous policy parameterization, the action probabilities change smoothly as a function of the learned parameter, whereas {{< katex >}}\epsilon{{< /katex >}}-greedy may change dramatically for an arbitrarily small change in the estimated action values.

Since the major intent of this article is to introduce PPO methods from PG, we omit some other forms of PG here.

### PG Theorem

An intuitive way to calculate policy gradient is to replace {{< katex >}}J(\theta){{< /katex >}} with {{< katex >}}V^{\pi_{\theta}} (s_0){{< /katex >}}. However, the calculation is hard as it directly depends on both the action selection and indirectly the distribution of states following the target selection. PG theorem provides a nice reformulation of the derivative of the objective function to not involve the state distribution derivation.
We omit {{< katex >}}\theta{{< /katex >}} in subscripts/superscripts and gradients, assuming {{< katex >}}\pi{{< /katex >}} depends on {{< katex >}}\theta{{< /katex >}} and all gradients are w.r.t. {{< katex >}}\theta{{< /katex >}}; i.e., {{< katex >}}V^{\pi} \equiv V^{\pi_{\theta}}{{< /katex >}}, {{< katex >}}Q^{\pi} \equiv Q^{\pi_{\theta}}{{< /katex >}} and {{< katex >}}\nabla \equiv \nabla_{\theta}{{< /katex >}}.

<div id="them:PG" class="theorem">
**Theorem 1**. *Taking the state-value function as the optimizing target, the objective gradient follows, {{< katex display=true >}}
\label{equ:pgthem}
    \nabla J(\theta) \propto \sum_s d^\pi(s) \sum_a Q^\pi(s,a) \nabla \pi(a|s),
{{< /katex >}}
 where {{< katex >}}d^\pi(s){{< /katex >}} is the stationary distribution of the policy {{< katex >}}\pi_{\theta}{{< /katex >}}.*

To sample with expectation equals or approximates the expression,

{{< katex display=true >}}
\label{equ:pgtheorem-sample}
\nabla J(\theta) \propto \sum_s d^\pi(s)\sum_a Q^\pi(s,a) \, \nabla\pi(a|s) \\
= \mathbb{E}_{d^\pi}\!\left[\sum_a Q^\pi(s,a) \, \nabla\pi(a|s) \right] \\
= \mathbb{E}_{d^\pi}\!\left[\sum_a \pi(a|s) \, Q^\pi(s,a) \, \frac{\nabla\pi(a|s)}{\pi(a|s)} \right] \\
= \mathbb{E}_{\pi}\!\left[Q^\pi(s,a) \, \frac{\nabla\pi(a|s)}{\pi(a|s)} \right] \\
= \mathbb{E}_{\pi}\!\left[Q^\pi(s,a) \, \nabla\ln\pi(a|s) \right] \, .
{{< /katex >}}

</div>

<div class="proof">

<strong>Proof.</strong> The gradient of {{< katex >}}V{{< /katex >}} can be written via {{< katex >}}Q{{< /katex >}} as

{{< katex display=true >}}
\begin{aligned}
\nabla V^{\pi}(s) &= \nabla\!\left[ \sum_a \pi(a\mid s)\, Q^\pi (s,a)\right] \\
&= \sum_a\!\left[ 
  \nabla\pi(a\mid s)\, Q^\pi (s,a) 
  + \pi(a\mid s)\, \nabla Q^\pi (s,a) \right] \\
&= \sum_a\!\left[ 
  \nabla\pi(a\mid s)\, Q^\pi (s,a) 
  + \pi(a\mid s)\, \nabla \sum_{s'} P(s'\mid s,a)\, (r + V^\pi(s')) \right] \\
&\stackrel{\text{(i)}}{=} \sum_a\!\left[ 
  \nabla\pi(a\mid s)\, Q^\pi (s,a) 
  + \pi(a\mid s) \sum_{s'} P(s'\mid s,a)\, \nabla V^\pi(s') \right] ,
\end{aligned}
{{< /katex >}}

where (i) uses that the immediate reward {{< katex >}}r{{< /katex >}} depends only on the environment dynamics (not on parameters).

Let {{< katex >}}\phi(s) = \sum_a \nabla\pi(a\mid s)\, Q^\pi (s,a){{< /katex >}}, and denote by {{< katex >}}\rho^\pi(s \to x, k){{< /katex >}} the probability of reaching {{< katex >}}x{{< /katex >}} from {{< katex >}}s{{< /katex >}} in {{< katex >}}k{{< /katex >}} steps under {{< katex >}}\pi{{< /katex >}} (e.g., {{< katex >}}\rho^\pi(s \to s',1)=\sum_a \pi(a\mid s) P(s'\mid s,a){{< /katex >}}). Unrolling the recursion gives

{{< katex display=true >}}
\begin{aligned}
\nabla V^{\pi}(s) 
&= \phi(s) + \sum_a \pi(a\mid s) \sum_{s'} P(s'\mid s,a)\, \nabla V^\pi(s') \\
&= \phi(s) + \sum_{s'} \rho^\pi(s \to s',1)\, \nabla V^\pi(s') \\
&= \phi(s) + \sum_{s'} \rho^\pi(s \to s',1)\!\left[ \phi(s') + \sum_{s''} \rho^\pi(s' \to s'',1)\, \nabla V^\pi(s'') \right] \\
&= \phi(s) + \sum_{s'} \rho^\pi(s \to s',1)\, \phi(s') + \sum_{s''} \rho^\pi(s \to s'',2)\, \nabla V^\pi(s'') \\
&= \phi(s) + \sum_{s'} \rho^\pi(s \to s',1)\, \phi(s') 
  + \sum_{s''} \rho^\pi(s \to s'',2)\, \phi(s'')
  + \sum_{s'''} \rho^\pi(s \to s''',3)\, \nabla V^\pi(s''') \\
&\quad \vdots \\
&= \sum_{k=0}^{\infty} \sum_x \rho^\pi(s \to x, k)\, \phi(x)\, .
\end{aligned}
{{< /katex >}}

Let {{< katex >}}\eta(s){{< /katex >}} be the expected number of visits to {{< katex >}}s{{< /katex >}} (episodic: {{< katex >}}\sum_s \eta(s){{< /katex >}} is the expected episode length; continuing: {{< katex >}}\sum_s \eta(s)=1{{< /katex >}}). Plugging into {{< katex >}}J{{< /katex >}} yields

{{< katex display=true >}}
\begin{aligned}
\nabla J(\theta) &= \nabla V^\pi(s_0) \\
&= \sum_s \Bigl( \sum_{k=0}^{\infty} \rho^\pi(s_0 \to s, k) \Bigr) \sum_a \nabla\pi(a\mid s)\, Q^\pi(s,a) \\
&= \sum_s \eta(s) \sum_a \nabla\pi(a\mid s)\, Q^\pi(s,a) \\
&\stackrel{\text{norm}}{=} \Bigl(\sum_s \eta(s)\Bigr) \Bigl( \sum_s \frac{\eta(s)}{\sum_s \eta(s)} \Bigr) \sum_a \nabla\pi(a\mid s)\, Q^\pi(s,a) \\
&\propto \sum_s d^\pi(s) \sum_a \nabla\pi(a\mid s)\, Q^\pi(s,a)\, .
\end{aligned}
{{< /katex >}}

</div>

The eligibility vector {{< katex >}}\nabla\ln\pi(a|s){{< /katex >}} is the only place the policy parameterization appears, which can be omitted {{< katex >}}L(\theta)=\mathbb{E}_{\pi}[Q^\pi(s,a)]{{< /katex >}} since it will be automatically recovered when differentiating.

### PG with Baseline

<div id="them:PG-baseline" class="theorem">

**Theorem 2**. *PG theorem can be generalized to include a comparison of the action value to an arbitrary baseline $b(s)$, as long as $b(s)$ does not depend on $a$, and this will reduce the variance while keeping it unbiased. {{< katex display=true >}}
\label{equ:reinforce-baseline}
    \begin{aligned}
        \nabla J(\theta) &\propto \sum_s d^\pi(s)\sum_a (Q^\pi (s,a) -b(s)) \nabla\pi(a|s) \\
        &= \mathbb{E}_{\pi} \left[(Q^\pi(s,a) -b(s)) \nabla\ln\pi(a|s)\right].
    \end{aligned}
{{< /katex >}}
*

</div>

According to the Theorem <a href="#them:PG-baseline" data-reference-type="ref" data-reference="them:PG-baseline">2</a>, the expected return {{< katex >}}Q(s,a){{< /katex >}} in Theorem <a href="#them:PG" data-reference-type="ref" data-reference="them:PG">1</a> can be replaced by {{< katex >}}G{{< /katex >}} (expected return of the full or following trajectory by Monte Carlo), {{< katex >}}A{{< /katex >}} (advantage by Generalized Advantage Estimation or state-value prediction), and {{< katex >}}\delta{{< /katex >}} (TD-residual by critic prediction).

### Off-Policy PG

Off-policy sampling reuses any past episodes, which has a higher efficiency and brings more exploration. To make PG off-policy, we adjust it with an importance weight {{< katex >}}\frac{\pi(a|s)}{\beta(a|s)}{{< /katex >}} to correct the mismatch between behavior and target policies.

{{< katex display=true >}}
\label{equ:pgthem-off-policy}
\nabla J(\theta) = \nabla \Bigl(\sum_s d^\beta(s) \, V^\pi(s)\Bigr) \\
= \nabla\Bigl(\sum_s d^\beta(s) \sum_a \pi(a|s) \, Q^\pi(s,a)\Bigr) \\
= \sum_s d^\beta(s) \sum_a \Bigl(\nabla \pi(a|s) \, Q^\pi(s,a) + \pi(a|s) \, \nabla Q^\pi(s,a)\Bigr) \\
\stackrel{\text{(i)}}{\approx} \sum_s d^\beta(s) \sum_a Q^\pi(s,a) \, \nabla \pi(a|s) \\
= \mathbb{E}_{d^\beta}\!\left[\sum_a \beta(a|s) \, \frac{\pi(a|s)}{\beta(a|s)} \, Q^\pi(s,a) \, \frac{\nabla \pi(a|s)}{\pi(a|s)}\right] \\
= \mathbb{E}_{\beta}\!\left[\frac{\pi(a|s)}{\beta(a|s)} \, Q^\pi(s,a) \, \nabla\ln \pi(a|s)\right] \, .
{{< /katex >}}

where {{< katex >}}d^\beta(s){{< /katex >}} is the stationary distribution of the behavior policy {{< katex >}}\beta{{< /katex >}}, and {{< katex >}}Q^\pi{{< /katex >}} is the Q-function estimated regard to the target policy {{< katex >}}\pi{{< /katex >}}. Because of hard computation in reality (i), we ignore the approximation term {{< katex >}}\nabla Q^\pi(s,a){{< /katex >}}.

## Proximal Policy Optimization (PPO)

In this section, we introduce standard PPO and it variants in different domains.

### Clip-PPO

Schulman et al., 2017 proposed the standard PPO that uses a clipped surrogate objective to ensure the policy updates are small and controlled (proximal). Since the advantage under current policy is intangible, we can use Generalized Advantage Estimation (GAE) of the last policy to estimate {{< katex >}}\hat{A}^{\pi_{\theta_{\text{old}}}}{{< /katex >}} to reduce the variance of policy gradient methods and maintain low bias Schulman et al., 2015.

{{< katex display=true >}}
\label{equ:Clip-PPO}
J^{\text{CLIP}}(\theta) = \mathbb{E}_{\pi_{\theta_{\text{old}}}} \left[ \min \left( \frac{\pi_{\theta}(a|s)}{\pi_{\theta_{\text{old}}}(a|s)} \hat{A}^{\pi_{\theta_{\text{old}}}}(s, a), \text{clip}\!\left(\frac{\pi_{\theta}(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}, 1 - \epsilon, 1 + \epsilon\right) \hat{A}^{\pi_{\theta_{\text{old}}}}(s, a) \right) \right]
{{< /katex >}}

where {{< katex >}}\hat{A}^\text{GAE}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}{{< /katex >}}, {{< katex >}}\delta{{< /katex >}} is the TD error, and {{< katex >}}\lambda{{< /katex >}} is a hyperparameter controlling the trade-off between bias and variance. Note that the clipping could also occur in the value network to stabilize the training process.

The objective function can be augmented with an entropy term to encourage exploration:

{{< katex display=true >}}
\label{equ:PPO}
J^{\text{CLIP+}}(\theta) = \mathbb{E}_{\pi_{\theta_{\text{old}}}} \left[ J^{\text{CLIP}}(\theta) - c \sum_{a} \pi_{\theta}(a|s) \log \pi_{\theta}(a|s) \right]
{{< /katex >}}

<div class="algorithm">

<div class="algorithmic">

**Initialize**: policy parameter {{< katex >}}\theta{{< /katex >}} for actor network {{< katex >}}\pi_{\theta}{{< /katex >}}, parameter {{< katex >}}w{{< /katex >}} for critic network {{< katex >}}V_{w}{{< /katex >}}, replay memory {{< katex >}}\mathcal{D}{{< /katex >}} Generate an episode following policy {{< katex >}}\pi_{\theta_{\text{old}}}{{< /katex >}} and store it into {{< katex >}}\mathcal{D}{{< /katex >}} Estimate reward-to-go {{< katex >}}\hat{R}{{< /katex >}} and {{< katex >}}\hat{A}^{\pi_{\theta_{\text{old}}}}{{< /katex >}} using GAE Compute {{< katex >}}J^{\text{CLIP+}}(\theta){{< /katex >}} for all samples according to Equ. <a href="#equ:PPO" data-reference-type="ref" data-reference="equ:PPO">[equ:PPO]</a> {{< katex >}}w \leftarrow w + \alpha_w \frac{1}{N}\sum_i\nabla_w (V_w(s_i)-\hat{R}(s_i, a_i))^2{{< /katex >}} {{< katex >}}\theta \leftarrow \theta + \alpha_\theta \frac{1}{N}\sum_i \nabla_\theta J^{\text{CLIP+H}}(\theta){{< /katex >}} {{< katex >}}\theta_{\text{old}} \leftarrow \theta{{< /katex >}}

</div>

</div>

### KL-PPO

Another formulation of PPO to improve training stability, so-called Trust Region Policy Optimization (TRPO), enforces a KL divergence constraint on the size of the policy update at each iteration Schulman et al., 2017.

{{< katex display=true >}}
\label{alg:TRPO}
J^{\text{KL}}(\theta) = \mathbb{E}_{\pi_{\theta_{\text{old}}}} \left[ \frac{\pi_{\theta}(a|s)}{\pi_{\theta_{\text{old}}}(a|s)} \hat{A}^{\pi_{\theta_{\text{old}}}}(s, a) - c\, \mathcal{D}_\text{KL}(\pi_{\theta_{\text{old}}} \| \pi_{\theta}) \right] \, .
{{< /katex >}}

where {{< katex >}}\mathcal{D}_\text{KL}(\pi_{\theta_{\text{old}}} \| \pi_{\theta}) = \sum_{a} \pi_{\theta_{\text{old}}}(a | s) \log \frac{\pi_{\theta_{\text{old}}}(a | s)}{\pi_{\theta}(a| s)}{{< /katex >}}.

Sometimes, the KL-penalty can be combined with policy clipping to achieve better performance in practice.

#### Adaptive-KL-PPO

(Schulman et al. 2017) also mentioned Adaptive-KL-PPO, where the KL penalty coefficient is adjusted dynamically. If the policy update is too aggressive {{< katex >}}\left( \mathcal{D}_\text{KL} \gg \mathcal{D}_\text{threshold} \right){{< /katex >}}, {{< katex >}}c{{< /katex >}} is increased to penalize large updates; else if the update is too conservative {{< katex >}}\left( \mathcal{D}_\text{KL} \ll \mathcal{D}_\text{threshold} \right){{< /katex >}}, {{< katex >}}c{{< /katex >}} is decreased to allow larger updates.

### Multi-Agent PPO

In the multi-agent setting, the PPO algorithm can be implemented independently (IPPO) or by a centralized critic (MAPPO). In IPPO, each agent has its own actor and critic and learns independently according to a joint reward Schroeder de Witt et al., 2020. Like IPPO, MAPPO employs weight sharing between agents’ critics, and the advantage in MAPPO is estimated through joint GAE Yu et al., 2022.

{{< katex display=true >}}
\label{equ:MAPPO}
J^\text{IPPO}(\theta_i) = \mathbb{E}_{\pi_{\theta_{i, \text{old}}}} \left[ \min \left( \frac{\pi_{\theta_i}(a|s)}{\pi_{\theta_{i, \text{old}}}(a|s)} \hat{A}^{\pi_{\theta_{i, \text{old}}}}(s, a), \text{clip}(\frac{\pi_{\theta_i}(a|s)}{\pi_{\theta_{i, \text{old}}}(a|s)}, 1 - \epsilon, 1 + \epsilon) \hat{A}^{\pi_{\theta_{i,\text{old}}}}(s, a) \right) \right] \\
J^\text{MAPPO}(\theta_i) = \mathbb{E}_{\pi_{\theta_{\text{old}}}} \left[ \min \left( \frac{\pi_{\theta_i}(a|s)}{\pi_{\theta_{i, \text{old}}}(a|s)} \hat{\boldsymbol{A}}^{\pi_{\theta_{\text{old}}}}(s, a), \text{clip}(\frac{\pi_{\theta_i}(a|s)}{\pi_{\theta_{i, \text{old}}}(a|s)}, 1 - \epsilon, 1 + \epsilon) \hat{\boldsymbol{A}}^{\pi_{\theta_{\text{old}}}}(s, a) \right) \right] \, .
{{< /katex >}}

Note that there are some other instantiations of IPPO, but not all of them are vulnerable to non-convergence issues. The one with full actor critic parameter or information sharing can be regarded as a centralized method. Besides, for cases where a general solution is still intangible even with parameter sharing (e.g. the exclusive game), heterogeneous-agent PPO allows the agents to take turns learning by using others’ information, which can work well with strong assumptions.

{{< hint info >}}
A great example is PettingZoo’s agent cycle and parallel environments.
{{< /hint >}}

### Group Relative Policy Optimization (GRPO)

As DeepSeek has made a splash in the LLM community, the RL method GRPO involved has received a lot of attention (Zhihong Shao 2024). GRPO is a variant of PPO, where the advantage is estimated using group-relative comparisons rather than GAE. This approach eliminates the critic model, which improves the training efficiency and stability. The DeepSeek framework consists of: (i) a frozen *reference model*, which is a stable baseline for computing rewards; (ii) a given *reward model*, responsible for evaluating generated outputs and assigning scores; (iii) a *value model*, which estimates the expected return of a given state to aid in policy optimization; and (iv) a *policy model*, which generates {{< katex >}}|\mathcal{G}|{{< /katex >}} responses and is continuously updated to improve performance based on feedback from the other components. The learning objective for GRPO is:

{{< katex display=true >}}
\small
J^\text{GRPO}(\theta) = \mathbb{E}_{\pi_{\theta_\text{old}}, i \in \mathcal{G}} \left[ \min \left( \frac{\pi_{\theta}(a_{i} | s, \vec{a}_{i})}{\pi_{\theta_\text{old}}(a_{i} | s, \vec{a}_{i})} \hat{A}^\mathcal{G}, \text{clip}\!\left(\frac{\pi_{\theta}(a_{i} | s, \vec{a}_{i})}{\pi_{\theta_\text{old}}(a_{i} | s, \vec{a}_{i})}, 1 - \epsilon, 1 + \epsilon\right) \hat{A}^{\mathcal{G}}\right) - c\,\mathcal{D}_\text{KL}(\pi_\text{ref} \| \pi_{\theta})\right]
{{< /katex >}}

where the advantage {{< katex >}}\hat{A}^\mathcal{G}_i=\frac{r_i-\text{mean}(r)}{\text{std}(r)}{{< /katex >}} is estimated by grouped actions produced at the same state. Also,

{{< katex display=true >}}
\mathcal{D}_\text{KL}(\pi_\text{ref} \| \pi_{\theta}) = \frac{\pi_{\text{ref}}(a_{i} \mid s, \vec{a}_{i})}{\pi_{\theta}(a_{i} \mid s, \vec{a}_{i})} - \ln \frac{\pi_{\text{ref}}(a_{i} \mid s, \vec{a}_{i})}{\pi_{\theta}(a_{i} \mid s, \vec{a}_i)} - 1
{{< /katex >}}

is a positive unbiased estimator, which measures the difference between the policy of trained model and reference model (like direct policy optimization).

![GRPO](/imgs/blog/from_pg_2_ppo/grpo.png)



<!-- footnotes converted to hints above -->
<!-- moved to root content -->
<!-- moved back under rl/ -->
