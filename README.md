# Code base for NIPS 2025 Poster Paper Efficient Utility-Preserving Machine Unlearning with Implicit Gradient Surgery (EUPMU)

Paper link: [https://arxiv.org/abs/2510.22124](https://arxiv.org/abs/2510.22124)

Abstract:
- Machine unlearning (MU) aims to efficiently remove sensitive or harmful memory from a pre-trained model. The key challenge is to balance the potential tradeoff between unlearning efficacy and utility preservation, which involves forgetting undesirable information as defined while maintaining the model’s original performance. One potential way to tackle this problem is to use multi-objective optimization to jointly optimize both the unlearning and utility preservation objectives. However, existing multi-objective methods only guarantee finding a Pareto-optimal solution without fine-grained control, which causes under-optimization of the unlearning objective. To this end, we first model MU as a constrained optimization problem, that is, optimizing the unlearning objective under the constraint of a bounded increase for utility loss. We then show that solving this optimization problem is equivalent to unilateral gradient surgery on the unlearning objective. To resolve the additional computational cost brought by gradient surgery, we propose an implicit gradient surgery method, which approximates the solution to the aforementioned constrained optimization problem via only one backpropagation, thereby achieving efficient utility-preserving MU. Theoretically, we provide a tight convergence analysis of the algorithm. Empirically, our extensive experiments show that the proposed algorithm achieves better tradeoff results than existing baselines.

The file structure:
```
.
├── classification/
│   └── classification_EU_RL/
├── DDPM/
├── SD_Class-wise_unlearn_NSFW_unlearn/
│   └── EUPMU/
└── SD_style+instance_unlearn/
    ├── ESD_with_multi_concept_erasing/
    ├── EUPMU_ConAbl_with_multi_concept_erasing/
    └── EUPMU_style+instance_unlearn/
```

- **classification**: For both class-wise and random data classification unlearning.
- **DDPM**: For class-wise unlearning on Cifar10 image generation DDPM.
- **SD_Class-wise_unlearn_NSFW_unlearn**: For class-wise unlearning on Imagenette dataset image generation Stable Diffusion.
- **ESD_with_multi_concept_erasing**: A slightly modified version of the ESD code base enabling multiple instance forgetting at once.
- **EUPMU_ConAbl_with_multi_concept_erasing**: A modified version of the Concept Ablation code base adding EUPMU to boost the pareto front of the algorithm and also enabling multi concept erasing. Also *Note that this is just for demonstating the effect of combining EUPMU on other unlearning algorithm*
- **EUPMU_style+instance_unlearn**: The final implementation of EUPMU in the paper for Stable Diffusion style and instance unlearning. Modified on the code of SPM. Having the best pareto front overall. 

## Recommended Setup
Except the SD_Class-wise_unlearn_NSFW_unlearn/ which is based on SalUn's SD unlearning code base, all other folders can be easily runned on a 24GB VRAM GPU. The SD_Class-wise_unlearn_NSFW_unlearn/ folder can be runned on 48GB VRAM but 32GB cards like RTX 5090 are not tested.

## Clarification on the Practical Implementation of EUPMU (EU Update)

We would like to clarify a subtle but important detail regarding the implementation of the implicit gradient surgery (EU / EUPMU) update, as it may appear slightly different from the derivation in the paper.

### Theoretical Form (Paper)

In the paper, the update of the dual variable is derived from the constrained optimization formulation:

$$
\tilde{\delta}_t = \frac{1}{\alpha_t} \bigl(\ell_r(\theta_t) - \ell_r(\theta_{t+1})\bigr) + \epsilon_t,
$$

and

$$
\lambda_{t+1} = \lambda_t - \beta_t \tilde{\delta}_t.
$$

Here, $\epsilon_t \ge 0$ represents a tolerance for utility degradation:  
we allow small increases in retaining loss without immediately increasing $\lambda_t$.



### Practical Implementation (Code)

In the released code, we use the following form:

```python
delta = (prev_loss.log() - curr_loss.log()) - self.error
````

and update the weight $w$ (i.e., $\lambda$ ) using Adam.

This can be written as:

$$
\tilde{\delta}_t^{\text{impl}} = \bigl(\log \ell_r(\theta_t) - \log \ell_r(\theta_{t+1})\bigr) - \epsilon,
$$

where `self.error` corresponds to $\epsilon$.



### Why the Form Looks Different (but is Equivalent)

Although the implementation uses a **log-difference with subtraction**, it is still consistent with the paper formulation.

#### (a) Log transformation

For small changes in loss, we have the first-order approximation:

$$
\log \ell_r(\theta_t) - \log \ell_r(\theta_{t+1})
\approx
\frac{\ell_r(\theta_t) - \ell_r(\theta_{t+1})}{\ell_r(\theta_t)}.
$$

Thus, the log-difference is simply a **scaled version of the original improvement term**, preserving its sign and relative magnitude.

#### (b) Sign and subtraction of $\epsilon$

In the paper: $ \tilde{\delta}_t = (\text{improvement}) + \epsilon_t $

In implementation: $ \tilde{\delta}_t^{\text{impl}} = (\text{improvement}) - \epsilon $

This difference comes from **where the threshold is applied**.

* In the paper, $\epsilon_t$ appears inside the dual objective.
* In the code, it is implemented as a **margin (dead zone)** directly on the improvement signal.

Both formulations enforce the same condition:

> **The retain loss must improve by at least $\epsilon$ (up to scaling) to prevent $\lambda$ from increasing.**

Equivalently:

* If improvement is large → $\lambda$ decreases or stays stable
* If improvement is small or negative → $\lambda$ increases

Therefore, `self.error` in code plays the same role as $\epsilon_t$ in the paper:
it controls the **tolerance of utility degradation**.



### Why Use Log Loss + Adam?

The implementation follows a practical design inspired by:

> **FAMO: Fast Adaptive Multitask Optimization (NeurIPS 2023)**
> [https://arxiv.org/abs/2306.03792](https://arxiv.org/abs/2306.03792)

Specifically, we adopt two choices from this line of work:

#### (a) Log-loss difference

* Makes the update depend on **relative improvement** instead of absolute scale
* Stabilizes training across different tasks and loss magnitudes
* Reduces sensitivity to noisy minibatch estimates

#### (b) Adam optimizer for $\lambda$

Instead of directly applying:
$$
\lambda_{t+1} = \lambda_t - \beta_t \tilde{\delta}_t,
$$

we treat $\lambda$ (or $w$) as a learnable scalar and update it with Adam:

$$
\lambda_{t+1} = \text{Adam}(\lambda_t, \tilde{\delta}_t).
$$

This provides:

* Smoother updates under stochastic noise
* Automatic step-size adaptation
* Better empirical stability (especially in diffusion models)



### Key Takeaway

The implementation and the theory are **fully aligned at the conceptual level**:

* Both enforce a **utility-preserving constraint via a tolerance parameter**
* Both adjust $\lambda$ based on **retain loss improvement**
* Both approximate the same dual optimization process

The differences (log transform, subtraction form, Adam update) are **practical enhancements**,
mainly inspired by FAMO-style adaptive weighting, to improve stability and performance in real training.



If you are reproducing results, we recommend treating:

* `self.error` as the practical version of $\epsilon_t$
* the log-loss + Adam update as a **stable surrogate of the theoretical update**

rather than expecting a strict one-to-one numerical match with Eq. (8).


## Known Issues and TODOs
- The EUPMU error and w_lr hyperparameter is not exactly the same thing as in the paper since we found that this way the code is just much simpler and better in practice. Overall the algorithm is the same as in the paper. More documentation will be added soon.
- EUPMU working fine with cosine scheduler but there could be issues when the lr is changing rapidly across a large range.

