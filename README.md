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

## Known Issues and TODOs
- The EUPMU error and w_lr hyperparameter is not exactly the same thing as in the paper since we found that this way the code is just much simpler and better in practice. Overall the algorithm is the same as in the paper. More documentation will be added soon.
- EUPMU working fine with cosine scheduler but there could be issues when the lr is changing rapidly across a large range.

