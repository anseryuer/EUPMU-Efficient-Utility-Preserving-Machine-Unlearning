# Code base for Efficient Utility-Preserving Machine Unlearning with Implicit Gradient Surgery (EUPMU)

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

## Known Issues and TODOs
- EUPMU not working well with MUON optimizer.
- EUPMU working fine with cosine scheduler but there could be issues when the lr is changing rapidly across a large range.
