# EUPMU for style and multi-instance unlearning

This is the code base for EUPMU for style and multi-instance unlearning for diffusion models (DMs). 

Our code is based the work of Semi-Permeable Membrane (SPM)[SPM](https://github.com/Con6924/SPM)

EUPMU is a multi-objective optimization (MOO) method for machine unlearning and can be used to achieve further pareto front on

- Prevent the generation of **unlearn concept** from the DMs, while
- Preserve the generation of **retain concept** of the DMs.

As the EUPMU boosts the SPM models to better performance and tradeoff, the EUPMU models still holds the advantages of SPM models, emphasize the flexibility of EUPMU to combine itself with other models to improve them, as long as they have the 2 targets in unlearning:

- Data-free: no extra text or image data is needed for training.
- Customizable: once obtained, unlearning EUPMU-SPM of different target concept can be equipped simultaneously on the DM according to your needs.

## Getting Started

### 0. Setup

We use Conda to setup the training environments:

```bash
conda create -n eupmu-style python=3.10
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install xformers
pip install -r requirements.txt
```

### 1. Training EUPMU models.

In the demo.ipynb notebook we provide tutorials for setting up configs and training. Please refer to the notebook for further details.

### 2. Generate with finished models

You can generate images by running the following commands:

```shell
python infer_spm.py \
		--config ${generation_config}	\
		--spm_path ${spm_1} ... ${spm_n} \
		--base_model ${base_model_path_or_link} \ 											# e.g. CompVis/stable-diffusion-v1-4
```

python infer_spm.py \
		--config ${generation_config}	\
		--spm_path ${spm_1} ... ${spm_n} \
		--base_model ${base_model_path_or_link} 

Here just select the EUPMU-SPMs for the concepts you need to unlearn.

You can change the facility factor of the unlearn to adjust the strength of the unlearning.

### 3. Evaluate 

Run the following code to evaluate the trained models on the four pre-defined tasks. 

```shell
accelerate launch --num_processes ${num_gpus} evaluate_task.py \
		--task ${task} \
		--task_args ${task_args} \
		--img_save_path ${img_save_path} \
		--save_path ${save_path}
```

An Example here with our recommended van gogh EUPMU+SPM facilitate factor = 1.6 inference parameter introduced in SPM. Increase ff increase unlearning strength and decrease overall quality.

```
accelerate launch --num_processes 1 evaluate_task.py   --task artwork   --task_args "datasets=[vangogh,picasso,rembrandt]"   --img_save_path ./generated_images/artwork_vangogh_unlearn_ff16   --save_path ./output/van_gogh_ff16/clip   --spm_paths ./output/van_gogh_eu   --precision bf16 --facilitate_factor 1.6
```

For object erasure task for snoopy, mickey, spongebob we recommend ff = 1.2 for the metrics in the paper.

```
accelerate launch evaluate_task.py \
    --task general \
    --task_args "concepts=[snoopy, mickey, spongebob]" \
    --img_save_path ./generated_images/snoopy_ff12 \
    --save_path ./output/snoopy_erasure_ff12 \
    --spm_paths output/snoopy \
    --precision bf16 --facilitate_factor 1.2

accelerate launch evaluate_task.py \
    --task general \
    --task_args "concepts=[snoopy, mickey, spongebob]" \
    --img_save_path ./generated_images/snoopy_mickey_ff12 \
    --save_path ./output/snoopy_mickey_erasure_ff12 \
    --spm_paths output/snoopy output/mickey \
    --precision bf16 --facilitate_factor 1.2

accelerate launch evaluate_task.py \
    --task general \
    --task_args "concepts=[snoopy, mickey, spongebob]" \
    --img_save_path ./generated_images/snoopy_mickey_spongebob_ff12 \
    --save_path ./output/snoopy_mickey_spongebob_erasure_ff12 \
    --spm_paths output/snoopy output/mickey output/spongebob  \
    --precision bf16 --facilitate_factor 1.2
```

For evaluating the FID metrics, you need to generate the images with original SD and compare the FID with unlearned model's generation.

```
accelerate launch --num_processes 1 evaluate_task.py   --task artwork   --task_args "datasets=[vangogh,picasso,rembrandt]"   --img_save_path ./generated_images/o   --save_path ./output/o/clip  --precision fp16accelerate launch --num_processes 1 evaluate_task.py   --task artwork   --task_args "datasets=[vangogh,picasso,rembrandt]"   --img_save_path ./generated_images/o   --save_path ./output/o/clip  --precision fp16
```

Then

```
python calculate_metrics.py --original generated_images/o --generated generated_images/artwork_picasso_unlearn_ff16
```

Things to notice:

Although the original spm code is kept, but in the demo.ipynb many hyperparameters are tuned for EUPMU model, so for the original spm please read their original repo and use the spm hyperparameter.

One **important** inference parameter in controlling unlearning strength introduced by SPM and kept in EUPMU+SPM is the facilitate factor, which is an arg used during inferencing and evaluating. Passing `--facilitate_factor 2.0` like arg into the command to change the SPM unlearning factor. This is crucial for reaching the desired balance between forget and retain. Sometimes you might overshoot on one of unlearn or retain so we suggest to run multiple ff value to see the whole picture of pareto front. We have done this in the llustration of Pareto Front Analysis figure in the paper and with integrating EUPMU the overall pareto front is improved for both SPM and ConAbl.

Also, notice that ESD change their code base to fit newer diffusers finetuning method and the following evaluate_task_ESD_new.py is needed to evaluate the model trained by the newer version of the ESD code base.

```
accelerate launch evaluate_task_ESD_new.py --task artwork --task_
args "datasets=[vangogh, picasso, rembrandt]" --img_save_path generated_images/esd-o-pic --save_path ./output/vg-400/clip  --ft_model_path
```

Unlearning models by other algorithms (ESD, ConAbl) are supported as well, just replace the --spm_paths to "--ft_model_path models/conabl/vangogh/delta.bin" and you can evaluate the other model's benchmark with the same procedure.
