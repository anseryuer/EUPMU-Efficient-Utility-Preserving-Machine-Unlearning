# EUPMU for SD class-wise image forgetting
This is the code base for EUPMU for unlearning class-wise image for stable diffusion. The code structure of this project is adapted from [ESD](https://github.com/rohitgandikota/erasing/tree/main) codebase.

Note that this code base aims to forget one class and preserve the other 9 in the imagenette 10-class image dataset, different from the 

# Installation Guide
* Download the weights from [here](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4-full-ema.ckpt) and move them to `SD/models/ldm/`

# Forgetting Training with EUPMU
1. Forgetting training with EUPMU

    ```
   python train-scripts/random_label_eu.py --train_method full --alpha 0.5 --lr 1e-5 --epochs 5  --class_to_forget 0  --device '0' --mtl --mtl_method "eu"
    ```
   This should create another folder in `SD/model`. 

   You can experiment with forgetting different class labels using the `--class_to_forget` flag, but we will consider forgetting the 0 (tench) class here.

# Generating Images
  1. To use `eval-scripts/generate-images.py` you would need a csv file with columns `prompt`, `evaluation_seed` and `case_number`. (Sample data in `data/`)
  2. To generate multiple images per prompt use the argument `num_samples`. It is default to 10.
  3. The path to model can be customised in the script.
  4. It is to be noted that the current version requires the model to be in saved in `SD/model/compvis-<based on hyperparameters>/diffusers-<based on hyperparameters>.pt`
        ```
        python eval-scripts/generate-images.py --prompts_path 'prompts/imagenette.csv' --save_path 'evaluation_folder/' --model_name {model} --device 'cuda:0'
        ``` 
      For other prompts you can use:
        ```
        python eval-scripts/generate-images.py --prompts_path 'prompts/big_artists_prompts.csv' --save_path evaluation_folder/ --model_name {model} 
        ```
# Evaluation
1. FID
   * First,we need to select some images from Imagenette as real images.
        ```
        .\eval-scripts\save_images.sh
        ```
        This will call the eval-scripts\save_base_dataset.py to randomly select 50 images from each class as real image, create 10 folders for each class' forgetting.
   * Then, we can compute FID between real images and generated images. 
        ```
        python eval-scripts/compute-fid.py --folder_path {images_path}
        ```

2. Unlearning Accuracy
   This will use the classifier to classify the generated image and thus the unlearning accuracy can be calculated.
   ```
   python eval-scripts/imageclassify.py --prompts_path 'prompts/imagenette.csv' --folder_path {images_path}
   ```


# NSFW-concept removal with EUPMU
1. To remove NSFW-concept, we initially utilize SD V1.4 to generate 800 images as Df with the prompt "a photo of a nude person" and store them in "SD/data/nsfw". Additionally, we generate another 800 images designated as Dr using the prompt "a photo of a person wearing clothes" and store them in "SD/data/not-nsfw".

2. Forgetting training with EUPMU

   ```
   python train-scripts/nsfw_removal_EU.py --train_method 'full' --device '0'
   ```
