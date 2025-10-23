from model_pipeline import CustomDiffusionPipeline
import torch

pipe = CustomDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16).to("cuda")
#image = pipe("Snoopy and Mickey going to ski on the snowy mountain with sunshine and blue sky", num_inference_steps=50, guidance_scale=6., eta=1.,generator=torch.manual_seed(42)).images[0]
#image.save("S0.png")

pipe.load_model('logs_ablation/mickey_snoopy/delta.bin')
#image = pipe("A still life featuring bold contrasts between light and shadow, and dramatic use of color, reminiscent of Caravaggio's paintings.", num_inference_steps=50, guidance_scale=6., eta=1.,generator=torch.manual_seed(42)).images[0]
image = pipe("Snoopy and Mickey going to ski on the snowy mountain with sunshine and blue sky", num_inference_steps=50, guidance_scale=6., eta=1.,generator=torch.manual_seed(42)).images[0]

image.save("S1.png")
