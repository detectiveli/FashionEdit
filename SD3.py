import torch
import os
from diffusers import StableDiffusion3Pipeline
import json
from tqdm import tqdm
pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
pipe = pipe.to("cuda")
output_dir = "images"
os.makedirs(output_dir, exist_ok=True)

with open('data/captions.json', 'r') as file:
    data = json.load(file)
    # print(data)

for key, value in tqdm(data.items()):
    print(f"Key: {key}, Value: {value}")
    image = pipe(
        value,
        negative_prompt="",
        num_inference_steps=28,
        guidance_scale=7.0,
    ).images[0]
    # Save the image to a file
    image.save(os.path.join(output_dir, f"{key.split('.')[0]}.png"))