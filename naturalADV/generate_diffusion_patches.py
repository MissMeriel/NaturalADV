import os

adv_dir = "C:/Users/Meriel/Documents/GitHub/contextualvalidation/high-strength-billboard-library-preproc/"

# from transformers import MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation
from PIL import Image
import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image, make_image_grid
import os
from pathlib import Path
import random
import string


'''
To figure out how to use strength and num_inference_steps:
https://github.com/huggingface/diffusers/blob/main/docs/source/en/using-diffusers/img2img.md
'''

def randstr():
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))

prompts = ["Road barriers, Asphalt road, Curbside markings, Guardrails",
           "foliage, Racing environment, Shadows, Traffic cones",
           "3D simulation, Game graphics, Test track, bleachers",
           "roadside, underbrush, raceway, landscape, bleachers",
           ]
steps = int(500)
negative_prompt = "unrealistic, abstract, muted, poor details, deformed, disfigured, bad anatomy"
strengths = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

def stabilityai_gen(image_path):
    global prompts, negative_prompt, steps, strengths

    init_image = load_image(str(image_path))

    init_image = init_image
    width, height = init_image.size
    if width < 20 or height < 20:
        newsize = (width * 5, height * 5)
        init_image = init_image.resize(newsize)

    pipeline = AutoPipelineForImage2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    )
    pipeline.enable_model_cpu_offload()
    # remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
    # pipeline.enable_xformers_memory_efficient_attention()

    outdir = f"./output-generated-patches/{image_path.stem}/{pipeline.name_or_path.replace('/', '')}-using_negative_prompt/"
    os.makedirs(outdir, exist_ok=True)
    steps = 100
    negative_prompt = "ugly, deformed, disfigured, muted, abstract, poor details, bad anatomy"
    strengths = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    strengths.reverse()
    for prompt in prompts:
        for strength in strengths:
            print(image_path, prompt)
            # The value of strength should in [0.0, 1.0]. 1.0 means the starting image is basically ignored.
            images = pipeline(prompt, negative_prompt=negative_prompt, image=init_image, num_inference_steps=steps, strength=strength).images
            if len(images) > 0:
                image = images[0]
                grid = make_image_grid([image], rows=1, cols=1)
                tags = prompt.replace(", ", "-").replace(" ", "")
                outfile = f"{outdir}/test-prompt-{tags}-steps{steps}-strength{strength}-{randstr()}.jpg"
                grid.save(outfile)
                print(outfile)
            else:
                print("No image generated")

def stablediffusionv15_gen(image_path):
    global prompts, negative_prompt, steps, strengths

    init_image = load_image(str(image_path))

    init_image = init_image
    width, height = init_image.size
    if width < 20 or height < 20:
        newsize = (width * 5, height * 5)
        init_image = init_image.resize(newsize)

    pipeline = AutoPipelineForImage2Image.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    )
    pipeline.enable_model_cpu_offload()
    # remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
    # pipeline.enable_xformers_memory_efficient_attention()
    # callback_steps has to be a positive integer
    # The value of strength should in [0.0, 1.0]. 1.0 means the starting image is basically ignored.
    outdir = f"./output-generated-patches/{image_path.stem}/{pipeline.name_or_path.replace('/', '')}-using_negative_prompt/"
    os.makedirs(outdir, exist_ok=True)
    guidance_scale = 0.8
    for prompt in prompts:
        for strength in strengths:
            print(image_path, prompt)
            images = pipeline(prompt, negative_prompt=negative_prompt, image=init_image, num_inference_steps=steps, strength=strength, guidance_scale=guidance_scale).images
            if len(images) > 0:
                image = images[0]
                grid = make_image_grid([image], rows=1, cols=1)
                tags = prompt.replace(", ", "-").replace(" ", "")
                outfile = f"{outdir}/test-prompt-{tags}-steps{steps}-strength{strength}-guidance_scale{guidance_scale}-{randstr()}.jpg"
                grid.save(outfile)
                print(outfile)
            else:
                print("No image generated")

def runwayml_gen(image_path):
    global prompts, negative_prompt, steps, strengths
    init_image = load_image(str(image_path))

    init_image = init_image
    width, height = init_image.size
    if width < 20 or height < 20:
        newsize = (width * 5, height * 5)
        init_image = init_image.resize(newsize)

    pipeline = AutoPipelineForImage2Image.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16,# variant="fp16", use_safetensors=True
    )
    pipeline.enable_model_cpu_offload()
    # remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
    # pipeline.enable_xformers_memory_efficient_attention()

    outdir = f"./output-generated-patches/{image_path.stem}/{pipeline.name_or_path.replace('/', '')}-using_negative_prompt/"
    os.makedirs(outdir, exist_ok=True)

    guidance_scale = 0.8
    for prompt in prompts:
        for strength in strengths:
            print(image_path, prompt)
            # The value of strength should in [0.0, 1.0]. 1.0 means the starting image is basically ignored.
            images = pipeline(prompt, negative_prompt=negative_prompt, image=init_image, num_inference_steps=steps, strength=strength, guidance_scale=guidance_scale).images
            if len(images) > 0:
                image = images[0]
                grid = make_image_grid([image], rows=1, cols=1)
                tags = prompt.replace(", ", "-").replace(" ", "")
                outfile = f"{outdir}/test-prompt-{tags}-steps{steps}-strength{strength}-guidance_scale{guidance_scale}-{randstr()}.jpg"
                grid.save(outfile)
                print(outfile)
            else:
                print("No image generated")

def kandinsky_gen(image_path):
    global prompts, negative_prompt, steps, strengths
    init_image = load_image(str(image_path))

    init_image = init_image
    width, height = init_image.size
    if width < 20 or height < 20:
        newsize = (width * 5, height * 5)
        init_image = init_image.resize(newsize)

    pipeline = AutoPipelineForImage2Image.from_pretrained(
        "kandinsky-community/kandinsky-2-2-decoder", torch_dtype=torch.float16, #use_safetensors=True
    )
    pipeline.enable_model_cpu_offload()
    # remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
    # pipeline.enable_xformers_memory_efficient_attention()
    # callback_steps has to be a positive integer
    # The value of strength should in [0.0, 1.0]. 1.0 means the starting image is basically ignored.
    outdir = f"./output-generated-patches/{image_path.stem}/{pipeline.name_or_path.replace('/', '')}-using_negative_prompt/"
    os.makedirs(outdir, exist_ok=True)

    guidance_scale = 0.8
    for prompt in prompts:
        for strength in strengths:
            print(image_path, prompt)
            # The value of strength should in [0.0, 1.0]. 1.0 means the starting image is basically ignored.
            images = pipeline(prompt, negative_prompt=negative_prompt, image=init_image, num_inference_steps=steps, strength=strength, guidance_scale=guidance_scale).images
            if len(images) > 0:
                image = images[0]
                grid = make_image_grid([image], rows=1, cols=1)
                tags = prompt.replace(", ", "-").replace(" ", "")
                outfile = f"{outdir}/test-prompt-{tags}-steps{steps}-strength{strength}-guidance_scale{guidance_scale}-{randstr()}.jpg"
                grid.save(outfile)
                print(outfile)
            else:
                print("No image generated")


if __name__ == "__main__":
    perturbationparentdir = "../high-strength-billboard-library-preproc/"
    # "C:/Users/Meriel/Documents/GitHub/contextualvalidation/high-strength-billboard-library-preproc/"
    imgs = os.listdir(perturbationparentdir)
    for img in imgs:
        # stabilityai_gen(Path(perturbationparentdir + img))
        # stablediffusionv15_gen(Path(perturbationparentdir + img))
        kandinsky_gen(Path(perturbationparentdir + img))
        runwayml_gen(Path(perturbationparentdir + img))