from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler
import torch
from PIL import Image
import pandas as pd
import argparse
import os
import random
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

def generate_image(row, model_name, device, guidance_scale, image_size, ddim_steps, num_samples, save_path, save_name):
    prompt = [str(row.prompt)] * num_samples
    seed = row.evaluation_seed
    case_number = row.case_number

    height = image_size
    width = image_size
    num_inference_steps = ddim_steps

    generator = torch.manual_seed(seed)
    batch_size = len(prompt)

    text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    text_embeddings = text_encoder(text_input.input_ids.to(device))[0]

    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]

    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    print("1")
    latents = torch.randn((batch_size, unet.in_channels, height // 8, width // 8), generator=generator)
    print("2")
    latents = latents.to(device)
    print("3")
    scheduler.set_timesteps(num_inference_steps)
    latents = latents * scheduler.init_noise_sigma
    print("4")
    for t in scheduler.timesteps:
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        latents = scheduler.step(noise_pred, t, latents).prev_sample

    latents = 1 / 0.18215 * latents
    with torch.no_grad():
        image = vae.decode(latents).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    folder_path = f'{save_path}/{save_name}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
    print(f"Saving images for case number: {case_number}")
    for num, im in enumerate(pil_images):
        im.save(f"{folder_path}/{case_number}_{num}.png")
    print(f"Generated images for case number: {case_number}")

def generate_images(model_name, save_name, prompts_path, save_path, device='cuda:0', guidance_scale=7.5, image_size=512, ddim_steps=100, num_samples=10, from_case=0, till_case=1000000, base='1.4', random_seed=42):
    global tokenizer, text_encoder, unet, scheduler, vae  # Declare as global for use in generate_image function

    # Load the models
    model_version = f"/mnt/bn/intern-disk/mlx/users/wrp/ecu-nullspace/SDv{base}"
    print(f"Loading models from {model_version}")
    vae = AutoencoderKL.from_pretrained(model_version, subfolder="vae").to(device)
    tokenizer = CLIPTokenizer.from_pretrained(model_version, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_version, subfolder="text_encoder").to(device)
    unet = UNet2DConditionModel.from_pretrained(model_version, subfolder="unet").to(device)

    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Read the CSV file
    df = pd.read_csv(prompts_path)
    # df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)  # Shuffle dataset
    print("initialized")
    with ThreadPoolExecutor(max_workers=4) as executor:  # Using 4 threads
        futures = []
        for _, row in df.iterrows():
            case_number = row.case_number
            if not (case_number >= from_case and case_number <= till_case):
                continue
            futures.append(executor.submit(generate_image, row, model_name, device, guidance_scale, image_size, ddim_steps, num_samples, save_path, save_name))

        for future in as_completed(futures):
            future.result()  # Wait for all threads to complete

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='generateImages',
        description='Generate Images using Diffusers Code')
    parser.add_argument('--model_name', help='name of model', type=str, required=False, default="littlenine")
    parser.add_argument('--prompts_path', help='path to csv file with prompts', type=str, required=True)
    parser.add_argument('--save_path', help='folder where to save images', type=str, required=True)
    parser.add_argument('--device', help='cuda device to run on', type=str, required=False, default='cuda:0')
    parser.add_argument('--base', help='version of stable diffusion to use', type=str, required=False, default='1.4')
    parser.add_argument('--guidance_scale', help='guidance to run eval', type=float, required=False, default=7.5)
    parser.add_argument('--image_size', help='image size used to train', type=int, required=False, default=512)
    parser.add_argument('--till_case', help='continue generating from case_number', type=int, required=False, default=1000000)
    parser.add_argument('--from_case', help='continue generating from case_number', type=int, required=False, default=0)
    parser.add_argument('--num_samples', help='number of samples per prompt', type=int, required=False, default=1)
    parser.add_argument('--ddim_steps', help='ddim steps of inference used to train', type=int, required=False, default=100)
    parser.add_argument('--save_name', help='aame of the save text', type=str, required=True)
    args = parser.parse_args()

    model_name = args.model_name
    prompts_path = args.prompts_path
    save_path = args.save_path
    device = args.device
    guidance_scale = args.guidance_scale
    image_size = args.image_size
    ddim_steps = args.ddim_steps
    num_samples = args.num_samples
    from_case = args.from_case
    till_case = args.till_case
    save_name = args.save_name
    base = args.base
    
    generate_images(model_name=model_name, prompts_path=prompts_path, save_path=save_path, device=device,
                    guidance_scale=guidance_scale, image_size=image_size, ddim_steps=ddim_steps, num_samples=num_samples,
                    from_case=from_case, till_case=till_case, base=base, save_name=save_name)
