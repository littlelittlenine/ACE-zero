from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler
import torch
from PIL import Image
import pandas as pd
import argparse
import os
# 如何操作一下上面这个内容，加上prompt之后的效果
def generate_images(prompts_path, save_path, vae, text_encoder, unet, device='cuda:0', guidance_scale=7.5, image_size=512, ddim_steps=100, num_samples=5, from_case=0, till_case=1000000):
    # 1. Load the prompts from CSV
    df = pd.read_csv(prompts_path)
    seeds = df.evaluation_seed
    case_numbers = df.case_number

    folder_path = f'{save_path}/'
    os.makedirs(folder_path, exist_ok=True)
    # embeddings = torch.load("/mnt/bn/intern-disk/mlx/users/wrp/uce_nullspace/model_save/text_embeddings/text_embeddings_0.pt")
    # print("embeddings.shape:", embeddings.shape) #(4, 77, 768)
    # embeddings = embeddings[0].to(device) # (77,768)
    for _, row in df.iterrows():
        # prompt = [str(row.prompt)] * num_samples
        prompt =  [str(row.prompt)] * num_samples
        seed = row.evaluation_seed
        case_number = row.case_number
        if not (case_number >= from_case and case_number <= till_case):
            continue

        height = image_size  # default height of Stable Diffusion
        width = image_size   # default width of Stable Diffusion

        num_inference_steps = ddim_steps  # Number of denoising steps
        generator = torch.manual_seed(seed)  # Seed generator to create the initial latent noise
        batch_size = len(prompt)

        # Tokenize and encode the text
        text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
        text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
        print("text_embeddings.shape:", text_embeddings.shape) 
        # torch.save(text_embeddings, f"{folder_path}/text_embeddings_{case_number}.pt")
        # print("TEXT_INPUT:",text_input)
        # 替换
        # text_embeddings[0][3] = embeddings[5]
        # text_embeddings[0][4] = embeddings[6]
        # text_embeddings[1][3] = embeddings[5]
        # text_embeddings[1][4] = embeddings[6]
        # text_embeddings[2][3] = embeddings[5]
        # text_embeddings[2][4] = embeddings[6]
        # text_embeddings[3][3] = embeddings[5]
        # text_embeddings[3][4] = embeddings[6]
        max_length = text_input.input_ids.shape[-1]
        uncond_input = tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        latents = torch.randn(
            (batch_size, unet.in_channels, height // 8, width // 8),
            generator=generator,
        ).to(device)

        scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
        scheduler.set_timesteps(num_inference_steps)
        latents = latents * scheduler.init_noise_sigma

        from tqdm.auto import tqdm

        for t in tqdm(scheduler.timesteps):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

            with torch.no_grad():
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # Guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = scheduler.step(noise_pred, t, latents).prev_sample

        # Scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            image = vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]

        # Save images with correct file naming
        for num, im in enumerate(pil_images):
            im.save(f"{folder_path}/{case_number}_{num}.png")  # Ensure filenames are case_number_num.png

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='generateImages',
        description='Generate Images using Diffusers Code'
    )
    parser.add_argument('--prompts_path', help='path to csv file with prompts', type=str, required=True)
    parser.add_argument('--save_path', help='folder where to save images', type=str, required=True)
    parser.add_argument('--device', help='cuda device to run on', type=str, required=False, default='cuda:0')
    parser.add_argument('--guidance_scale', help='guidance to run eval', type=float, required=False, default=7.5)
    parser.add_argument('--image_size', help='image size used to train', type=int, required=False, default=512)
    parser.add_argument('--till_case', help='continue generating from case_number', type=int, required=False, default=1000000)
    parser.add_argument('--from_case', help='continue generating from case_number', type=int, required=False, default=0)
    parser.add_argument('--num_samples', help='number of samples per prompt', type=int, required=False, default=1)
    parser.add_argument('--ddim_steps', help='ddim steps of inference used to train', type=int, required=False, default=100)
    args = parser.parse_args()

    # Load models
    model_version = "/mnt/bn/intern-disk/mlx/users/wrp/ecu-nullspace/SDv1.4"
    vae = AutoencoderKL.from_pretrained(model_version, subfolder="vae").to(args.device)
    tokenizer = CLIPTokenizer.from_pretrained(model_version, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_version, subfolder="text_encoder").to(args.device)
    unet = UNet2DConditionModel.from_pretrained(model_version, subfolder="unet").to(args.device)

    generate_images(
        prompts_path=args.prompts_path,
        save_path=args.save_path,
        vae=vae,
        text_encoder=text_encoder,
        unet=unet,
        device=args.device,
        guidance_scale=args.guidance_scale,
        image_size=args.image_size,
        ddim_steps=args.ddim_steps,
        num_samples=args.num_samples,
        from_case=args.from_case,
        till_case=args.till_case
    )
