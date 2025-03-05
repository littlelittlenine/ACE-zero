import time 
import numpy as np
import torch
import random
import pandas as pd
from PIL import Image
import pandas as pd 
import argparse
import requests
import os, glob, json
from transformers import CLIPProcessor, CLIPModel
from diffusers import StableDiffusionPipeline
import abc
import copy
from functools import reduce
import operator
from tqdm import tqdm
import csv
def get_top_100_indices_torch(lst):
    tensor = torch.tensor(lst)
    tensor = tensor.unsqueeze(0)
    _, top_100_indices = torch.topk(tensor, 1)
    return top_100_indices.flatten().tolist()  
def get_bottom_100_indices_torch(lst):
    tensor = torch.tensor(lst)
    tensor = tensor.unsqueeze(0)
    _, bottom_100_indices = torch.topk(tensor, 1, largest=False)
    return bottom_100_indices.flatten().tolist()
def cosine_similarity(a, b):
    a_norm = a / a.norm(dim=-1, keepdim=True)  
    b_norm = b / b.norm(dim=-1, keepdim=True)  
    return (a_norm * b_norm).sum(dim=-1)  
def old_expand(old_emb):
    device = old_emb.device
    print("old_emb:", old_emb.size())
    std_dev = torch.std(old_emb)
    noise_std_dev = std_dev / 10
    noise = torch.randn(49, old_emb.size(1), device=device) * noise_std_dev
    print("noise:", noise.size())
    old_emb_expanded = old_emb.repeat(49, 1)
    new_embeddings = old_emb_expanded + noise

    new_embeddings = torch.vstack((old_emb, new_embeddings))
    print('new_embeddings.size():', new_embeddings.size())
    return new_embeddings

    
def get_embedding(ldm_stable, text):
    texts = [text]  
    text_input = ldm_stable.tokenizer(
        texts,
        padding="max_length",
        max_length=ldm_stable.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    
    with torch.no_grad():  
        text_embeddings = ldm_stable.text_encoder(text_input.input_ids.to(ldm_stable.device))[0]
    
    # 提取索引为 2 的嵌入
    return text_embeddings[:, 2, :]  # 返回形状为 (1, 768) 的嵌入

def get_similar_token(ldm_stable, path, layers_to_edit=None, lamb=0.1,  
                      with_to_k=True, top_k=100):
    token = 'nudity'
    max_bias_diff = 0.05
    
    sub_nets = ldm_stable.unet.named_children()
    ca_layers = []
    
    for net in sub_nets:
        if 'up' in net[0] or 'down' in net[0]:
            for block in net[1]:
                if 'Cross' in block.__class__.__name__:
                    for attn in block.attentions:
                        for transformer in attn.transformer_blocks:
                            ca_layers.append(transformer.attn2)
        if 'mid' in net[0]:
            for attn in net[1].attentions:
                for transformer in attn.transformer_blocks:
                    ca_layers.append(transformer.attn2)

    projection_matrices = [l.to_v for l in ca_layers]
    og_matrices = [copy.deepcopy(l.to_v) for l in ca_layers]
    
    if with_to_k:
        projection_matrices += [l.to_k for l in ca_layers]
        og_matrices += [copy.deepcopy(l.to_k) for l in ca_layers]

    num_ca_clip_layers = len(ca_layers)
    for idx_, l in enumerate(ca_layers):
        l.to_v = copy.deepcopy(og_matrices[idx_])
        projection_matrices[idx_] = l.to_v
        
        if with_to_k:
            l.to_k = copy.deepcopy(og_matrices[num_ca_clip_layers + idx_])
            projection_matrices[num_ca_clip_layers + idx_] = l.to_k

    layers_to_edit = ast.literal_eval(layers_to_edit) if type(layers_to_edit) == str else layers_to_edit
    lamb = ast.literal_eval(lamb) if type(lamb) == str else lamb

    nudity_embedding = get_embedding(ldm_stable, token)

    df = pd.read_csv(path)
    texts_to_check = df['subject'].tolist()  

    similarity_scores = [] 
    seen_texts = set()

    for text in texts_to_check:

        if text == token:
            continue  
        
        text_embedding = get_embedding(ldm_stable, text)
        similarity = cosine_similarity(nudity_embedding, text_embedding)
        
        if text not in seen_texts:  
            similarity_scores.append((similarity.item(), text))
            seen_texts.add(text)   

    similarity_scores.sort(key=lambda x: x[0], reverse=True) 
    top_similar_texts = [text for score, text in similarity_scores[:top_k]]

    if top_similar_texts:
        results_df = pd.DataFrame(top_similar_texts, columns=['subject'])
        results_df.to_csv('/mlx_devbox/users/wangruipeng/playground/paper_run/uce_nullspace/unified-concept-editing-main/data/nudity_similar_texts_100.csv', index=False)
        print(f"Similar text has been stored in a CSV file of similar text: nudity_similar_texts.csv")
    else:
        print("No similar text found")

    return top_similar_texts  
def view_images(images, num_rows=3, offset_ratio=0.02):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0
    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    return pil_img
def get_project_input_3(ldm_stable, data_path, subject_column='subject', num_smallest_singular=300, batch_size=16):
    data = pd.read_csv(data_path)
    data = [artist for artist in data[subject_column] if isinstance(artist, str)]

    total_embeddings = None

    for i in tqdm(range(0, len(data), batch_size)):  
        batch_prompts = data[i:i + batch_size]

        cleaned_prompts = [prompt.replace('“', '').replace('”', '').strip() for prompt in batch_prompts]

        text_input = ldm_stable.tokenizer(
            cleaned_prompts,
            padding="max_length",
            max_length=ldm_stable.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():  
            idx_list = [input_ids.tolist().index(49407) for input_ids in text_input.input_ids]
            text_embeddings = ldm_stable.text_encoder(text_input.input_ids.to(ldm_stable.device))[0]
            print("text_embeddings:", text_embeddings.size())
            text_embeddings = text_embeddings.detach()  
  
            batch_embeddings = []
            for j, idx in enumerate(idx_list):
                batch_embeddings.append(text_embeddings[j, 1:idx+1, :])

            batch_embeddings = torch.cat(batch_embeddings, dim=0)

        text_embeddings = text_embeddings.reshape(-1, batch_embeddings.size(-1))

        if total_embeddings is None:
            total_embeddings = text_embeddings
        else:
            total_embeddings = torch.cat((total_embeddings, text_embeddings), dim=0)

        del text_input, text_embeddings
        torch.cuda.empty_cache()  
    product = total_embeddings.T @ total_embeddings
    U, S, _ = torch.linalg.svd(product, full_matrices=False)
    print("Singular values size:", S.size())

    print(f"Smallest 50 singular values: {S[-50:]}")

    smallest_indices = torch.topk(S, num_smallest_singular, largest=False).indices
    smallest_indices = smallest_indices.sort().values
    print(f"Indices of the smallest {num_smallest_singular} singular values: {smallest_indices}")
    
    projection_matrix = U[:, smallest_indices] @ U[:, smallest_indices].T
    print("Projection matrix size:", projection_matrix.size())
    
    return projection_matrix
def get_project_input(ldm_stable, data_path, subject_column='subject', num_smallest_singular=300, batch_size=16):
    data = pd.read_csv(data_path)

    data = [artist for artist in data[subject_column] if isinstance(artist, str)]

    additional_prompts = [
        'painting by {concept}',
        'art by {concept}',
        'artwork by {concept}',
        'picture by {concept}',
        'style of {concept}',
    ]

    all_prompts = []
    for i, concept in enumerate(data):
        if i < 500:  
            all_prompts.append(concept)  
            for template in additional_prompts:  
                all_prompts.append(template.format(concept=concept))
        else:  
            all_prompts.append(concept)

    print("Total prompts generated:", len(all_prompts))
    print("Sample prompts:", all_prompts[:10])  

    total_embeddings = None

    for i in tqdm(range(0, len(all_prompts), batch_size)):  
        batch_prompts = all_prompts[i:i + batch_size]
        cleaned_prompts = [prompt.replace('“', '').replace('”', '').strip() for prompt in batch_prompts]
        text_input = ldm_stable.tokenizer(
            cleaned_prompts,
            padding="max_length",
            max_length=ldm_stable.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad(): 
            idx_list = [input_ids.tolist().index(49407) for input_ids in text_input.input_ids]
            text_embeddings = ldm_stable.text_encoder(text_input.input_ids.to(ldm_stable.device))[0]
            print("text_embeddings:", text_embeddings.size())
            text_embeddings = text_embeddings.detach()  
            batch_embeddings = []
            for j, idx in enumerate(idx_list):
                batch_embeddings.append(text_embeddings[j, 1:idx+1, :])

            batch_embeddings = torch.cat(batch_embeddings, dim=0)

        print("batch_embeddings:", batch_embeddings.size())
        text_embeddings = text_embeddings.reshape(-1, batch_embeddings.size(-1))
        print("text_embeddings,size:", text_embeddings.size())  # (112, 768)
        print("batch_embeddings,size:", batch_embeddings.size())  # (16, 768)

        if total_embeddings is None:
            total_embeddings = text_embeddings
        else:
            total_embeddings = torch.cat((total_embeddings, text_embeddings), dim=0)

        del text_input, text_embeddings
        torch.cuda.empty_cache()  

    print("Total embeddings size:", total_embeddings.size())  

    product = total_embeddings.T @ total_embeddings

    U, S, _ = torch.linalg.svd(product, full_matrices=False)
    print("Singular values size:", S.size())

    print(f"Smallest 50 singular values: {S[-50:]}")

    smallest_indices = torch.topk(S, num_smallest_singular, largest=False).indices
    smallest_indices = smallest_indices.sort().values
    print(f"Indices of the smallest {num_smallest_singular} singular values: {smallest_indices}")

    projection_matrix = U[:, smallest_indices] @ U[:, smallest_indices].T
    print("Projection matrix size:", projection_matrix.size())

    return projection_matrix
def get_project_input_expand(ldm_stable, data_path, subject_column='subject', num_smallest_singular=300, batch_size=1):

    data = pd.read_csv(data_path)
    data = data[data[subject_column].apply(lambda x: isinstance(x, str))]
    total_embeddings = None
    print(len(data))
    
    for i in tqdm(range(0, len(data[subject_column]), batch_size)):  
        batch_prompts = data[subject_column][i:i + batch_size].tolist()  

        additional_prompts = []
        for concept in batch_prompts:
            additional_prompts.append(f'painting by {concept}')
            additional_prompts.append(f'art by {concept}')
            additional_prompts.append(f'artwork by {concept}')
            additional_prompts.append(f'picture by {concept}')
            additional_prompts.append(f'style of {concept}')
        
        combined_prompts = batch_prompts + additional_prompts
        
        cleaned_prompts = [prompt.replace('“', '').replace('”', '').strip() for prompt in combined_prompts]
        text_input = ldm_stable.tokenizer(
            cleaned_prompts,
            padding="max_length",
            max_length=ldm_stable.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():  
            batch_text_embeddings = []
            print(len(text_input.input_ids))
            for idx in range(len(text_input.input_ids)):  
                input_id = text_input.input_ids[idx].tolist()
                if 49407 in input_id:  
                    print(f"input_id: {input_id}")
                    idx_value = input_id.index(49407)
                    print(f"idx_value: {idx_value}")
                    text_embeddings = ldm_stable.text_encoder(text_input.input_ids[idx].unsqueeze(0).to(ldm_stable.device))[0]
                    text_embeddings = text_embeddings.detach()  
                    text_embeddings = text_embeddings[:, idx_value-1:idx_value:,:]  
                    batch_text_embeddings.append(text_embeddings)
                else:
                    batch_text_embeddings.append(torch.zeros(1, 1, ldm_stable.text_encoder.config.hidden_size).to(ldm_stable.device)) 
            if batch_text_embeddings:
                batch_text_embeddings = torch.cat(batch_text_embeddings, dim=0)
            else:
                batch_text_embeddings = torch.zeros(0, ldm_stable.text_encoder.config.hidden_size).to(ldm_stable.device)  
        text_embeddings = batch_text_embeddings.reshape(-1, batch_text_embeddings.size(-1))

        if total_embeddings is None:
            total_embeddings = text_embeddings
        else:
            total_embeddings = torch.cat((total_embeddings, text_embeddings), dim=0)
        del text_input, batch_text_embeddings
        torch.cuda.empty_cache()  
    product = total_embeddings.T @ total_embeddings
    U, S, _ = torch.linalg.svd(product, full_matrices=False)
    print("Singular values size:", S.size())
    
    print(f"Smallest 50 singular values: {S[-50:]}")
    smallest_indices = torch.topk(S, num_smallest_singular, largest=False).indices
    smallest_indices = smallest_indices.sort().values
    print(f"Indices of the smallest {num_smallest_singular} singular values: {smallest_indices}")

    projection_matrix = U[:, smallest_indices] @ U[:, smallest_indices].T
    print("Projection matrix size:", projection_matrix.size())
    
    return projection_matrix


def diffusion_step(model, latents, context, t, guidance_scale, low_resource=False):
    if low_resource:
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
        noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)

    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)

    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    return latents


def latent2image(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image


def init_latent(latent, model, height, width, generator, batch_size):
    if latent is None:
        latent = torch.randn(
            (batch_size, model.unet.in_channels, height // 8, width // 8),
            generator=generator,
        )
    latents = latent.to(model.device)
    return latent, latents


@torch.no_grad()

def text2image_ldm_stable(
    model,
    prompt,
    num_inference_steps = 50,
    guidance_scale = 7.5,
    generator = None,
    latent = None,
    low_resource = False,
):
    height = width = 512
    batch_size = len(prompt)
    # 处理输入
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    # 文本条件编码
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    # 空的文本嵌入和prompt的batchsize相同
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    # 无关编码
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    # 将文本嵌入和无关编码合并
    context = [uncond_embeddings, text_embeddings]
    # 空间足够
    if not low_resource:
        context = torch.cat(context)
    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    # 调用扩散步骤
    model.scheduler.set_timesteps(num_inference_steps)
    for t in model.scheduler.timesteps:
        latents = diffusion_step(model, latents, context, t, guidance_scale, low_resource)
    image = latent2image(model.vae, latents)

#     image, _ = model.run_safety_checker(image=image, device=model.device, dtype=text_embeddings.dtype)
  
    return image

def generate_for_text(ldm_stable, test_text, num_samples = 9, seed = 1231):
    g = torch.Generator(device='cpu')
    g.manual_seed(seed)
    images = text2image_ldm_stable(ldm_stable, [test_text]*num_samples, latent=None, num_inference_steps=NUM_DIFFUSION_STEPS, guidance_scale=GUIDANCE_SCALE, generator=g, low_resource=LOW_RESOURCE)
    return view_images(images)

clip_model = CLIPModel.from_pretrained("/mnt/bn/intern-disk/mlx/users/wrp/ecu-nullspace/clip-vit")
clip_processor = CLIPProcessor.from_pretrained("/mnt/bn/intern-disk/mlx/users/wrp/ecu-nullspace/clip-vit")

def get_ratios(ldm_stable, prev_ratio, ratio_diff, max_ratio_gap, concepts, classes, num_samples=10, num_loops=3):
    seeds = np.random.randint(5000,size=5) 
    ratios = []

    for idx, concept in enumerate(concepts):
        if ratio_diff is not None:
            if ratio_diff[idx] < max_ratio_gap:
                print(f'Bypassing Concept {idx+1}')
                ratios.append(prev_ratio[idx])
                continue
        prompt = f'{concept}'
        probs_full = []
        test_prompts = [f'{class_}' for class_ in classes[idx]]
        with torch.no_grad():
            for seed in seeds:
                g = torch.Generator(device='cpu')
                g.manual_seed(int(seed))
                images = ldm_stable(prompt,num_images_per_prompt=num_samples, num_inference_steps=20, generator = g).images

                inputs = clip_processor(text=test_prompts, images=images, return_tensors="pt", padding=True)

                outputs = clip_model(**inputs)
                logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
                probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
                tmax = probs.max(1, keepdim=True)[0]
                mask = probs.ge(tmax)
                probs_full.append(mask.float())
                
        ratios.append(torch.cat(probs_full).mean(axis=0))

    return ratios


with_to_k = False
with_augs = True
train_func = "train_closed_form"

LOW_RESOURCE = False
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77
def find_most_diff(ldm_stable, data_path, with_to_k=True):

    df = pd.read_csv(data_path)

    concepts = [artist for artist in df['Artist'] if isinstance(artist, str)]
    print('concepts:',concepts[:20])    
    sub_nets = ldm_stable.unet.named_children()
    ca_layers = []

    for net in sub_nets:
        if 'up' in net[0] or 'down' in net[0]:
            for block in net[1]:
                if 'Cross' in block.__class__.__name__:
                    for attn in block.attentions:
                        for transformer in attn.transformer_blocks:
                            ca_layers.append(transformer.attn2)
        if 'mid' in net[0]:
            for attn in net[1].attentions:
                for transformer in attn.transformer_blocks:
                    ca_layers.append(transformer.attn2)

    projection_matrices = [l.to_v for l in ca_layers]
    print(f"Number of projection matrices, Wk number: {len(projection_matrices)}")
    og_matrices = [copy.deepcopy(l.to_v) for l in ca_layers]
    if with_to_k:
        projection_matrices += [l.to_k for l in ca_layers]
        og_matrices += [copy.deepcopy(l.to_k) for l in ca_layers]

    num_ca_clip_layers = len(ca_layers)
    for idx_, l in enumerate(ca_layers):
        l.to_v = copy.deepcopy(og_matrices[idx_])
        projection_matrices[idx_] = l.to_v
        if with_to_k:
            l.to_k = copy.deepcopy(og_matrices[num_ca_clip_layers + idx_])
            projection_matrices[num_ca_clip_layers + idx_] = l.to_k    

    layer_target = []
    print('concepts:',concepts[:20])
    for layer_num in range(0,len(projection_matrices)):
        scores = []
        total_embeddings = None
        for i in tqdm(range(0, len(concepts), 1)):  
            batch_prompts = concepts[i:i + 1]
            print('batch_prompts:',batch_prompts)

            cleaned_prompts = [prompt.replace('“', '').replace('”', '').strip() for prompt in batch_prompts]

            text_input = ldm_stable.tokenizer(
                cleaned_prompts,
                padding="max_length",
                max_length=ldm_stable.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            nudity_input = ldm_stable.tokenizer(
                ['nudity'],
                padding="max_length",
                max_length=ldm_stable.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )

            with torch.no_grad():  
                idx = text_input.input_ids[0].tolist().index(49407)
                idx_nudity = nudity_input.input_ids[0].tolist().index(49407)
                text_embeddings = ldm_stable.text_encoder(text_input.input_ids.to(ldm_stable.device))[0]
                output_embeddings = projection_matrices[layer_num](text_embeddings).detach()
                nudity_embeddings = ldm_stable.text_encoder(nudity_input.input_ids.to(ldm_stable.device))[0]
                nudity_output_embeddings = projection_matrices[layer_num](nudity_embeddings).detach()
                output_embeddings = output_embeddings[:,idx-1,:]
                nudity_output_embeddings = nudity_output_embeddings[:,idx_nudity-1,:]

                score = cosine_similarity(output_embeddings, nudity_output_embeddings)
                scores.append(score)

        bottom_1_indices_torch = get_bottom_100_indices_torch(scores)
        print('scores:',scores)
        print('bottom_1_indices_torch:',bottom_1_indices_torch)
        bottom_1_word = [concepts[idx] for idx in bottom_1_indices_torch]        

        layer_target.append(bottom_1_word)
        filename = f"/mlx_devbox/users/wangruipeng/playground/paper_run/uce_nullspace/unified-concept-editing-main/data/nudity_bottom_similar/top_100_words_layer_{layer_num}.csv"

        with open(filename, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Word'])  
            for word in bottom_1_word:
                writer.writerow([word])  
    return layer_target
def alpha_edit_5_copy(ldm_stable, old_text_, new_text_, layer_target=None, retain_text_=None, add=False, layers_to_edit=None, lamb=0.1, erase_scale=0.1, preserve_scale=0.1, with_to_k=True, technique='tensor', P=None, P_nude=None, lamda=10, cache_c = None, P_outs=None):

    sub_nets = ldm_stable.unet.named_children()
    ca_layers = []

    for net in sub_nets:
        if 'up' in net[0] or 'down' in net[0]:
            for block in net[1]:
                if 'Cross' in block.__class__.__name__:
                    for attn in block.attentions:
                        for transformer in attn.transformer_blocks:
                            ca_layers.append(transformer.attn2)
        if 'mid' in net[0]:
            for attn in net[1].attentions:
                for transformer in attn.transformer_blocks:
                    ca_layers.append(transformer.attn2)

    projection_matrices = [l.to_v for l in ca_layers]
    og_matrices = [copy.deepcopy(l.to_v) for l in ca_layers]
    if with_to_k:
        projection_matrices = projection_matrices + [l.to_k for l in ca_layers]
        og_matrices = og_matrices + [copy.deepcopy(l.to_k) for l in ca_layers]

    num_ca_clip_layers = len(ca_layers)
    for idx_, l in enumerate(ca_layers):
        l.to_v = copy.deepcopy(og_matrices[idx_])
        projection_matrices[idx_] = l.to_v
        if with_to_k:
            l.to_k = copy.deepcopy(og_matrices[num_ca_clip_layers + idx_])
            projection_matrices[num_ca_clip_layers + idx_] = l.to_k
    layers_to_edit = ast.literal_eval(layers_to_edit) if type(layers_to_edit) == str else layers_to_edit
    lamb = ast.literal_eval(lamb) if type(lamb) == str else lamb
    # print("layers_to_edit", layers_to_edit)      
    old_texts = []
    new_texts = []
    for old_text, new_text in zip(old_text_, new_text_):
        old_texts.append(old_text)
        n_t = new_text if new_text != '' else ' '
        new_texts.append(n_t)
        
    ret_texts = retain_text_ if retain_text_ is not None else ['']
    retain = retain_text_ is not None

    print(old_texts, new_texts)
    # print("layers_to_edit", layers_to_edit) 
    ######################## Edit ###################################
    print(len(projection_matrices))
    for layer_num in range(len(projection_matrices)):
        W_old = projection_matrices[layer_num].weight.detach()      

    P1 = P.clone()
    P2 = P.clone()
    for layer_num in range(len(projection_matrices)):
        # print("layers_to_edit", layers_to_edit) 
        if (layers_to_edit is not None) and (layer_num not in layers_to_edit):
            continue
        # print(f'Editing layer {layer_num}')
        with torch.no_grad():  

            W_old = projection_matrices[layer_num].weight.detach()  
            print('W_old:',W_old.size())
  
            for_mat1 = torch.eye(projection_matrices[layer_num].weight.shape[1],dtype=torch.float,device = projection_matrices[layer_num].weight.device)
            context = None
            value_vector = None
            is_nude = 0
            for cnt, (old_text, new_text) in enumerate(zip(old_texts, new_texts)):
                target_bool = 0
                if 'nudity' in old_text:
                    target_bool = 1
                    print('yesyesyes')
                    is_nude = 1
                    new_text = str(layer_target[layer_num][0]) 
                    # print('new_text：',new_text)
                    print([old_text,new_text])                
                text_input = ldm_stable.tokenizer(
                    [old_text, new_text],
                    padding="max_length",
                    max_length=ldm_stable.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                text_embeddings = ldm_stable.text_encoder(text_input.input_ids.to(ldm_stable.device))[0]

                final_token_idx = text_input.attention_mask[0].sum().item() - 2
                final_token_idx_new = text_input.attention_mask[1].sum().item() - 2
                farthest = max(final_token_idx_new, final_token_idx)
                
                old_emb = text_embeddings[0][final_token_idx:len(text_embeddings[0])-max(0, farthest-final_token_idx)].detach()
                new_emb = text_embeddings[1][final_token_idx_new:len(text_embeddings[1])-max(0, farthest-final_token_idx_new)].detach()  

                n_embs = new_emb @ W_old.T # (4,768) @ (768,320) = (4,320)
                if cnt == 0:
                    context = old_emb.detach() 
                    value_vector = n_embs.detach() 
                else:
                    context = torch.cat((context,old_emb),dim=0)
                    value_vector = torch.cat((value_vector,n_embs.detach()),dim=0)
   
            for_mat2 = (context.T @ context) 
            for_mat3 = value_vector.T @ context - W_old @ for_mat2
        result1 = lamb * (for_mat2 @ P1 + cache_c @ P1) +  lamda * for_mat1
        result2 = lamb * for_mat3 @ P2

        upd_matrix = torch.linalg.solve(
            result1.transpose(0, 1), 
            result2.transpose(0, 1)
        )

        projection_matrices[layer_num].weight = torch.nn.Parameter(W_old + upd_matrix.T)
    cache_c += for_mat2
    print(f'当前模型状态: 将"{str(old_text_)}"编辑为"{str(new_texts)}"，并保留"{str(ret_texts)}"')
    return ldm_stable, cache_c
def alpha_edit_5_copy_2(ldm_stable, old_text_, new_text_, layer_target, retain_text_, add=False, layers_to_edit=None, lamb=0.1, erase_scale=0.1, preserve_scale=0.1, with_to_k=True, technique='tensor', P=None, P_nude=None, lamda=10, cache_c = None, P_outs=None):

    sub_nets = ldm_stable.unet.named_children()
    ca_layers = []

    for net in sub_nets:
        if 'up' in net[0] or 'down' in net[0]:
            for block in net[1]:
                if 'Cross' in block.__class__.__name__:
                    for attn in block.attentions:
                        for transformer in attn.transformer_blocks:
                            ca_layers.append(transformer.attn2)
        if 'mid' in net[0]:
            for attn in net[1].attentions:
                for transformer in attn.transformer_blocks:
                    ca_layers.append(transformer.attn2)

    projection_matrices = [l.to_v for l in ca_layers]
    og_matrices = [copy.deepcopy(l.to_v) for l in ca_layers]
    if with_to_k:
        projection_matrices = projection_matrices + [l.to_k for l in ca_layers]
        og_matrices = og_matrices + [copy.deepcopy(l.to_k) for l in ca_layers]

    num_ca_clip_layers = len(ca_layers)
    for idx_, l in enumerate(ca_layers):
        l.to_v = copy.deepcopy(og_matrices[idx_])
        projection_matrices[idx_] = l.to_v
        if with_to_k:
            l.to_k = copy.deepcopy(og_matrices[num_ca_clip_layers + idx_])
            projection_matrices[num_ca_clip_layers + idx_] = l.to_k

    layers_to_edit = ast.literal_eval(layers_to_edit) if type(layers_to_edit) == str else layers_to_edit
    lamb = ast.literal_eval(lamb) if type(lamb) == str else lamb
    # print("layers_to_edit", layers_to_edit)      
    old_texts = []
    new_texts = []
    for old_text, new_text in zip(old_text_, new_text_):
        old_texts.append(old_text)
        n_t = new_text if new_text != '' else ' '
        new_texts.append(n_t)
        
    ret_texts = retain_text_ if retain_text_ is not None else ['']
    retain = retain_text_ is not None

    print(old_texts, new_texts)
    # print("layers_to_edit", layers_to_edit) 
    ######################## 开始编辑 ###################################
    print(len(projection_matrices))
    for layer_num in range(len(projection_matrices)):
        W_old = projection_matrices[layer_num].weight.detach()  
    P1 = P.clone()
    P2 = P.clone()
    for layer_num in range(len(projection_matrices)):
        # 我们要寻找的是和Wv矩阵对应的Wk矩阵,以及和Wk矩阵对应的Wv矩阵
        # print("layers_to_edit", layers_to_edit) 
        if (layers_to_edit is not None) and (layer_num not in layers_to_edit):
            continue
        # print(f'Editing layer {layer_num}')
        with torch.no_grad():  

            W_old = projection_matrices[layer_num].weight.detach()  
            print('W_old:',W_old.size())

            values = []
            old_embeddings = []
            new_embeddings = []

            for_mat1 = torch.eye(projection_matrices[layer_num].weight.shape[1],dtype=torch.float,device = projection_matrices[layer_num].weight.device)

            for_mat2 = torch.zeros(768,768, device=W_old.device)
            for_mat3 = torch.zeros(projection_matrices[layer_num].weight.shape[0],768, device=W_old.device)

            context = None
            value_vector = None

            # is_nude = 0
            for cnt, (old_text, new_text) in enumerate(zip(old_texts, new_texts)):
                # target_bool = 0
                if 'nudity' in old_text:
                    # target_bool = 1
                    print('yesyesyes')
                    # is_nude = 1
                    new_text = str(layer_target[layer_num][0]) 
                    print([old_text,new_text])                
                text_input = ldm_stable.tokenizer(
                    [old_text, new_text],
                    padding="max_length",
                    max_length=ldm_stable.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                text_embeddings = ldm_stable.text_encoder(text_input.input_ids.to(ldm_stable.device))[0]

                final_token_idx = text_input.attention_mask[0].sum().item() - 2
                final_token_idx_new = text_input.attention_mask[1].sum().item() - 2
                farthest = max(final_token_idx_new, final_token_idx)
                
                old_emb = text_embeddings[0][final_token_idx:len(text_embeddings[0])-max(0, farthest-final_token_idx)].detach()
                new_emb = text_embeddings[1][final_token_idx_new:len(text_embeddings[1])-max(0, farthest-final_token_idx_new)].detach()
                new_embs = projection_matrices[layer_num](new_emb).detach() 

                old_embeddings.append(old_emb) 
                new_embeddings.append(new_embs)               
                context = old_emb.detach() # (4,768)
                context_vector = context.reshape(context.shape[0], context.shape[1], 1) # (75, 768, 1)
                context_vector_T = context.reshape(context.shape[0], 1, context.shape[1]) # (75, 1, 768)

                value_vector = new_embs.reshape(new_embs.shape[0], new_embs.shape[1], 1) # (76, 1280, 1) 
                for_mat2 += (context_vector @ context_vector_T).sum(dim=0)      
                for_mat3 += (value_vector @ context_vector_T).sum(dim=0)        
        result1 = (for_mat2 @ P1 + cache_c @ P1) +  lamda * for_mat1
        result2 = for_mat3 @ P2


        upd_matrix = torch.linalg.solve(
            result1.transpose(0, 1), 
            result2.transpose(0, 1)
        )
        projection_matrices[layer_num].weight = torch.nn.Parameter(W_old + upd_matrix.T)
    cache_c += for_mat2
    print(f'当前模型状态: 将"{str(old_text_)}"编辑为"{str(new_texts)}"，并保留"{str(ret_texts)}"')
    return ldm_stable, cache_c
def get_new_concept():
    target_concept = []
    for layer_num in range(32):
        filename = f"/mlx_devbox/users/wangruipeng/playground/paper_run/uce_nullspace/unified-concept-editing-main/data/nudity_bottom_similar/top_100_words_layer_{layer_num}.csv"
        data = pd.read_csv(filename)
        concept = data.Word.unique()
        print(f"layer_{layer_num}的概念词为：{concept}")
        target_concept.append(concept)
    return target_concept

if __name__ == '__main__':
    seed_value = 1234  
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  
    parser = argparse.ArgumentParser(
                    prog = 'TrainUSD',
                    description = 'Finetuning stable diffusion to debias the concepts')
    parser.add_argument('--concepts', help='prompt corresponding to concept to erase', type=str, required=True)
    parser.add_argument('--guided_concepts', help='Concepts to guide the erased concepts towards', type=str, default=None)
    parser.add_argument('--preserve_concepts', help='Concepts to preserve', type=str, default=None)
    parser.add_argument('--technique', help='technique to erase (either replace or tensor)', type=str, required=False, default='replace')
    parser.add_argument('--device', help='cuda devices to train on', type=str, required=False, default='0')
    parser.add_argument('--base', help='base version for stable diffusion', type=str, required=False, default='1.4')
    parser.add_argument('--preserve_scale', help='scale to preserve concepts', type=float, required=False, default=None)
    parser.add_argument('--preserve_number', help='number of preserve concepts', type=int, required=False, default=None)
    parser.add_argument('--erase_scale', help='scale to erase concepts', type=float, required=False, default=1)
    parser.add_argument('--concept_type', help='type of concept being erased', type=str, required=True)
    parser.add_argument('--add_prompts', help='option to add additional prompts', type=bool, required=False, default=False)
    parser.add_argument('--model_save_path', help='Path to save the model', type=str, required=False, default=None)
    parser.add_argument('--concepts_save_path', help='Path to save the concepts', type=str, required=False,default=None)
    parser.add_argument('--num_smallest_singular', help='Number of smallest singular values to consider', type=int, required=False, default=300)
    parser.add_argument('--coco_path', help='coco dataset path', type=str, required=False, default=None)
    parser.add_argument('--lamb', help='lambda value for optimization', type=float, required=False, default=0.1)  
    parser.add_argument('--lamda', help='Lambda value for scaling the regularization term', type=float, required=False, default=20.0)
    parser.add_argument('--percentage_of_smallest_singular', help='Lambda value for scaling the regularization term', type=float, required=False, default=0.01)
    parser.add_argument('--preserve_artists_path', help='path to preserve artist', type=str, required=False, default=None)

    args = parser.parse_args()
    percentage_of_smallest_singular = args.percentage_of_smallest_singular
    nude_erase_num_smallest_singular = 650
    lamb = args.lamb
    lamda = args.lamda
    coco_path = args.coco_path
    num_smallest_singular = args.num_smallest_singular
    technique = args.technique
    device = f'cuda:{args.device}'
    preserve_scale = args.preserve_scale
    erase_scale = args.erase_scale
    add_prompts = args.add_prompts
    guided_concepts = args.guided_concepts
    preserve_concepts = args.preserve_concepts
    preserve_number = args.preserve_number
    concepts = args.concepts.split(',') 
    concepts = [con.strip() for con in concepts] 
    concept_type = args.concept_type 
    model_save_path = args.model_save_path
    concepts_save_path = args.concepts_save_path
    print_text=''
    for txt in concepts:
        print_text += txt.lower()+'_' 
    print_text  = print_text[:-1]     
    # PROMPT CLEANING
    if concepts[0] == 'allartist':
        concepts = ["Kelly Mckernan", "Thomas Kinkade", "Pablo Picasso", "Tyler Edlin", "Kilian Eng"]
    if concepts[0] == 'i2g':
        concepts = ['hate', 'harassment', 'violence', 'suffering', 'humiliation', 'harm', 'suicide', 'sexual', 'nudity', 'bodily fluids', 'blood']
    if concepts[0] == '10artists':
        concepts = ["Asger Jorn", "Eric Fischl", "Johannes Vermeer", "Apollinary Vasnetsov", "Naoki Urasawa", "Nicolas Mignard", "John Whitcomb", "John Constable", "Warwick Globe", "Albert Marquet"]
    if concepts[0] == 'imagenette':
        concepts = ['Cassette Player', 'Chain Saw', 'Church', 'Gas Pump', 'Tench', 'Garbage Truck', 'English Springer', 'Golf Ball', 'Parachute', 'French Horn']

    if 'artists1000' in concepts[0]:
        df = pd.read_csv('/mlx_devbox/users/wangruipeng/playground/paper_run/uce_nullspace/unified-concept-editing-main/data/erase_artist/erase_nude_artists_1000.csv')    
        artists = list(df.Artist.unique())   
        concepts = artists[:1000]
        print('concepts:',concepts[:20])
        print('concepts:', 'nudity' in concepts)
    if 'something' in concepts[0]:
        df = pd.read_csv('/mlx_devbox/users/wangruipeng/playground/paper_run/uce_nullspace/unified-concept-editing-main/data/concat_data/object_erase_1000.csv')
        concepts = list(df.concept.unique())
        print('concepts:',concepts[:10])
    old_texts = []
    P = None
    print("Loading concept embeddings...")
    additional_prompts = []
    if concept_type == 'art':
        additional_prompts.append('painting by {concept}')
        additional_prompts.append('art by {concept}')
        additional_prompts.append('artwork by {concept}')
        additional_prompts.append('picture by {concept}')
        additional_prompts.append('style of {concept}')
    elif concept_type=='object':
        additional_prompts.append('image of {concept}')
        additional_prompts.append('photo of {concept}')
        additional_prompts.append('portrait of {concept}')
        additional_prompts.append('picture of {concept}')
        additional_prompts.append('painting of {concept}')

    elif concept_type=='bias_profession':  
        additional_prompts.append('Image of {concept}')
        additional_prompts.append('Picture of {concept}')
        additional_prompts.append('Photo of {concept}')
        additional_prompts.append('Headshot of {concept}')
        additional_prompts.append('Portrait of {concept}')

    if not add_prompts:
        additional_prompts = []
    concepts_ = []
    for concept in concepts:
        old_texts.append(f'{concept}')
        for prompt in additional_prompts:
            old_texts.append(prompt.format(concept=concept)) 
        length = 1 + len(additional_prompts)
        concepts_.extend([concept]*length) 
    if guided_concepts is None:
        new_texts = [' ' for _ in old_texts]
        print_text+=f'-towards_uncond'

    elif guided_concepts == 'bias':
        new_texts = ["Make sure the gender ratio is evenly distributed and achieve a balanced ratio of male and female analyst " + item for item in old_texts]
        print_text+=f'-towards_bias'
    else:
        guided_concepts = [con.strip() for con in guided_concepts.split(',')]
        if len(guided_concepts) == 1:
            new_texts = [guided_concepts[0] for _ in old_texts]
            print_text+=f'-towards_{guided_concepts[0]}'
        else:
            new_texts = [[con]*length for con in guided_concepts]
            new_texts = reduce(operator.concat, new_texts) 
            print_text+=f'-towards'
            for t in new_texts:
                if t not in print_text:
                    print_text+=f'-{t}'
     
    assert len(new_texts) == len(old_texts)
    sd14="/mnt/bn/intern-disk/mlx/users/wrp/ecu-nullspace/SDv1.4"
    sd21='/mnt/bn/intern-disk/mlx/users/wrp/ecu-nullspace/SDv2.1'
    data_path = '/mlx_devbox/users/wangruipeng/playground/paper_run/uce_nullspace/unified-concept-editing-main/data/extracted_subjects_big_only_2000.csv'
    if args.base=='1.4':
        model_version = sd14
    elif args.base=='2.1':
        model_version = sd21
    else:
        model_version = sd14
    print("model_version:", model_version)
    ldm_stable = StableDiffusionPipeline.from_pretrained(model_version).to(device)
  
    if preserve_concepts is None:
        if concept_type == 'single_art':
            df = pd.read_csv('data/artists1734_prompts.csv')
            retain_texts = list(df.artist.unique())
            old_texts_lower = [text.lower() for text in old_texts]
            preserve_concepts = [text for text in retain_texts if text.lower() not in old_texts_lower]
            if preserve_number is not None:
                print_text+=f'-preserving_{preserve_number}artists'
                preserve_concepts = random.sample(preserve_concepts, preserve_number)

            artists_df = pd.read_csv(args.preserve_artists_path)
            subjects_df = pd.read_csv(coco_path)

            subjects_df.columns = ['Artist']
            merged_df = pd.concat([artists_df, subjects_df], ignore_index=True)
            with open('/mlx_devbox/users/wangruipeng/playground/paper_run/uce_nullspace/unified-concept-editing-main/train_art/length_of_merged_df.txt', 'w') as file:
                file.write(f"Length of merged DataFrame: {len(merged_df)}\n")
            merged_df.to_csv(args.preserve_artists_path, index=False)

            P = get_project_input_3(ldm_stable, args.preserve_artists_path, subject_column='Artist', num_smallest_singular=num_smallest_singular, batch_size=16)
            
        elif concept_type == 'art':
            layer_target = []
            layer_target = get_new_concept()
            print('done!')
            P = get_project_input_3(ldm_stable, coco_path, subject_column='Artist', num_smallest_singular=num_smallest_singular, batch_size=16)
            preserve_concepts = []
        elif concept_type == 'object':
            # coco_path
            layer_target = None
            P = get_project_input_3(ldm_stable, data_path = '/mlx/mlx_devbox/users/wangruipeng/playground/paper_run/uce_nullspace/unified-concept-editing-main/data/concat_data/preserve_object.csv', subject_column='subject', num_smallest_singular=num_smallest_singular, batch_size=16)
            preserve_concepts = []
        else:
            preserve_concepts = []
    # 保留知识
    retain_texts = ['']+preserve_concepts
    print("len(retain_texts):)", len(retain_texts))  
    if len(retain_texts) > 1:
        print_text+=f'-preserve_true'     
    else:
        print_text+=f'-preserve_false'
    if preserve_scale is None:
        preserve_scale = max(0.1, 1/len(retain_texts))
    print_text += f"-sd_{args.base.replace('.','_')}" 
    print_text += f"-method_{technique}" 
    print_text = print_text.lower()
    print(print_text)

    with open('/mlx/mlx_devbox/users/wangruipeng/playground/paper_run/uce_nullspace/unified-concept-editing-main/train_nude_new/SD21_alpha.txt', 'a') as time_log:
        batch_size = 100
        cache_c = torch.zeros(768, 768, device=device)
        for i in tqdm(range(0, len(old_texts), batch_size)): 
            start_time = time.time()
            old_text = old_texts[i:i + batch_size]
            new_text = new_texts[i:i + batch_size]
            batch_index = i // batch_size 
            model_save_path_with_index = args.model_save_path.replace('.pt', f'_copy_{lamda}_batch_{batch_index}_lamb_{lamb}_num_smallest_{args.num_smallest_singular}_nudity.pt')  # 添加lamb到文件名
            concepts_save_path_with_index = args.concepts_save_path.replace('.txt', f'_copy_{lamda}_batch_{batch_index}_lamb_{lamb}_num_smallest_{args.num_smallest_singular}_nudity.txt')  # 添加lamb到文件名
            
            ldm_stable, cache_c = alpha_edit_5_copy(ldm_stable=ldm_stable, old_text_=old_text, new_text_=new_text, layer_target = layer_target, add=False, retain_text_=retain_texts, lamb=lamb, erase_scale=erase_scale, preserve_scale=preserve_scale, technique=technique, lamda=lamda, P=P, P_nude=None, cache_c=cache_c)
            
            torch.save(ldm_stable.unet.state_dict(), model_save_path_with_index)
            with open(concepts_save_path_with_index, 'w') as fp:
                json.dump(concepts, fp)
            end_time = time.time()
            elapsed_time = end_time - start_time
            time_log.write(f"Batch {batch_index} processed in {elapsed_time:.2f} seconds.\n")
            time_log.flush() 