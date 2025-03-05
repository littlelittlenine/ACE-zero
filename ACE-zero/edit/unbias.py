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
from CLIP_classify import CLIP_classification
from generate_images_copy import generate_images_1

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
    
    return text_embeddings[:, 2, :]  #

def cosine_similarity(a, b):
    """Calculate the cosine similarity between two vectors"""
    a_norm = a / a.norm(dim=-1, keepdim=True)  # Normalize vector a
    b_norm = b / b.norm(dim=-1, keepdim=True)  # Normalize vector b
    return (a_norm * b_norm).sum(dim=-1)  # Return cosine similarity

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

    # Get the embedding for "nudity"
    nudity_embedding = get_embedding(ldm_stable, token)

    # Load text data from the specified CSV file
    df = pd.read_csv(path)
    texts_to_check = df['subject'].tolist()  # Extract text from the 'subject' column

    similarity_scores = []  # Store similarity scores and text tuples
    seen_texts = set()  # Track added texts

    # Iterate through text data to calculate similarity with "nudity"
    for text in texts_to_check:
        # Ensure the original nudity token is not included
        if text == token:
            continue  # Skip the same token
        
        text_embedding = get_embedding(ldm_stable, text)
        similarity = cosine_similarity(nudity_embedding, text_embedding)
        
        # Store similarity score and text as a tuple
        if text not in seen_texts:  # Check if already added
            similarity_scores.append((similarity.item(), text))
            seen_texts.add(text)  # Add text to the seen set

    # Sort by similarity and extract the top_k texts
    similarity_scores.sort(key=lambda x: x[0], reverse=True)  # Sort in descending order
    top_similar_texts = [text for score, text in similarity_scores[:top_k]]

    # Store similar texts in a CSV file
    if top_similar_texts:
        results_df = pd.DataFrame(top_similar_texts, columns=['subject'])
        results_df.to_csv('/mlx_devbox/users/wangruipeng/playground/paper_run/uce_nullspace/unified-concept-editing-main/data/nudity_similar_texts_100.csv', index=False)
        print(f"Similar texts have been stored in the CSV file: nudity_similar_texts.csv")
    else:
        print("No similar texts found.")

    return top_similar_texts  # Return the list of similar texts
def project_to_null_space(a, P, b):
    # Project tensor a to the null space
    a_proj =  a @ P
    # Project vector b to the orthogonal complement of the null space
    b_proj = b - (b @ P)  # Original projection of b
    # Calculate the unit vector of b_proj
    b_proj_norm = torch.norm(b_proj)
    if b_proj_norm > 0:
        b_proj_unit = b_proj / b_proj_norm  # Unit vector
    else:
        b_proj_unit = b_proj  # If b_proj is zero, avoid division by zero
    # Calculate the adjustment
    adjustment = torch.dot(a_proj.flatten(), b_proj_unit.flatten()) * b_proj_unit
    # Subtract this adjustment from a_proj
    a_final = a_proj - adjustment
    
    return a_final
def view_images(images, num_rows=3, offset_ratio=0.02):
    if type(images) is list:
        # Calculate how many images need to be replaced with blank images
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0
    # Create blank images with the same shape as the input images
    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    # Add blank images to the input image list
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
def diffusion_step(model, latents, context, t, guidance_scale, low_resource=False):
    # Calculate noise prediction: predict noise under unconditional and text conditions, two identical things, but one is through time for space
    if low_resource:
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
        noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    # Calculate the final noise prediction: adjust noise prediction based on guidance scale
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    # Update latent state
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    return latents

# Convert latent space representation to image
def latent2image(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image

# Initialize latent space representation
def init_latent(latent, model, height, width, generator, batch_size):
    if latent is None:
        latent = torch.randn(
            (batch_size, model.unet.in_channels, height // 8, width // 8),
            generator=generator,
        )
    latents = latent.to(model.device)
    return latent, latents


@torch.no_grad()
# Call the above functions to generate images
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
    # Process input
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    # Text condition encoding
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    # Empty text embeddings with the same batch size as prompt
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    # Unrelated encoding
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    # Merge text embeddings and unrelated encoding
    context = [uncond_embeddings, text_embeddings]
    # Sufficient space
    if not low_resource:
        context = torch.cat(context)
    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    # Call diffusion step
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
    # Specific concepts
    for idx, concept in enumerate(concepts):
        if ratio_diff is not None:
            # If the ratio difference is less than the maximum ratio gap, skip the current concept
            if ratio_diff[idx] < max_ratio_gap:
                print(f'Bypassing Concept {idx+1}')
                ratios.append(prev_ratio[idx])
                continue
        prompt = f'{concept}'
        probs_full = []
        test_prompts = [f'{class_}' for class_ in classes[idx]]
        with torch.no_grad():
            for seed in seeds:
    #             if i == num_loops:
    #                 break
                g = torch.Generator(device='cpu')
                g.manual_seed(int(seed))
                images = ldm_stable(prompt,num_images_per_prompt=num_samples, num_inference_steps=20, generator = g).images

                inputs = clip_processor(text=test_prompts, images=images, return_tensors="pt", padding=True)
                # Calculate image and text similarity
                outputs = clip_model(**inputs)
                logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
                probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
                tmax = probs.max(1, keepdim=True)[0]
                mask = probs.ge(tmax)
                probs_full.append(mask.float())
                
        ratios.append(torch.cat(probs_full).mean(axis=0))
#     male = float(probs[0][0])
    return ratios

## Get arguments for our script
# Get script parameters
with_to_k = False
with_augs = True
train_func = "train_closed_form"

### Load model
LOW_RESOURCE = False
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77

def alpha_edit_2(ldm_stable, old_text_, new_text_, retain_text_, add=False, layers_to_edit=None, lamb=0.1, erase_scale=0.1, preserve_scale=0.1, with_to_k=True, technique='tensor', P=None, lamda=10, cache_c = None, P_outs = None):

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
    print("layers_to_edit", layers_to_edit)      
    old_texts = []
    new_texts = []
    for old_text, new_text in zip(old_text_, new_text_):
        old_texts.append(old_text)
        n_t = new_text if new_text != '' else ' '
        new_texts.append(n_t)
        
    ret_texts = retain_text_ if retain_text_ is not None else ['']
    retain = retain_text_ is not None

    print(old_texts, new_texts)
    print("layers_to_edit", layers_to_edit) 
    ######################## EDIT ###################################
    print(len(projection_matrices))
    P1 = P.clone()
    P2 = P.clone()
    for layer_num in range(len(projection_matrices)):
        print("layers_to_edit", layers_to_edit) 
        if (layers_to_edit is not None) and (layer_num not in layers_to_edit):
            continue
        print(f'Editing layer {layer_num}')
        with torch.no_grad(): 
            W_old = projection_matrices[layer_num].weight.detach()  
            for_mat1 = torch.eye(projection_matrices[layer_num].weight.shape[1],dtype=torch.float,device = projection_matrices[layer_num].weight.device)
            for_mat2 = torch.zeros(768,768, device=W_old.device)
            for_mat3 = torch.zeros(projection_matrices[layer_num].weight.shape[0],768, device=W_old.device)
            for cnt, (old_text, new_text) in enumerate(zip(old_texts, new_texts)):
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
                new_emb = projection_matrices[layer_num](new_emb).detach()

                context = old_emb.detach()
                context_vector = context.reshape(context.shape[0], context.shape[1], 1) # (1, 768, 1)
                context_vector_T = context.reshape(context.shape[0], 1, context.shape[1]) # (1, 1, 768)

                value_vector = new_emb.reshape(new_emb.shape[0], new_emb.shape[1], 1) # (2, 320, 1)
         
                for_mat2 += (context_vector @ context_vector_T).sum(dim=0) 
                for_mat3 += (value_vector @ context.T - W_old @ for_mat2).sum(dim=0) 

            print("P1.device:",P1.device)
            print('for_mat2.device:',for_mat1.device)
            print('for_mat3.device:',for_mat2.device)
            result1 = (for_mat2 @ P1 + cache_c @ P1) + lamda * for_mat1
            result2 = for_mat3 @ P2
            # (320,768) @ (768,768) = (320,768)
            upd_matrix = torch.linalg.solve(
                result1.transpose(0, 1), 
                result2.transpose(0, 1)
            )

            projection_matrices[layer_num].weight = torch.nn.Parameter(W_old + upd_matrix.T)
    cache_c += for_mat2
    print(f'Current model state: Edited "{str(old_text_)}" to "{str(new_texts)}" and retained "{str(ret_texts)}"')
    return ldm_stable, cache_c

def alpha_edit_3(ldm_stable, old_text_, new_text_, retain_text_, add=False, layers_to_edit=None, lamb=0.1, erase_scale=0.1, preserve_scale=0.1, with_to_k=True, technique='tensor', P=None, lamda=10, cache_c = None, P_outs = None):
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
    print("layers_to_edit", layers_to_edit)      

    old_texts = []
    new_texts = []
    for old_text, new_text in zip(old_text_, new_text_):
        old_texts.append(old_text)
        n_t = new_text if new_text != '' else ' '
        new_texts.append(n_t)
        
    ret_texts = retain_text_ if retain_text_ is not None else ['']
    retain = retain_text_ is not None

    print(old_texts, new_texts)
    print("layers_to_edit", layers_to_edit) 
    ######################## EDIT ###################################
    print(len(projection_matrices))
    for layer_num in range(len(projection_matrices)):
        print("layers_to_edit", layers_to_edit) 
        if (layers_to_edit is not None) and (layer_num not in layers_to_edit):
            continue
        print(f'Editing layer {layer_num}')
        with torch.no_grad():  
            W_old = projection_matrices[layer_num].weight.detach()  
            old_embeddings = []
            new_embeddings = []
            for old_text, new_text in zip(old_texts, new_texts):
                print([old_text, new_text])
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

                print("old_emb",old_emb.size())
                old_embeddings.append(old_emb) # (768)
                new_embs = projection_matrices[layer_num](new_emb).detach() # (320)

                new_embeddings.append(new_embs)

            old_embs = torch.cat(old_embeddings, dim=0)  # (1, 768)
            new_embs = torch.cat(new_embeddings, dim=0) 
            print("old_embs",old_embs.size())

            Kp1 = P  
            Kp2 = P
            # n_embs = new_emb @ W_old.T # (4,768) @ (768,320) = (4,320)
            context = old_embs.detach()
            for_mat1 = torch.eye(projection_matrices[layer_num].weight.shape[1],dtype=torch.float,device = projection_matrices[layer_num].weight.device)  
            for_mat2 = (context.T @ context) # (768,768)
            for_mat3 = new_embs.T @ context - W_old @ for_mat2

        result1 = (for_mat2 @ Kp1 + cache_c @ Kp2) + lamda * for_mat1
        result2 = for_mat3 @ Kp2
        upd_matrix = torch.linalg.solve(
            result1.transpose(0, 1), 
            result2.transpose(0, 1)
        )
        projection_matrices[layer_num].weight = torch.nn.Parameter(W_old + upd_matrix.T)
    print(f'Current model state: Edited "{str(old_text_)}" to "{str(new_texts)}" and retained "{str(ret_texts)}"')
    cache_c += for_mat2
    return ldm_stable, cache_c

def get_project_input(ldm_stable, data_path, subject_column='subject', num_smallest_singular=300, batch_size=16):
    data = pd.read_csv(data_path)
    data = [artist for artist in data[subject_column] if isinstance(artist, str)]
    total_embeddings = None
    print(len(data))
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
                batch_embeddings.append(text_embeddings[j, idx-1, :].unsqueeze(0))
            batch_embeddings = torch.cat(batch_embeddings, dim=0)
        if total_embeddings is None:
            total_embeddings = batch_embeddings
        else:
            total_embeddings = torch.cat((total_embeddings, batch_embeddings), dim=0)
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
    # parser.add_argument('--projection_path', help='Path to the projection matrix', type=str, required=True)
    parser.add_argument('--num_smallest_singular', help='Number of smallest singular values to consider', type=int, required=False, default=300)
    parser.add_argument('--coco_path', help='coco dataset path', type=str, required=False, default=None)
    parser.add_argument('--lamb', help='lambda value for optimization', type=float, required=False, default=0.1)  # Newly added lamb parameter
    parser.add_argument('--lamda', help='Lambda value for scaling the regularization term', type=float, required=False, default=20.0)

    args = parser.parse_args()
    coco_path = args.coco_path
    num_smallest_singular = args.num_smallest_singular
    lamb = args.lamb
    lamda = args.lamda
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
    if 'artists' in concepts[0]:
        df = pd.read_csv('data/artists1734_prompts.csv')
        artists = list(df.artist.unique())
        number = int(concepts[0].replace('artists', ''))
        concepts = random.sample(artists,number)
    if 'professions5' in concepts[0]: 
        concepts = ['librarian', 'teacher', 'sheriff', 'analyst', 'doctor']
    elif 'professions' in concepts[0] and 'professions5' not in concepts[0]:
        df = pd.read_csv('/mlx_devbox/users/wangruipeng/playground/paper_run/uce_nullspace/unified-concept-editing-main/data/profession_prompts.csv')
        professions = list(df.profession.unique())
        number = int(concepts[0].replace('professions', ''))
        concepts = random.sample(professions,number)     
    old_texts = []
    
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
            old_texts.append(prompt.format(concept=concept)) #
        length = 1 + len(additional_prompts)
        concepts_.extend([concept]*length) 
    if guided_concepts is None:
        new_texts = [' ' for _ in old_texts]
        print_text+=f'-towards_uncond'
    elif guided_concepts == 'origin':
        new_texts = [item for item in old_texts]
        print_text+=f'-towards_origin'
    else:
        guided_concepts = [con.strip() for con in guided_concepts.split(',')]
        if len(guided_concepts) == 1:
            new_texts = [guided_concepts[0] for _ in old_texts]
            print_text+=f'-towards_{guided_concepts[0]}'
        else:
            new_texts = [[con]*length for con in guided_concepts]
            new_texts = reduce(operator.concat, new_texts) # Convert 2D list to 1D list
            print_text+=f'-towards'
            for t in new_texts:
                if t not in print_text:
                    print_text+=f'-{t}'
    #   Equal      
    assert len(new_texts) == len(old_texts)
    
    sd14="/mnt/bn/intern-disk/mlx/users/wrp/ecu-nullspace/SDv1.4"
    sd21='/mnt/bn/intern-disk/mlx/users/wrp/ecu-nullspace/SDv2.1'
    if args.base=='1.4':
        model_version = sd14
    elif args.base=='2.1':
        model_version = sd21
    else:
        model_version = sd14
    ldm_stable = StableDiffusionPipeline.from_pretrained(model_version).to(device)    
    
    if preserve_concepts is None:
        if concept_type == 'art':
            df = pd.read_csv('data/artists1734_prompts.csv')

            retain_texts = list(df.artist.unique())
            old_texts_lower = [text.lower() for text in old_texts]
            preserve_concepts = [text for text in retain_texts if text.lower() not in old_texts_lower]
            if preserve_number is not None:
                print_text+=f'-preserving_{preserve_number}artists'
                preserve_concepts = random.sample(preserve_concepts, preserve_number)
        elif concept_type == 'bias_profession':
            # P = get_project_input_3(ldm_stable, coco_path, num_smallest_singular=num_smallest_singular, batch_size=16)
            preserve_concepts = []
        else:
            preserve_concepts = []
    # First, do not retain knowledge, UCE
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

    batch_size = 1
    print("len(old_texts):", len(old_texts))
    cache_c = torch.zeros(768, 768, device=device)
    cache_c_last = torch.zeros(768, 768, device=device)

    last_saved_model_state = ldm_stable.unet.state_dict()
    get_saved_model_state = ldm_stable.unet.state_dict()
    log_file_path = '/mlx_devbox/users/wangruipeng/playground/paper_run/uce_nullspace/unified-concept-editing-main/train_unbias/log_file_10.txt'  # Replace with your desired log file path
    # Open file in append mode
    with open(log_file_path, 'a') as log_file:
        for i in tqdm(range(0, len(old_texts), batch_size)):
            old_text = old_texts[i:i + batch_size]
            new_text = new_texts[i:i + batch_size]
            # Save current model state
            current_model_state = ldm_stable.unet.state_dict()
            ori_path = '/mlx_devbox/users/wangruipeng/playground/paper_run/uce_nullspace/unified-concept-editing-main/data/bias/bias/bias2/'
            counts_save = []
            for num_smallest_singular in range(10, 450, 10):
                P = get_project_input(ldm_stable, coco_path, num_smallest_singular=num_smallest_singular, batch_size=16)
                batch_index = i // batch_size
                print("old_text:", old_text[0])
                generate_images_1(ldm_stable, concept=old_text[0], device=device, num_samples=20)
                img_path = '/mlx_devbox/users/wangruipeng/playground/paper_run/uce_nullspace/unified-concept-editing-main/data/bias/bias/bias2/class'
                counts = CLIP_classification(im_path=img_path, attributes='a man,a woman', prompts_path='/mlx/mlx_devbox/users/wangruipeng/playground/paper_run/uce_nullspace/unified-concept-editing-main/data/profession3.csv', save_path=None, from_case=0, till_case=10000)
                
                if counts >= 11:
                    new_text = [f'female {old_text[0]}']
                elif counts <= 9:
                    new_text = [f'male {old_text[0]}']
                    
                log_file.write(f"new_text old_text: {[old_text[0], new_text]}\n")  # Log new and old text
                log_file.flush()
                
                # model_save_path_with_index = args.model_save_path.replace('.pt', f'_bias_batch_{batch_index}_lamb_{lamb}_num_smallest_{num_smallest_singular}.pt')
                # concepts_save_path_with_index = args.concepts_save_path.replace('.txt', f'_bias_batch_{batch_index}_lamb_{lamb}_num_smallest_{num_smallest_singular}.txt')
                
                ldm_stable, cache_c = alpha_edit_3(ldm_stable=ldm_stable, old_text_=old_text, new_text_=new_text, add=False, retain_text_=retain_texts, lamb=lamb, erase_scale=erase_scale, preserve_scale=preserve_scale, technique=technique, lamda=lamda, P=P, cache_c=cache_c)
                generate_images_1(ldm_stable, concept=old_text[0], device=device, num_samples=20)
                counts = CLIP_classification(im_path=img_path, attributes='a man,a woman', prompts_path='/mlx/mlx_devbox/users/wangruipeng/playground/paper_run/uce_nullspace/unified-concept-editing-main/data/profession3.csv', save_path=None, from_case=0, till_case=10000)
                counts_save.append(counts)
                # log_file.write(f'counts_save: {counts_save}\n')  # Log counts_save
                
                min_difference = min(abs(10 - x) for x in counts_save)
                # Directly find the optimal point
                if counts == 10:
                    # Save the model
                    model_save_path_with_index = args.model_save_path.replace('.pt', f'_bias_batch_{batch_index}_lamb_{lamb}_num_smallest_{num_smallest_singular}.pt')
                    concepts_save_path_with_index = args.concepts_save_path.replace('.txt', f'_bias_batch_{batch_index}_lamb_{lamb}_num_smallest_{num_smallest_singular}.txt')
                    torch.save(ldm_stable.unet.state_dict(), model_save_path_with_index)
                    # Save the concepts
                    with open(concepts_save_path_with_index, 'w') as fp:
                        json.dump(concepts, fp)
                    # Update the last saved model state
                    last_saved_model_state = ldm_stable.unet.state_dict()
                    ldm_stable.unet.load_state_dict(last_saved_model_state)
                    cache_c_last = cache_c   
                    torch.cuda.empty_cache()
                    log_file.write("yes, find it!\n")  # Log the message
                    log_file.flush()
                    break  # Jump to the next i
                # Find a new better point: get_saved_model_state records the model weights, but still load the initial model weights, only change the model weights when the optimal is found or the loop ends
                elif abs(10 - counts) <= min_difference:
                    # If the condition is not met, revert to the last saved model state
                    # Save the path
                    model_save_path_with_index = args.model_save_path.replace('.pt', f'_bias_batch_{batch_index}_lamb_{lamb}_num_smallest_{num_smallest_singular}.pt')
                    concepts_save_path_with_index = args.concepts_save_path.replace('.txt', f'_bias_batch_{batch_index}_lamb_{lamb}_num_smallest_{num_smallest_singular}.txt')
                    get_saved_model_state = ldm_stable.unet.state_dict()
                    ldm_stable.unet.load_state_dict(last_saved_model_state)
                    cache_c = cache_c_last
                    # ldm_stable.unet.load_state_dict(last_saved_model_state)
                    log_file.write("find a better one\n")  # Log the message
                    log_file.flush()
                    torch.cuda.empty_cache()
                # If the above conditions are not met, just load the initial model
                else:
                    ldm_stable.unet.load_state_dict(last_saved_model_state)   
                    cache_c = cache_c_last
                if num_smallest_singular == 445:
                    # Save the model
                    # model_save_path_with_index = args.model_save_path.replace('.pt', f'_bias_batch_{batch_index}_lamb_{lamb}_num_smallest_{num_smallest_singular}.pt')
                    rate = min(abs(10 - x) for x in counts_save)
                    log_file.write(f'rate: {rate}\n')  # Log the rate
                    log_file.flush()
                    ldm_stable.unet.load_state_dict(get_saved_model_state)
                    last_saved_model_state = ldm_stable.unet.state_dict()
                    cache_c_last = cache_c 
                    torch.cuda.empty_cache()
                    log_file.write("only can do this\n")  # Log the message
                    log_file.flush()
                    torch.save(ldm_stable.unet.state_dict(), model_save_path_with_index)
                    # Save the concepts
                    with open(concepts_save_path_with_index, 'w') as fp:
                        json.dump(concepts, fp)
            log_file.write(f'counts_save: {counts_save}\n')
            log_file.flush()
