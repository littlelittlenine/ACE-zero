# erase single-one
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
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline
import abc
import copy
from functools import reduce
import operator
import pandas as pd
from tqdm import tqdm
import pdb
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from diffusers import LMSDiscreteScheduler
import spacy

# 加载 spaCy 的英文语言模型
nlp = spacy.load("/mnt/bn/intern-disk/mlx/users/wrp/ecu-nullspace/english_tok/")

def extract_noun_phrases(prompt):
    """
    从 prompt 中提取名词短语。
    
    参数:
        prompt (str): 输入的文本 prompt。
    
    返回:
        list: 提取的名词短语列表。
    """
    doc = nlp(prompt)  # 使用 spaCy 解析 prompt
    noun_phrases = [chunk.text for chunk in doc.noun_chunks]  # 提取名词短语
    return noun_phrases
# 初始化clip模型
# 单独擦除nudity，两个都擦除
clip_model = CLIPModel.from_pretrained("/mnt/bn/intern-disk/mlx/users/wrp/ecu-nullspace/clip-vit")
clip_processor = CLIPProcessor.from_pretrained("/mnt/bn/intern-disk/mlx/users/wrp/ecu-nullspace/clip-vit")
## get arguments for our script
# 获得脚本参数
with_to_k = False
with_augs = True
train_func = "train_closed_form"

### load model
LOW_RESOURCE = False
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77

def get_project_input(ldm_stable, data_path, subject_column='subject', num_smallest_singular=300, token_nums=1000,  batch_size=16):
    data = pd.read_csv(data_path)
    data = [artist for artist in data[subject_column] if isinstance(artist, str)]
    total_embeddings = None
    print(len(data))
    data = data[:token_nums]
    for i in tqdm(range(0, len(data), batch_size)):  # 使用 tqdm 监控进度
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
            text_embeddings = text_embeddings.detach()  
            batch_embeddings = []
            for j, idx in enumerate(idx_list):
                batch_embeddings.append(text_embeddings[j, idx, :].unsqueeze(0))
            batch_embeddings = torch.cat(batch_embeddings, dim=0)
        text_embeddings = batch_embeddings
        if total_embeddings is None:
            total_embeddings = text_embeddings
        else:
            total_embeddings = torch.cat((total_embeddings, text_embeddings), dim=0)
        del text_input, text_embeddings
        torch.cuda.empty_cache()  
    product = total_embeddings.T @ total_embeddings

    U, S, _ = torch.linalg.svd(product, full_matrices=False)
    smallest_indices = torch.topk(S, num_smallest_singular, largest=False).indices
    smallest_indices = smallest_indices.sort().values
    projection_matrix = U[:, smallest_indices] @ U[:, smallest_indices].T
    print("Projection matrix size:", projection_matrix.size())
    
    return projection_matrix
def edit_model(ldm_stable, old_text_, new_text_, retain_text_, add=False, layers_to_edit=None, lamb=0.1, erase_scale=0.1, preserve_scale=0.1, with_to_k=True, technique='tensor', P=None, P_out=None, P_Q_out=None, lamda=10, mode = 'no'):

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
            for_mat1 = torch.eye(projection_matrices[layer_num].weight.shape[1],dtype=torch.float,device = projection_matrices[layer_num].weight.device)
            for_mat2 = torch.zeros(768,768, device=W_old.device)
            for_mat3 = torch.zeros(projection_matrices[layer_num].weight.shape[0],768, device=W_old.device)

            context = None
            value_vector = None
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
                new_embs = projection_matrices[layer_num](new_emb).detach() 

                context = old_emb.detach() 

            Kp1 = P  
            Kp2 = P

            print("context:", context.size())
            if mode == 'left':
                new_embs = new_embs @ P_out[layer_num] 
            elif mode == 'right':
                new_embs = new_embs @ P_out[(layer_num+16)%32]
            elif mode == 'q' and layer_num >= 16:

                new_embs = new_embs @ P_Q_out[layer_num - 16]            
            else:
                new_embs = new_embs
            context_vector = context.reshape(context.shape[0], context.shape[1], 1) 
            context_vector_T = context.reshape(context.shape[0], 1, context.shape[1]) 
            value_vector = new_embs.reshape(new_embs.shape[0], new_embs.shape[1], 1)  
            for_mat2 += (context_vector @ context_vector_T).sum(dim=0)      
            for_mat3 += (value_vector @ context_vector_T).sum(dim=0)

        result1 = for_mat2 @ Kp1 + lamda * for_mat1
        result2 = (for_mat3 - W_old @ for_mat2) @ Kp2
        upd_matrix = torch.linalg.solve(
            result1.transpose(0, 1), # 这个的转置没有什么用处
            result2.transpose(0, 1)
        )
        projection_matrices[layer_num].weight = torch.nn.Parameter(W_old + upd_matrix.T)
    print(f'当前模型状态: 将"{str(old_text_)}"编辑为"{str(new_texts)}"，并保留"{str(ret_texts)}"')
    return ldm_stable
def edit_model_copy(ldm_stable, old_text_, new_text_, retain_text_, add=False, layers_to_edit=None, lamb=0.1, erase_scale=0.1, preserve_scale=0.1, with_to_k=True, technique='tensor', P=None, P_out=None, P_Q_out=None, lamda=10, mode = 'no'):

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

                old_embeddings.append(old_emb) # (768)
                new_embs = projection_matrices[layer_num](new_emb).detach() # (320)
                new_embeddings.append(new_embs)
            old_embs = torch.cat(old_embeddings, dim=0)  # (1, 768)
            new_embs = torch.cat(new_embeddings, dim=0) 
            Kp1 = P  
            Kp2 = P
            context = old_embs.detach()

            if mode == 'left':
                new_embs = new_embs @ P_out[layer_num] # (1, 320)
            elif mode == 'right':
                new_embs = new_embs @ P_out[(layer_num+16)%32]
            elif mode == 'q' and layer_num >= 16:
                new_embs = new_embs @ P_Q_out[layer_num - 16] # (1, 320)            
            else:
                new_embs = new_embs  
            for_mat1 = torch.eye(projection_matrices[layer_num].weight.shape[1],dtype=torch.float,device = projection_matrices[layer_num].weight.device)  
            for_mat2 = (context.T @ context) # (768,768)
            for_mat3 = new_embs.T @ context - W_old @ for_mat2
        result1 = for_mat2 @ Kp1 + 0.5 * for_mat1
        result2 = for_mat3 @ Kp2
        upd_matrix = torch.linalg.solve(
            result1.transpose(0, 1), 
            result2.transpose(0, 1)
        )
        projection_matrices[layer_num].weight = torch.nn.Parameter(W_old + upd_matrix.T)
    print(f'当前模型状态: 将"{str(old_text_)}"编辑为"{str(new_texts)}"，并保留"{str(ret_texts)}"')
    return ldm_stable
def hook_fn(module, input, output):
    """钩子函数，用于捕获 to_q 的输出，保留在 GPU 上"""
    conditional_output = output[output.shape[0] // 2:]  
    hook_fn.outputs.append(conditional_output)  
def compute_projection_matrices(final_wq_outputs, num_smallest_singular=20):
    """
    Calculate the projection matrix for each layer from final_wq_outputs.

    parameter:
        final_wq_outputs (list): List containing Wq outputs for each layer.
        num_smallest_singular (int): The minimum number of singular values ​​selected.

    return:
        list: A list containing the projection matrices for each layer.
    """
    P_outs = []  # 存储每一层的投影矩阵

    for layer_idx, total_embeddings in enumerate(final_wq_outputs):
        # 计算协方差矩阵
        product = torch.mm(total_embeddings.T, total_embeddings)
        print(f"Layer {layer_idx} - Product size:", product.size())

        # 进行 SVD 分解
        U, S, _ = torch.linalg.svd(product, full_matrices=False)
        print(f"Layer {layer_idx} - Singular values size:", S.size())

        # 选择最小的 N 个奇异值的索引
        smallest_indices = torch.topk(S, num_smallest_singular, largest=False).indices
        smallest_indices = smallest_indices.sort().values

        # 计算投影矩阵
        projection_matrix = U[:, smallest_indices] @ U[:, smallest_indices].T
        print(f"Layer {layer_idx} - Projection matrix size:", projection_matrix.size())

        P_outs.append(projection_matrix)

    return P_outs

def compute_projection_matrices_ratio(final_wq_outputs, singular_value_ratio=0.1):
    """
    从 final_wq_outputs 计算每一层的投影矩阵。

    参数:
        final_wq_outputs (list): 包含每一层 Wq 输出的列表。
        singular_value_ratio (float): 选择的最小奇异值比例（默认为 10%）。

    返回:
        list: 包含每一层投影矩阵的列表。
    """
    P_outs = []  # 存储每一层的投影矩阵

    for layer_idx, total_embeddings in enumerate(final_wq_outputs):
        # 计算协方差矩阵
        product = torch.mm(total_embeddings.T, total_embeddings)
        print(f"Layer {layer_idx} - Product size:", product.size())

        # 根据 product 的维度计算奇异值数量
        num_singular_values = product.size(0)  # product 是一个方阵，大小为 [d, d]
        num_smallest_singular = max(1, int(num_singular_values * singular_value_ratio))  # 至少选择 1 个奇异值
        print(f"Layer {layer_idx} - Number of smallest singular values to keep:", num_smallest_singular)

        # 进行 SVD 分解
        U, S, _ = torch.linalg.svd(product, full_matrices=False)
        print(f"Layer {layer_idx} - Singular values size:", S.size())

        # 选择最小的 N 个奇异值的索引
        smallest_indices = torch.topk(S, num_smallest_singular, largest=False).indices
        smallest_indices = smallest_indices.sort().values

        # 计算投影矩阵
        projection_matrix = U[:, smallest_indices] @ U[:, smallest_indices].T
        print(f"Layer {layer_idx} - Projection matrix size:", projection_matrix.size())

        P_outs.append(projection_matrix)

    return P_outs

def generate_images_toget_Q(model_name, save_path, device='cuda:0', guidance_scale=7.5, image_size=512, ddim_steps=10, num_samples=1, from_case=0, till_case=1000000, base='1.4', random_seed=42, prompt='image of a dog', project=0.1, with_to_k=True):
    # 1. Load the autoencoder model which will be used to decode the latents into image space.
    if base == '1.4':
        model_version = "/mnt/bn/intern-disk/mlx/users/wrp/ecu-nullspace/SDv1.4"
    elif base == '2.1':
        model_version = '/mnt/bn/intern-disk/mlx/users/wrp/ecu-nullspace/SDv2.1'
    else:
        model_version = "/mnt/bn/intern-disk/mlx/users/wrp/ecu-nullspace/SDv1.4"
    vae = AutoencoderKL.from_pretrained(model_version, subfolder="vae")

    # 2. Load the tokenizer and text encoder to tokenize and encode the text.
    tokenizer = CLIPTokenizer.from_pretrained(model_version, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_version, subfolder="text_encoder")

    # 3. The UNet model for generating the latents.
    unet = UNet2DConditionModel.from_pretrained(model_version, subfolder="unet")
    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    print("init done")
    vae.to(device)
    text_encoder.to(device)
    unet.to(device)
    torch_device = device
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    folder_path = f'{save_path}/{model_name.replace("diffusers-","").replace(".pt","_unsafe")}'
    os.makedirs(folder_path, exist_ok=True)

    # 存储每一层的 Wq 输出，使用列表
    sub_nets = ldm_stable.unet.named_children()
    ca_layers = []
    
    # 遍历所有子网络，收集交叉注意力层
    for net_name, net in sub_nets:
        if 'up' in net_name or 'down' in net_name or 'mid' in net_name:
            print(f"Processing {net_name} layer...")
            if 'up' in net_name or 'down' in net_name:
                for block in net:
                    if 'Cross' in block.__class__.__name__:
                        for attn in block.attentions:
                            for transformer in attn.transformer_blocks:
                                ca_layers.append(transformer.attn2)
                                print(f"Matrix shape in {net_name}: {transformer.attn2.to_k.weight.shape}")
            elif 'mid' in net_name:
                for attn in net.attentions:
                    for transformer in attn.transformer_blocks:
                        ca_layers.append(transformer.attn2)
                        print(f"Matrix shape in {net_name}: {transformer.attn2.to_k.weight.shape}")

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
    print(f"Number of projection matrices, Wk number: {len(projection_matrices)}")

    # extract COCO prompts
    filename = "/mlx/mlx_devbox/users/wangruipeng/playground/paper_run/uce_nullspace/unified-concept-editing-main/data/coco_30k.csv"
    data = pd.read_csv(filename)
    prompts = [artist for artist in data['prompt'] if isinstance(artist, str)]
    prompts = random.sample(prompts, 1200)
    all_embeddings = []

    # prompts
    for prompt in tqdm(prompts, desc="Processing prompts"):
        wq_outputs = [[] for _ in range(16)]  
        layer_order = []
        hooks = []
        hook_to_layer_idx = {}

        # hook
        layer_count = 0
        for layer in unet.down_blocks:
            if hasattr(layer, 'attentions'):
                for attn in layer.attentions:
                    for transformer in attn.transformer_blocks:
                        hook = transformer.attn2.to_q.register_forward_hook(hook_fn)
                        hooks.append(hook)
                        layer_order.append(('down', layer_count))
                        hook_to_layer_idx[len(hooks) - 1] = layer_count
                        layer_count += 1
        if hasattr(unet.mid_block, 'attentions'):
            for attn in unet.mid_block.attentions:
                for transformer in attn.transformer_blocks:
                    hook = transformer.attn2.to_q.register_forward_hook(hook_fn)
                    hooks.append(hook)
                    layer_order.append(('mid', layer_count))
                    hook_to_layer_idx[len(hooks) - 1] = layer_count
                    layer_count += 1
        for layer in unet.up_blocks:
            if hasattr(layer, 'attentions'):
                for attn in layer.attentions:
                    for transformer in attn.transformer_blocks:
                        hook = transformer.attn2.to_q.register_forward_hook(hook_fn)
                        hooks.append(hook)
                        layer_order.append(('up', layer_count))
                        hook_to_layer_idx[len(hooks) - 1] = layer_count
                        layer_count += 1
        print("layer_order:", layer_order)
        # Extract the noun subject in prompt
        noun_phrases = extract_noun_phrases(prompt)  
        if not noun_phrases:
            for hook in hooks:
                hook.remove()
            hooks.clear()
            hook_fn.outputs.clear()
            continue  

        # Compute embeddings for each noun subject
        k_embeddings = []
        for layer_num in range(len(projection_matrices)):  
            noun_embeddings = []
            for noun in noun_phrases:
                text_input = tokenizer([noun], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
                with torch.no_grad():
                    idx = text_input.attention_mask[0].sum().item() - 2  
                    text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
                    noun_embedding = text_embeddings[0, idx, :].unsqueeze(0)  
                    noun_embedding = projection_matrices[layer_num](noun_embedding).detach() 
                    noun_embeddings.append(noun_embedding)
            noun_embeddings = torch.cat(noun_embeddings, dim=0)  
            k_embeddings.append(noun_embeddings)
        k_embeddings = k_embeddings[16:]  

        height = image_size
        width = image_size
        num_inference_steps = ddim_steps
        guidance_scale = guidance_scale
        generator = torch.manual_seed(random_seed)
        batch_size = 1

        with torch.no_grad():
            text_input = tokenizer([prompt], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
            text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]

            max_length = text_input.input_ids.shape[-1]
            uncond_input = tokenizer(
                [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
            )
            uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]

            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

            latents = torch.randn(
                (batch_size, unet.in_channels, height // 8, width // 8),
                generator=generator,
            )
            latents = latents.to(torch_device)

            scheduler.set_timesteps(num_inference_steps)
            latents = latents * scheduler.init_noise_sigma

            for t in tqdm(scheduler.timesteps):
                hook_fn.outputs = []
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

                with torch.no_grad():
                    noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
                # print('length of hook_fn.outputs:', len(hook_fn.outputs))
                for hook_idx, output in enumerate(hook_fn.outputs):
                    if hook_idx in hook_to_layer_idx:
                        layer_idx = hook_to_layer_idx[hook_idx]
                        wq_outputs[layer_idx].append(output)
                    else:
                        print(f"Warning: hook_idx {hook_idx} not found in hook_to_layer_idx")
                        continue

                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                latents = scheduler.step(noise_pred, t, latents).prev_sample

            # Splice the Wq output of each layer
            final_wq_outputs = []
            for layer_idx in range(layer_count):
                wq_tensors = wq_outputs[layer_idx]
                if wq_tensors:  
                    concatenated_wq = torch.cat(wq_tensors, dim=0)  # (num_timesteps, ...)
                    print(f"Concatenated Wq for layer {layer_idx} shape: {concatenated_wq.shape}")
                    concatenated_wq = concatenated_wq.view(-1, concatenated_wq.size(-1))  # (num_timesteps * ..., hidden_dim)
                    print(f"Concatenated Wq for layer {layer_idx} shape after view: {concatenated_wq.shape}")
                    final_wq_outputs.append(concatenated_wq)
                else:
                    final_wq_outputs.append(torch.empty(0))  

            # Find the index of the Mid layer
            mid_layer_index = None
            for idx, (layer_type, layer_num) in enumerate(layer_order):
                if layer_type == 'mid':
                    mid_layer_index = idx
                    break

            if mid_layer_index is not None:
                mid_tensor = final_wq_outputs[mid_layer_index]
                final_wq_outputs.append(mid_tensor)
                final_wq_outputs.pop(mid_layer_index)
                print(f"Moved Mid layer (index {mid_layer_index}) to the end of final_wq_outputs.")
            else:
                print("Warning: Mid layer not found in final_wq_outputs.")

            print(f"Final Wq outputs shape: {[output.shape for output in final_wq_outputs]}")
            print(f"K embeddings shape: {[embedding.shape for embedding in k_embeddings]}")
            # Calculate similarity and select top1 embedding
            # if final_wq_outputs[-1].shape[1] != k_embeddings[-1].shape[1]:
            #     continue
            for layer_idx in range(layer_count):
                final_wq_output = final_wq_outputs[layer_idx]
                noun_embeddings = k_embeddings[layer_idx]
                a_normalized = final_wq_output / final_wq_output.norm(dim=1, keepdim=True)
                b_normalized = noun_embeddings / noun_embeddings.norm(dim=1, keepdim=True)
                scores = torch.mm(a_normalized, b_normalized.t())
                sum_scores = torch.sum(scores, dim=1)
                top1_index = torch.argmax(sum_scores)  
                top1_value = final_wq_output[top1_index].unsqueeze(0)  

                if len(all_embeddings) < 16:
                    all_embeddings.append(top1_value)
                else:
                    all_embeddings[layer_idx] = torch.cat([all_embeddings[layer_idx], top1_value], dim=0)

            del latents, noise_pred, text_embeddings, uncond_embeddings
            torch.cuda.empty_cache()

        for hook in hooks:
            hook.remove()
        hooks.clear()
        hook_fn.outputs.clear()

    projection_matrices = compute_projection_matrices_ratio(all_embeddings, singular_value_ratio=project)
    return projection_matrices

if __name__ == '__main__':
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
    parser.add_argument('--input_token_number', help='option to add additional prompts', type=int, required=False, default=False)
    parser.add_argument('--output_token_number', help='option to add additional prompts', type=int, required=False, default=False)
    parser.add_argument('--num_smallest_singular', help='option to add additional prompts', type=int, required=False, default=False)
    parser.add_argument('--lamb', help='option to add additional prompts', type=float, required=False, default=False)
    parser.add_argument('--output_singular', help='option to add additional prompts', type=int, required=False, default=100)
    parser.add_argument('--output_token_num', help='option to add additional prompts', type=int, required=False, default=1000)
    parser.add_argument('--mode', help='option to add additional prompts', type=str, required=False, default= 'no')
    parser.add_argument('--project', help='option to add additional prompts', type=float, required=False, default=0.01)
    parser.add_argument('--ddim_steps', help='option to add additional prompts', type=int, required=False, default=10)

    args = parser.parse_args()
    ddim_steps = args.ddim_steps
    project = args.project
    lamb = args.lamb
    mode = args.mode
    output_singular = args.output_singular
    output_token_num = args.output_token_num
    input_token_number = args.input_token_number
    num_smallest_singular = args.num_smallest_singular
    # output_token_number = args.output_token_number
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

    old_texts = []
    
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
    
    
    if preserve_concepts is None:
        if concept_type == 'art':
            df = pd.read_csv('data/artists1734_prompts.csv')

            retain_texts = list(df.artist.unique())
            old_texts_lower = [text.lower() for text in old_texts]
            preserve_concepts = [text for text in retain_texts if text.lower() not in old_texts_lower]
            if preserve_number is not None:
                print_text+=f'-preserving_{preserve_number}artists'
                preserve_concepts = random.sample(preserve_concepts, preserve_number)
        else:
            preserve_concepts = []

    retain_texts = ['']+preserve_concepts   
    if len(retain_texts) > 1:
        print_text+=f'-preserve_true'     
    else:
        print_text+=f'-preserve_false'
    if preserve_scale is None:
        preserve_scale = max(0.1, 1/len(retain_texts))
    sd14="/mnt/bn/intern-disk/mlx/users/wrp/ecu-nullspace/SDv1.4"
    sd21='/mnt/bn/intern-disk/mlx/users/wrp/ecu-nullspace/SDv2.1'
    if args.base=='1.4':
        model_version = sd14
    elif args.base=='2.1':
        model_version = sd21
    else:
        model_version = sd14
    ldm_stable = StableDiffusionPipeline.from_pretrained(model_version).to(device)
    print_text += f"-sd_{args.base.replace('.','_')}" 

    P_Q_out = generate_images_toget_Q(model_name='nude_1.0_1000_num_smallest_300_nudity.pt', save_path='/mlx/mlx_devbox/users/wangruipeng/playground/paper_run/uce_nullspace/unified-concept-editing-main/data/1', device=device,project=project, ddim_steps=ddim_steps)
    P = get_project_input(ldm_stable,'/mlx/mlx_devbox/users/wangruipeng/playground/paper_run/uce_nullspace/unified-concept-editing-main/data/extracted_subjects_big_only_1000.csv', num_smallest_singular=num_smallest_singular, token_nums=input_token_number, batch_size=16)
    P.to(device)
    print_text += f"-method_{technique}" 
    print_text = print_text.lower()
    print(print_text)
    model_save_path = '/mnt/bn/intern-disk/mlx/users/wrp/uce_nullspace/model_save/art_erase/rich_paper/nudity/single_output/nude.pt'
    concepts_save_path = '/mnt/bn/intern-disk/mlx/users/wrp/uce_nullspace/model_save/art_erase/rich_paper/nudity/single_output/nude.txt'
    model_save_path_with_index = model_save_path.replace('.pt', f'_no_{project}_nudity.pt')  
    concepts_save_path_with_index = concepts_save_path.replace('.txt', f'_no_{project}_nudity.txt')  
    ldm_stable = edit_model_copy(ldm_stable= ldm_stable, old_text_= old_texts, new_text_=new_texts, add=False, retain_text_= retain_texts, lamb=lamb, erase_scale = erase_scale, preserve_scale = preserve_scale,  technique=technique, P=P, P_out=None, P_Q_out=P_Q_out, mode=mode)
    torch.save(ldm_stable.unet.state_dict(), model_save_path_with_index)
    with open(concepts_save_path, 'w') as fp:
        json.dump(concepts, fp)