# 擦除
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
# 接受一个图像列表，并将它们排列成一个网格布局。网格具有指定的行数 (num_rows)，并且图像之间有一定的偏移量 (offset_ratio)
# images是输入，可能是单个图像，也可能是多个图像，也可能是一个四维的numpy数组
# 功能：在紧凑的网格格式中可视化一批图像
# 正交
def get_embedding(ldm_stable, text):
    texts = [text]  # 将要获取嵌入的文本放入列表中
    text_input = ldm_stable.tokenizer(
        texts,
        padding="max_length",
        max_length=ldm_stable.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    
    with torch.no_grad():  # 禁用梯度计算
        text_embeddings = ldm_stable.text_encoder(text_input.input_ids.to(ldm_stable.device))[0]
    
    # 提取索引为 2 的嵌入
    return text_embeddings[:, 2, :]  # 返回形状为 (1, 768) 的嵌入

def cosine_similarity(a, b):
    """计算两个向量的余弦相似度"""
    a_norm = a / a.norm(dim=-1, keepdim=True)  # 归一化向量 a
    b_norm = b / b.norm(dim=-1, keepdim=True)  # 归一化向量 b
    return (a_norm * b_norm).sum(dim=-1)  # 返回余弦相似度

def get_similar_token(ldm_stable, path, layers_to_edit=None, lamb=0.1,  
                      with_to_k=True, top_k=10000):
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

    # 获取 "nudity" 的嵌入
    nudity_embedding = get_embedding(ldm_stable, token)

    # 从指定路径加载 CSV 文件中的文本数据
    df = pd.read_csv(path)
    texts_to_check = df['subject'].tolist()  # 提取 'subject' 列的文本

    similarity_scores = []  # 用于存储相似度分数和文本的元组
    seen_texts = set()  # 用于跟踪已添加的文本

    # 遍历文本数据，计算与 "nudity" 的相似度
    for text in texts_to_check:
        # 确保不包括原始的 nudity token
        if text == token:
            continue  # 跳过相同的 token
        
        text_embedding = get_embedding(ldm_stable, text)
        similarity = cosine_similarity(nudity_embedding, text_embedding)
        
        # 将相似度分数和文本存储为元组
        if text not in seen_texts:  # 检查是否已添加
            similarity_scores.append((similarity.item(), text))
            seen_texts.add(text)  # 将文本添加到已见集合

    # 根据相似度排序并提取前 top_k 个文本
    similarity_scores.sort(key=lambda x: x[0], reverse=True)  # 降序排序
    top_similar_texts = [text for score, text in similarity_scores[:top_k]]

    # 将相似文本存储到 CSV 文件中
    if top_similar_texts:
        results_df = pd.DataFrame(top_similar_texts, columns=['subject'])
        results_df.to_csv('/mlx_devbox/users/wangruipeng/playground/paper_run/uce_nullspace/unified-concept-editing-main/data/nudity_similar_texts_10000.csv', index=False)
        print(f"相似文本已存储到类似文本的 CSV 文件中: nudity_similar_texts.csv")
    else:
        print("未找到相似文本。")

    return top_similar_texts  # 返回相似文本的列表
def project_to_null_space(a, P, b):
    # 将张量 a 投影到零空间
    a_proj =  a @ P
    # 将向量 b 投影到零空间的正交补
    b_proj = b - (b @ P)  # 原来 b 的投影
    # 计算 b_proj 的单位向量
    b_proj_norm = torch.norm(b_proj)
    if b_proj_norm > 0:
        b_proj_unit = b_proj / b_proj_norm  # 单位向量
    else:
        b_proj_unit = b_proj  # 如果 b_proj 为零，避免除以零
    # 计算调整量
    adjustment = torch.dot(a_proj.flatten(), b_proj_unit.flatten()) * b_proj_unit
    # 从 a_proj 中减去这个调整量
    a_final = a_proj - adjustment
    
    return a_final
def view_images(images, num_rows=3, offset_ratio=0.02):
    if type(images) is list:
        # 计算有多少图像需要用空白图像替换
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0
    # 创建与输入图像形状相同的空白图像
    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    # 将空白图像添加到输入图像列表中
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

# 函数执行文本到图像生成模型中的单个扩散步骤
# model：文本到图像生成模型。
# latents：当前潜在状态。
# context：上下文，包括文本嵌入。
# t：当前时间步。
# guidance_scale：引导比例。
# low_resource：一个可选标志，指示是否使用低资源模式
def diffusion_step(model, latents, context, t, guidance_scale, low_resource=False):
    # 计算噪声预测：在无条件和文本条件下预测噪声，完全相同的两个东西，只不过一个是通过时间换空间
    if low_resource:
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
        noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    # 计算最终的噪声预测：根据引导比例调整噪声预测
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    # 更新潜在状态
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    return latents

# 潜在空间表示转换为图像
def latent2image(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image

# 初始化潜在空间表示
def init_latent(latent, model, height, width, generator, batch_size):
    if latent is None:
        latent = torch.randn(
            (batch_size, model.unet.in_channels, height // 8, width // 8),
            generator=generator,
        )
    latents = latent.to(model.device)
    return latent, latents


@torch.no_grad()
# 调用上面的函数，生成图像
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

# ldm_stable：预训练的稳定扩散模型。
# test_text：用于生成图像的文本提示。
# num_samples：要生成的图像数量，默认为 9。
# seed：随机数生成器的种子，默认为 1231。
# 生成num_samples个图像，并返回它们的列表。
def generate_for_text(ldm_stable, test_text, num_samples = 9, seed = 1231):
    g = torch.Generator(device='cpu')
    g.manual_seed(seed)
    images = text2image_ldm_stable(ldm_stable, [test_text]*num_samples, latent=None, num_inference_steps=NUM_DIFFUSION_STEPS, guidance_scale=GUIDANCE_SCALE, generator=g, low_resource=LOW_RESOURCE)
    return view_images(images)
# 初始化clip模型
clip_model = CLIPModel.from_pretrained("/mnt/bn/intern-disk/mlx/users/wrp/ecu-nullspace/clip-vit")
clip_processor = CLIPProcessor.from_pretrained("/mnt/bn/intern-disk/mlx/users/wrp/ecu-nullspace/clip-vit")
# 用于计算给定概念和类别的图像生成模型的概率比率
# 函数接受多个参数，包括模型、先前的比率、比率差异、最大比率间隙、概念列表、类别列表、样本数量和循环次数
# 功能：计算特定概念和类别的概率比例
# 如果使用投影的话，我应该不需要这个概念
def get_ratios(ldm_stable, prev_ratio, ratio_diff, max_ratio_gap, concepts, classes, num_samples=10, num_loops=3):
    seeds = np.random.randint(5000,size=5) 
    ratios = []
    # 特定的概念
    for idx, concept in enumerate(concepts):
        if ratio_diff is not None:
            # 如果比率差异小于最大比率间隙，则跳过当前概念
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
                # 计算图像和文本相似度
                outputs = clip_model(**inputs)
                logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
                probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
                tmax = probs.max(1, keepdim=True)[0]
                mask = probs.ge(tmax)
                probs_full.append(mask.float())
                
        ratios.append(torch.cat(probs_full).mean(axis=0))
#     male = float(probs[0][0])
    return ratios

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
# 编辑模型
def edit_model(ldm_stable, old_text_, new_text_, retain_text_, add=False, layers_to_edit=None, lamb=0.1, erase_scale = 0.1, preserve_scale = 0.1, with_to_k=True, technique='tensor'):
    ### collect all the cross attns modules
    # 获取所有交叉注意力模块
    # 定义了一个名为 max_bias_diff 的变量，并将其赋值为 0.05，控制或限制某些操作中的偏差差异
    max_bias_diff = 0.05
    # 获取模型 unet 的所有子网络，并将其存储在 sub_nets 变量中。named_children() 返回一个元组列表，每个元组包含子网络的名称和模块
    sub_nets = ldm_stable.unet.named_children()
    # 空列表，存储所有的交叉注意力层
    ca_layers = []
    # 遍历所有子网络
    for net in sub_nets:
        # 检查子网络的名称是否包含 "down" 或 "up"，这意味着它是下采样或上采样层
        if 'up' in net[0] or 'down' in net[0]:
            for block in net[1]:
                if 'Cross' in block.__class__.__name__ :
                    for attn in block.attentions:
                        for  transformer in attn.transformer_blocks:
                            ca_layers.append(transformer.attn2)
        if 'mid' in net[0]:
            for attn in net[1].attentions:
                for  transformer in attn.transformer_blocks:
                    ca_layers.append(transformer.attn2)
    # 手机所有交叉注意力层的K矩阵和V矩阵，
    ### get the value and key modules，得到key和value模块
    projection_matrices = [l.to_v for l in ca_layers]
    og_matrices = [copy.deepcopy(l.to_v) for l in ca_layers]
    if with_to_k:
        projection_matrices = projection_matrices + [l.to_k for l in ca_layers]
        og_matrices = og_matrices + [copy.deepcopy(l.to_k) for l in ca_layers]

    ## reset the parameters
    # 重置交叉注意层的权重矩阵，将其恢复到原始状态
    # ca_layers是交叉注意力层的列表，og_matrices是原始的权重矩阵列表
    # 避免之前的编辑对当前编辑产生影响，可以确保每次编辑都是基于模型的原始状态进行的
    num_ca_clip_layers = len(ca_layers)
    for idx_, l in enumerate(ca_layers):
        l.to_v = copy.deepcopy(og_matrices[idx_])
        projection_matrices[idx_] = l.to_v
        # true是固定的
        if with_to_k:
            l.to_k = copy.deepcopy(og_matrices[num_ca_clip_layers + idx_])
            projection_matrices[num_ca_clip_layers + idx_] = l.to_k
    # 这段代码的主要目的是确保传入的参数在函数内部是正确的数据类型。通过使用 ast.literal_eval()，可以安全地将字符串参数解析为对应的 Python 数据类型，避免手动解析可能带来的错误。
    ### check the layers to edit (by default it is None; one can specify)
    layers_to_edit = ast.literal_eval(layers_to_edit) if type(layers_to_edit) == str else layers_to_edit
    lamb = ast.literal_eval(lamb) if type(lamb) == str else lamb
    print("layers_to_edir", layers_to_edit)    
    ### Format the edits
    old_texts = []
    new_texts = []
    for old_text, new_text in zip(old_text_, new_text_):
        old_texts.append(old_text)
        n_t = new_text
        if n_t == '':
            n_t = ' '
        new_texts.append(n_t)
    if retain_text_ is None:
        ret_texts = ['']
        retain = False
    else:
        ret_texts = retain_text_
        retain = True

    print(old_texts, new_texts)
    ######################## START ERASING ###################################
    for layer_num in range(len(projection_matrices)):
        # print('projection_matrices[layer_num].weight', projection_matrices[layer_num].weight.shape)
        # projection = torch.load("/mnt/bn/intern-disk/mlx/users/wrp/uce_nullspace/model_save/nullspace_project/output_2000_100/projection_matrix_{}.pt".format((layer_num+16)%32))
        print("layers_to_edit", layers_to_edit) 
        if (layers_to_edit is not None) and (layer_num not in layers_to_edit): # 如果不是这样的话就是所有层全部编辑！！！！！！！！
            continue
        print(f'Editing layer {layer_num}')
        #### prepare input k* and v*
        with torch.no_grad():
            #mat1 = \lambda W + \sum{v k^T}
            mat1 = lamb * projection_matrices[layer_num].weight
            #mat2 = \lambda I + \sum{k k^T}
            mat2 = lamb * torch.eye(projection_matrices[layer_num].weight.shape[1], device = projection_matrices[layer_num].weight.device)
            for cnt, t in enumerate(zip(old_texts, new_texts)):
                old_text = t[0]
                new_text = t[1]
                texts = [old_text, new_text]
                text_input = ldm_stable.tokenizer(
                    texts,
                    padding="max_length",
                    max_length=ldm_stable.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_embeddings = ldm_stable.text_encoder(text_input.input_ids.to(ldm_stable.device))[0]
                # (2, 77, 768)  text_input.attention_mask[0]是一个二进制掩码，其中1表示有效token，0表示padding token
                # -2去除CLS 和 EOS token
                final_token_idx = text_input.attention_mask[0].sum().item()-2
                final_token_idx_new = text_input.attention_mask[1].sum().item()-2
                farthest = max([final_token_idx_new, final_token_idx])
                old_emb = text_embeddings[0] # (77, 768)
                old_emb = old_emb[final_token_idx:len(old_emb)-max(0,farthest-final_token_idx)] # （76，768）
                new_emb = text_embeddings[1] # (77, 768)
                new_emb = new_emb[final_token_idx_new:len(new_emb)-max(0,farthest-final_token_idx_new)] # （76，768）               
                context = old_emb.detach()
                print("context:",old_text)
                values = []
                with torch.no_grad():
                    for layer_number, layer in enumerate(projection_matrices):
                        # projection = torch.load("/mnt/bn/intern-disk/mlx/users/wrp/uce_nullspace/model_save/nullspace_project/output_5000_5/projection_matrix_{}.pt".format((layer_number+16)%32))
                        # print("layer.weight.shape", layer.weight.shape)
                        o_embs = layer(old_emb).detach()
                        if technique == 'tensor':
                            o_embs = layer(old_emb).detach()
                            # u = o_embs
                            # u = u / u.norm()
                            new_embs = layer(new_emb).detach()
                            # new_emb_proj = (u*new_embs).sum()
                            target = project_to_null_space(new_embs, projection, o_embs)
                            # target = new_embs - (new_emb_proj)*u
                            values.append(target.detach()) 
                        elif technique == 'replace': # 默认值是replace
                            # 将context替换成new_emb
                            # 计算旧嵌入
                            # o_embs = layer(old_emb).detach()  # (3, 128)
                            # # 计算新嵌入
                            # layer_new_emb = layer(new_emb)  # (3, 128)
                            # # 计算新嵌入在旧嵌入的空间上的投影
                            # # 首先，计算旧嵌入的单位向量
                            # u = o_embs / o_embs.norm(dim=1, keepdim=True)  # (3, 128)
                            # # 接下来，计算 `layer_new_emb` 在 `o_embs` 空间的投影
                            # # 计算投影系数
                            # projection_coef = (layer_new_emb @ u.T)  # (3, 128) @ (128, 3) -> (3, 3)
                            # # 计算投影值
                            # projected_values = projection_coef @ u  # (3, 3) @ (3, 128) -> (3, 128)

                            # # 计算 `layer_new_emb` 在 `o_embs` 的正交分量
                            # orthogonal_component = layer_new_emb - projected_values
                            values.append(layer(new_emb).detach())
                            # print("values",values[-1].size())
                        else:
                            values.append(layer(new_emb).detach())
                #  先乘法再求和和先求和再乘的区别
                context_vector = context.reshape(context.shape[0], context.shape[1], 1) # (76, 768, 1)
                context_vector_T = context.reshape(context.shape[0], 1, context.shape[1]) # (76, 1, 768)
                value_vector = values[layer_num].reshape(values[layer_num].shape[0], values[layer_num].shape[1], 1) # (76, 1280, 1)
                for_mat1 = (value_vector @ context_vector_T).sum(dim=0) # （1280，768）进行了求和操作，为什么
                for_mat2 = (context_vector @ context_vector_T).sum(dim=0) # （768，768）
                # print("for_mat1",for_mat1.size())
                # print("mat1",mat1.size())
                mat1 += erase_scale*for_mat1
                mat2 += erase_scale*for_mat2
            for old_text, new_text in zip(ret_texts, ret_texts):
                print("old_text",old_text)
                old_text = str(old_text) if old_text is not None else ""
                new_text = str(new_text) if new_text is not None else ""
                text_input = ldm_stable.tokenizer(
                    [old_text, new_text],
                    padding="max_length",
                    max_length=ldm_stable.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                # print("text_input",len(text_input))
                text_embeddings = ldm_stable.text_encoder(text_input.input_ids.to(ldm_stable.device))[0]
                # print("text_embeddings",text_embeddings.size())
                old_emb, new_emb = text_embeddings
                context = old_emb.detach()
                values = []
                with torch.no_grad():
                    for layer in projection_matrices:
                        values.append(layer(new_emb[:]).detach())
                context_vector = context.reshape(context.shape[0], context.shape[1], 1)
                context_vector_T = context.reshape(context.shape[0], 1, context.shape[1])
                value_vector = values[layer_num].reshape(values[layer_num].shape[0], values[layer_num].shape[1], 1)
                for_mat1 = (value_vector @ context_vector_T).sum(dim=0)
                for_mat2 = (context_vector @ context_vector_T).sum(dim=0)
                # print("for_mat1",for_mat1.size())
                # print("mat1",mat1.size())
                mat1 += preserve_scale*for_mat1
                mat2 += preserve_scale*for_mat2
                #update projection matrix
            projection_matrices[layer_num].weight = torch.nn.Parameter(mat1 @ torch.inverse(mat2))

    print(f'Current model status: Edited "{str(old_text_)}" into "{str(new_texts)}" and Retained "{str(retain_text_)}"')
    return ldm_stable


def edit_model_deta(ldm_stable, old_text_, new_text_, retain_text_, add=False, layers_to_edit=None, lamb=0.1, erase_scale = 0.1, preserve_scale = 0.1, with_to_k=True, technique='tensor'):
    ### collect all the cross attns modules
    # 获取所有交叉注意力模块
    # 定义了一个名为 max_bias_diff 的变量，并将其赋值为 0.05，控制或限制某些操作中的偏差差异
    max_bias_diff = 0.05
    # 获取模型 unet 的所有子网络，并将其存储在 sub_nets 变量中。named_children() 返回一个元组列表，每个元组包含子网络的名称和模块
    sub_nets = ldm_stable.unet.named_children()
    # 空列表，存储所有的交叉注意力层
    ca_layers = []
    # 遍历所有子网络
    for net in sub_nets:
        # 检查子网络的名称是否包含 "down" 或 "up"，这意味着它是下采样或上采样层
        if 'up' in net[0] or 'down' in net[0]:
            for block in net[1]:
                if 'Cross' in block.__class__.__name__ :
                    for attn in block.attentions:
                        for  transformer in attn.transformer_blocks:
                            ca_layers.append(transformer.attn2)
        if 'mid' in net[0]:
            for attn in net[1].attentions:
                for  transformer in attn.transformer_blocks:
                    ca_layers.append(transformer.attn2)
    # 手机所有交叉注意力层的K矩阵和V矩阵，
    ### get the value and key modules，得到key和value模块
    projection_matrices = [l.to_v for l in ca_layers]
    og_matrices = [copy.deepcopy(l.to_v) for l in ca_layers]
    if with_to_k:
        projection_matrices = projection_matrices + [l.to_k for l in ca_layers]
        og_matrices = og_matrices + [copy.deepcopy(l.to_k) for l in ca_layers]

    ## reset the parameters
    # 重置交叉注意层的权重矩阵，将其恢复到原始状态
    # ca_layers是交叉注意力层的列表，og_matrices是原始的权重矩阵列表
    # 避免之前的编辑对当前编辑产生影响，可以确保每次编辑都是基于模型的原始状态进行的
    num_ca_clip_layers = len(ca_layers)
    for idx_, l in enumerate(ca_layers):
        l.to_v = copy.deepcopy(og_matrices[idx_])
        projection_matrices[idx_] = l.to_v
        # true是固定的
        if with_to_k:
            l.to_k = copy.deepcopy(og_matrices[num_ca_clip_layers + idx_])
            projection_matrices[num_ca_clip_layers + idx_] = l.to_k
    # 这段代码的主要目的是确保传入的参数在函数内部是正确的数据类型。通过使用 ast.literal_eval()，可以安全地将字符串参数解析为对应的 Python 数据类型，避免手动解析可能带来的错误。
    ### check the layers to edit (by default it is None; one can specify)
    layers_to_edit = ast.literal_eval(layers_to_edit) if type(layers_to_edit) == str else layers_to_edit
    lamb = ast.literal_eval(lamb) if type(lamb) == str else lamb
    print("layers_to_edir", layers_to_edit)    
    ### Format the edits
    old_texts = []
    new_texts = []
    for old_text, new_text in zip(old_text_, new_text_):
        old_texts.append(old_text)
        n_t = new_text
        if n_t == '':
            n_t = ' '
        new_texts.append(n_t)
    if retain_text_ is None:
        ret_texts = ['']
        retain = False
    else:
        ret_texts = retain_text_
        retain = True

    print(old_texts, new_texts)
    ######################## START ERASING ###################################
    for layer_num in range(len(projection_matrices)):
        print("layers_to_edit", layers_to_edit) 
        if (layers_to_edit is not None) and (layer_num not in layers_to_edit): # 如果不是这样的话就是所有层全部编辑！！！！！！！！
            continue
        print(f'Editing layer {layer_num}')
        #### prepare input k* and v*
        with torch.no_grad():
            #mat1 = \lambda W + \sum{v k^T}
            mat1 = lamb * projection_matrices[layer_num].weight
            #mat2 = \lambda I + \sum{k k^T}
            mat2 = lamb * torch.eye(projection_matrices[layer_num].weight.shape[1], device = projection_matrices[layer_num].weight.device)
            for cnt, t in enumerate(zip(old_texts, new_texts)):
                old_text = t[0]
                new_text = t[1]
                texts = [old_text, new_text]
                text_input = ldm_stable.tokenizer(
                    texts,
                    padding="max_length",
                    max_length=ldm_stable.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_embeddings = ldm_stable.text_encoder(text_input.input_ids.to(ldm_stable.device))[0]
                # (2, 77, 768)  text_input.attention_mask[0]是一个二进制掩码，其中1表示有效token，0表示padding token
                # -2去除CLS 和 EOS token
                final_token_idx = text_input.attention_mask[0].sum().item()-2
                final_token_idx_new = text_input.attention_mask[1].sum().item()-2
                farthest = max([final_token_idx_new, final_token_idx])
                old_emb = text_embeddings[0] # (77, 768)
                old_emb = old_emb[final_token_idx:len(old_emb)-max(0,farthest-final_token_idx)] # （76，768）
                new_emb = text_embeddings[1] # (77, 768)
                new_emb = new_emb[final_token_idx_new:len(new_emb)-max(0,farthest-final_token_idx_new)] # （76，768）
                # 新加上的正交代码：在输出端k和对应的输出v正交
                o_embs = layer(old_emb).detach()
                max_min_difference = o_embs.max() - o_embs.min()
                print("max_diff:",max_min_difference)
                projection = torch.load("/mnt/bn/intern-disk/mlx/users/wrp/uce_nullspace/model_save/nullspace_project/output_2000_100/projection_matrix_{}.pt".format((layer_num+16)%32))
                context = old_emb.detach()
                values = []
                with torch.no_grad():
                    for layer in projection_matrices:
                        if technique == 'tensor':
                            o_embs = layer(old_emb).detach()
                            max_min_difference = o_embs.max() - o_embs.min()
                            print("max_diff:",max_min_difference)
                            u = o_embs
                            u = u / u.norm()
                            
                            new_embs = layer(new_emb).detach()
                            new_emb_proj = (u*new_embs).sum()
                            
                            target = new_embs - (new_emb_proj)*u
                            values.append(target.detach()) 
                        elif technique == 'replace': # 默认值是replace
                            # 将context替换成new_emb
                            o_embs = layer(old_emb).detach()
                            layer_new_emb = layer(new_emb)
                            projection = (layer_new_emb @ o_embs.T)  # (3, 128) @ (128, 3) -> (3, 3)
                            projected_values = projection @ o_embs  # (3, 3) @ (3, 128) -> (3, 128)
                            # 将 layer(new_emb) 投影到 o_embs 的正交方向上
                            orthogonal_component = layer_new_emb - projected_values
                            values.append(orthogonal_component.detach())
                            # print("values",values[-1].size())
                        else:
                            values.append(layer(new_emb).detach())
                # 先乘法再求和和先求和再乘的区别
                context_vector = context.reshape(context.shape[0], context.shape[1], 1) # (75, 768, 1)
                context_vector_T = context.reshape(context.shape[0], 1, context.shape[1]) # (75, 1, 768)
                value_vector = values[layer_num].reshape(values[layer_num].shape[0], values[layer_num].shape[1], 1) # (76, 1280, 1)
                o_embs = projection_matrices[layer_num](old_emb).detach()
                for_mat1 = (value_vector @ context_vector_T).sum(dim=0) # （1280，768）进行了求和操作，为什么
                for_mat2 = (context_vector @ context_vector_T).sum(dim=0) # （768，768）
                R = ((value_vector - o_embs.reshape(values[layer_num].shape[0], values[layer_num].shape[1], 1)) @ context_vector_T).sum(dim=0) # （1280，768）
                # mat1 += erase_scale*for_mat1
                mat2 += for_mat2 
            for old_text, new_text in zip(ret_texts, ret_texts):
                text_input = ldm_stable.tokenizer(
                    [old_text, new_text],
                    padding="max_length",
                    max_length=ldm_stable.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                # print("text_input",len(text_input))
                text_embeddings = ldm_stable.text_encoder(text_input.input_ids.to(ldm_stable.device))[0]
                # print("text_embeddings",text_embeddings.size())
                old_emb, new_emb = text_embeddings
                context = old_emb.detach()
                values = []
                with torch.no_grad():
                    for layer in projection_matrices:
                        values.append(layer(new_emb[:]).detach())
                context_vector = context.reshape(context.shape[0], context.shape[1], 1)
                context_vector_T = context.reshape(context.shape[0], 1, context.shape[1])
                value_vector = values[layer_num].reshape(values[layer_num].shape[0], values[layer_num].shape[1], 1)
                for_mat1 = (value_vector @ context_vector_T).sum(dim=0)
                for_mat2 = (context_vector @ context_vector_T).sum(dim=0)
                # mat1 += preserve_scale*for_mat1
                mat2 += for_mat2
                #update projection matrix
            # R: （320，768） mat2: (768，768)
            upd_matrix = torch.linalg.solve(
                mat2, 
                R.T
            )
            projection_matrices[layer_num].weight = torch.nn.Parameter(upd_matrix.T+projection_matrices[layer_num].weight.detach())  # 使用 .detach() 防止反向传播)
            print("edit done")

    print(f'Current model status: Edited "{str(old_text_)}" into "{str(new_texts)}" and Retained "{str(retain_text_)}"')
    return ldm_stable
# 但是比如30k的数据叠加在一起是(30000, 77, 768)维度的输入数据，那么这个时候输入的零空间如何寻找呢
if __name__ == '__main__':
    seed_value = 1234  # 选择一个合适的整数作为种子
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # 如果使用多个GPU，确保所有GPU的种子都被设置
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

    args = parser.parse_args()
    model_save_path = args.model_save_path
    concepts_save_path = args.concepts_save_path
    technique = args.technique
    device = f'cuda:{args.device}'
    preserve_scale = args.preserve_scale
    erase_scale = args.erase_scale
    add_prompts = args.add_prompts
    guided_concepts = args.guided_concepts
    preserve_concepts = args.preserve_concepts
    preserve_number = args.preserve_number
    concepts = args.concepts.split(',') # 得到概念列表[violence, nudity, harm]
    concepts = [con.strip() for con in concepts] # 去除空格
    concept_type = args.concept_type # 得到概念类型
    print_text=''
    for txt in concepts:
        print_text += txt.lower()+'_' # 拼接成字符串,形式是violence_nudity_harm_
    print_text  = print_text[:-1]     # 去除最后一个下划线
    # PROMPT CLEANING
    if concepts[0] == 'allartist':
        concepts = ["Kelly Mckernan", "Thomas Kinkade", "Pablo Picasso", "Tyler Edlin", "Kilian Eng"]
    if concepts[0] == 'i2g':
        concepts = ['hate', 'harassment', 'violence', 'suffering', 'humiliation', 'harm', 'suicide', 'sexual', 'nudity', 'bodily fluids', 'blood']
    if concepts[0] == '10artists':
        concepts = ["Asger Jorn", "Eric Fischl", "Johannes Vermeer", "Apollinary Vasnetsov", "Naoki Urasawa", "Nicolas Mignard", "John Whitcomb", "John Constable", "Warwick Globe", "Albert Marquet"]
    if concepts[0] == 'imagenette':
        concepts = ['Cassette Player', 'Chain Saw', 'Church', 'Gas Pump', 'Tench', 'Garbage Truck', 'English Springer', 'Golf Ball', 'Parachute', 'French Horn']
    # type为unsafe的时候，这段代码是无用的
    if 'artists' in concepts[0]:
        # df = pd.read_csv('data/artists1734_prompts.csv')
        df = pd.read_csv('/mlx_devbox/users/wangruipeng/playground/paper_run/uce_nullspace/unified-concept-editing-main/data/erase_artist/erase_nude_artists_1000.csv')
        artists = list(df.Artist.unique())
        # number = int(concepts[0].replace('artists', ''))
        # concepts = random.sample(artists,number)
        # concepts = artists[:999]+['nudity']
        concepts = artists
        # concepts = [concept for concept in concepts if concept != 'nudity']
        # print('artists:',concepts)
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
    if not add_prompts:
        additional_prompts = []
    concepts_ = []
    # 
    for concept in concepts:
        old_texts.append(f'{concept}')
        for prompt in additional_prompts:
            old_texts.append(prompt.format(concept=concept))
        length = 1 + len(additional_prompts)
        concepts_.extend([concept]*length)
    # 这个guide_concepts是一个指导概念，指导概念是要将原来的概念编辑为新的概念
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
            new_texts = reduce(operator.concat, new_texts) # 将二维的列表转化为一维的列表
            print_text+=f'-towards'
            for t in new_texts:
                if t not in print_text:
                    print_text+=f'-{t}'
    #   相等      
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
            df = pd.read_csv('/mlx_devbox/users/wangruipeng/playground/paper_run/uce_nullspace/unified-concept-editing-main/data/concat_data/500_20000.csv')

            # 将 'subject' 列赋值给 preserve_concepts 列表
            if 'Artist' in df.columns:
                print('yesysee')
                preserve_concepts = df['Artist'].tolist()
                # print('preserve_concepts:', preserve_concepts)
            else:
                print("Error: 'subject' column not found in the CSV file.")
            # 打印 preserve_concepts 列表以确认
    # 唯一改动,被保留概念的存在
    # data = pd.read_csv('/mlx_devbox/users/wangruipeng/playground/paper_run/uce_nullspace/unified-concept-editing-main/data/extracted_subjects_big_only_500.csv')
    # preserve_concepts += data['subject'].tolist()
    # 唯一改动
    retain_texts = ['']+preserve_concepts
    print("len(retain_texts):)", len(retain_texts))  
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
    print("model_version:", model_version)
    ldm_stable = StableDiffusionPipeline.from_pretrained(model_version).to(device)
    model_path = '/mnt/bn/intern-disk/mlx/users/wrp/uce_nullspace/model_save/art_erase/500_preserve_1000_erase/10.0-500-2.pt'
    ldm_stable.unet.load_state_dict(torch.load(model_path, map_location=device))
    print('load done')
    print_text += f"-sd_{args.base.replace('.','_')}" 
    print_text += f"-method_{technique}" 
    print_text = print_text.lower()
    print(print_text)
    # 模型ldm_stable
    # old_texts
    # new_texts
    # get_similar_token(ldm_stable, path='/mlx_devbox/users/wangruipeng/playground/paper_run/uce_nullspace/unified-concept-editing-main/data/nude_NOUN_VERB_ADJ.csv', layers_to_edit=None, lamb=0.1, with_to_k=True)
    # ldm_stable = edit_model(ldm_stable= ldm_stable, old_text_= old_texts, new_text_=new_texts, add=False, retain_text_= retain_texts, lamb=0.1, erase_scale = erase_scale, preserve_scale = preserve_scale,  technique=technique)
    # torch.save(ldm_stable.unet.state_dict(), f'/mnt/bn/intern-disk/mlx/users/wrp/uce_nullspace/model_save/art_erase/10_erase-artists_output/uce_500_lamb_0.1.pt')
    # with open(f'/mnt/bn/intern-disk/mlx/users/wrp/uce_nullspace/model_save/art_erase/10_erase-artists_output/uce_500_lamb_0.1.txt', 'w') as fp:
    #     json.dump(concepts,fp)
    # model_path = f'/mnt/bn/intern-disk/mlx/users/wrp/uce_nullspace/model_save/art_erase/500_preserve_1000_erase/lamda_200_uce_1000_batch_8.pt'
    # ldm_stable.unet.load_state_dict(torch.load(model_path, map_location=device))
    with open('/mlx/mlx_devbox/users/wangruipeng/playground/paper_run/uce_nullspace/unified-concept-editing-main/train-scripts/uce.txt', 'a') as time_log:
        batch_size = 100   
        for i in tqdm(range(0, len(old_texts), batch_size)): # 方便修改！！！！！！！！！！！！！！！！！
            start_time = time.time()
            old_text = old_texts[i:i + batch_size]
            new_text = new_texts[i:i + batch_size]
            # 使用批次索引更新模型保存路径 
            batch_index = i // batch_size  # 计算批次索引
            model_save_path_with_index = args.model_save_path.replace('.pt', f'_batch_{batch_index}.pt')  # 添加lamb到文件名
            concepts_save_path_with_index = args.concepts_save_path.replace('.txt', f'_batch_{batch_index}.txt')  # 添加lamb到文件名
            ldm_stable = edit_model(ldm_stable= ldm_stable, old_text_= old_texts, new_text_=new_texts, add=False, retain_text_= retain_texts, lamb=100, erase_scale = erase_scale, preserve_scale = preserve_scale,  technique=technique)
            torch.save(ldm_stable.unet.state_dict(), model_save_path_with_index)
            # 保存概念，文件名包含批次索引
            with open(concepts_save_path_with_index, 'w') as fp:
                json.dump(concepts, fp)  
            retain_texts += old_text 
            end_time = time.time()
            elapsed_time = end_time - start_time
            time_log.write(f"Batch {batch_index} processed in {elapsed_time:.2f} seconds.\n")
            time_log.flush() 