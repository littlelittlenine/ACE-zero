# 去除多个职业的race方面的偏见
# indian, asian, african, european, latino + doctor
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
from CLIP_classify import CLIP_classification,CLIP_classification_race
from generate_images import generate_images
import sys
# 比如我要去偏见的职业是doctor这个东西，那么我现在的输入是
# 一个是加上上下文知识的编辑
# case_number,prompt,evaluation_seed,profession
# 0,"gender balance, Image of Doctor",2495,Doctor
# 1,"gender balance, Picture of Doctor",171,Doctor
# 2,"gender balance, Photo of Doctor",2177,Doctor
# 3,"gender balance, Headshot of Doctor",911,Doctor
# 4,"gender balance, Portrait of Doctor",1902,Doctor
# 5,"gender balance, Doctor",42,Doctor
# 一个是
# case_number,prompt,evaluation_seed,profession
# 0,"gender balance, Image of Doctor",2495,Doctor
# 1,"gender balance, Picture of Doctor",171,Doctor
# 2,"gender balance, Photo of Doctor",2177,Doctor
# 3,"gender balance, Headshot of Doctor",911,Doctor
# 4,"gender balance, Portrait of Doctor",1902,Doctor
# 5,"gender balance, Doctor",42,Doctor
# 所以说我现在是编辑v*,加上doctor，和后面的<EOS>都对齐，Image of Doctor和gender balance, Image of Doctor对齐，那么可以去掉偏见
def orthogonal_projection_matrix_torch(A, device): # (320,100) @ (100,320)
    # 确保A是一个PyTorch张量
    A = torch.tensor(A, dtype=torch.float32, device=device)
    
    # 计算矩阵A的转置
    AT = A.t()
    
    # 计算矩阵AAT的逆
    AAT_inv = torch.inverse(A @ AT)
    
    # 在GPU上创建单位矩阵
    I = torch.eye(A.shape[1], device=device)
    
    # 计算正交投影矩阵
    P = I - AT @ AAT_inv @ A
    return P
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
        results_df.to_csv('/mlx_devbox/users/wangruipeng/playground/paper_run/uce_nullspace/unified-concept-editing-main/data/nudity_similar_texts_100.csv', index=False)
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
def edit_model(ldm_stable, old_text_, new_text_, retain_text_, add=False, layers_to_edit=None, lamb=0.1, erase_scale = 0.1, preserve_scale = 0.1, with_to_k=True, technique='tensor', projection_path='/mnt/bn/intern-disk/mlx/users/wrp/uce_nullspace/model_save/nullspace_project/output_2000_100/projection_matrix_{}.pt'):
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
                # 寻找职业比如doctor对应的id所处的位置
                # 需要找到old_emb和new_emb中doctor的编码从什么位置开始
                text_embeddings = ldm_stable.text_encoder(text_input.input_ids.to(ldm_stable.device))[0]
                print("start to find doctor")
                # (2, 77, 768)  text_input.attention_mask[0]是一个二进制掩码，其中1表示有效token，0表示padding token
                # -2去除CLS 和 EOS token
                # old_emb = old_emb[final_token_idx:len(old_emb)-max(0,farthest-final_token_idx)] # （76，768）
                idx_old = text_input.input_ids[0].tolist().index(5547)
                idx_new = text_input.input_ids[1].tolist().index(5547)
                old_emb = text_embeddings[0] # (77, 768)
                # old_emb = old_emb[final_token_idx:len(old_emb)-max(0,farthest-final_token_idx)] # （76，768）
                old_emb = old_emb[idx_old:] # （76，768）
                new_emb = text_embeddings[1] # (77, 768)
                # new_emb = new_emb[final_token_idx_new:len(new_emb)-max(0,farthest-final_token_idx_new)] # （76，768）
                new_emb = new_emb[idx_new:] # （73，768）
                old_emb = old_emb[:len(new_emb)] # (73, 768)       
                context = old_emb.detach()
                values = []
                with torch.no_grad():
                    for layer_number, layer in enumerate(projection_matrices):
                        projection_file_path = projection_path.format((layer_number + 16) % 32)
                        projection = torch.load(projection_file_path)
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
                            values.append(layer(new_emb).detach())
                            # print("values",values[-1].size())
                        else:
                            values.append(layer(new_emb).detach())
                #  先乘法再求和和先求和再乘的区别
                context_vector = context.reshape(context.shape[0], context.shape[1], 1) # (2, 768, 1)
                context_vector_T = context.reshape(context.shape[0], 1, context.shape[1]) # (2, 1, 768)
                value_vector = values[layer_num].reshape(values[layer_num].shape[0], values[layer_num].shape[1], 1) # (2, 320, 1)
                for_mat1 = (value_vector @ context_vector_T).sum(dim=0) # （320，768）进行了求和操作，为什么
                for_mat2 = (context_vector @ context_vector_T).sum(dim=0) # （768，768）
                print("for_mat1",for_mat1.size())
                print("mat1",mat1.size())
                mat1 += erase_scale*for_mat1
                mat2 += erase_scale*for_mat2
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
                print("for_mat1",for_mat1.size())
                print("mat1",mat1.size())
                mat1 += preserve_scale*for_mat1
                mat2 += preserve_scale*for_mat2
                #update projection matrix
            projection_matrices[layer_num].weight = torch.nn.Parameter(mat1 @ torch.inverse(mat2))

    print(f'Current model status: Edited "{str(old_text_)}" into "{str(new_texts)}" and Retained "{str(retain_text_)}"')
    return ldm_stable
def alpha_edit(ldm_stable, old_text_, new_text_, retain_text_, add=False, layers_to_edit=None, lamb=0.1, erase_scale=0.1, preserve_scale=0.1, with_to_k=True, technique='tensor', P=None, lamda=10):
    ### 收集所有交叉注意力模块
    sub_nets = ldm_stable.unet.named_children()
    ca_layers = []
    
    # 遍历所有子网络，收集交叉注意力层
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

    # 获取交叉注意力层的投影矩阵
    projection_matrices = [l.to_v for l in ca_layers]
    og_matrices = [copy.deepcopy(l.to_v) for l in ca_layers]
    if with_to_k:
        projection_matrices = projection_matrices + [l.to_k for l in ca_layers]
        og_matrices = og_matrices + [copy.deepcopy(l.to_k) for l in ca_layers]

    # 重置交叉注意力层的权重矩阵
    num_ca_clip_layers = len(ca_layers)
    for idx_, l in enumerate(ca_layers):
        l.to_v = copy.deepcopy(og_matrices[idx_])
        projection_matrices[idx_] = l.to_v
        if with_to_k:
            l.to_k = copy.deepcopy(og_matrices[num_ca_clip_layers + idx_])
            projection_matrices[num_ca_clip_layers + idx_] = l.to_k
    # 检查要编辑的层
    layers_to_edit = ast.literal_eval(layers_to_edit) if type(layers_to_edit) == str else layers_to_edit
    lamb = ast.literal_eval(lamb) if type(lamb) == str else lamb
    print("layers_to_edit", layers_to_edit)      
    ### 格式化编辑内容
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
    ######################## 开始编辑 ###################################
    print(len(projection_matrices))
    # 这个是Wv矩阵和Wk矩阵的分界线
    k_v_limit =len(projection_matrices)/2
    for layer_num in range(len(projection_matrices)):
        # 我们要寻找的是和Wv矩阵对应的Wk矩阵,以及和Wk矩阵对应的Wv矩阵
        # projection = None
        # opposite_layer_num = int((layer_num + k_v_limit) % len(projection_matrices))
        # print("path:/mnt/bn/intern-disk/mlx/users/wrp/uce_nullspace/model_save/nullspace_project/output_5000/projection_matrix_{}.pt".format(layer_num))
        # # 对应的output空间的投影矩阵
        # projection = torch.load("/mnt/bn/intern-disk/mlx/users/wrp/uce_nullspace/model_save/nullspace_project/output_5000/projection_matrix_{}.pt".format(opposite_layer_num))
        print("layers_to_edit", layers_to_edit) 
        if (layers_to_edit is not None) and (layer_num not in layers_to_edit):
            continue
        print(f'Editing layer {layer_num}')
        with torch.no_grad():  # 计算图控制
            # 获取当前投影矩阵的权重
            W_old = projection_matrices[layer_num].weight.detach()  # 使用 .detach() 防止反向传播
            # 计算目标概念的嵌入
            values = []
            old_embeddings = []
            new_embeddings = []
            for_mat2 = torch.zeros(768,768, device=W_old.device)
            for cnt, (old_text, new_text) in enumerate(zip(old_texts, new_texts)):
                text_input = ldm_stable.tokenizer(
                    [old_text, new_text],
                    padding="max_length",
                    max_length=ldm_stable.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                
                # 获取文本的嵌入
                text_embeddings = ldm_stable.text_encoder(text_input.input_ids.to(ldm_stable.device))[0]
                # print("text_embeddings",text_embeddings)
                # 计算有效 token 的索引
                idx_old = text_input.input_ids[0].tolist().index(25148)
                idx_new = text_input.input_ids[1].tolist().index(25148)
                print([idx_old, idx_new])
                old_emb = text_embeddings[0] # (77, 768)
                new_emb = text_embeddings[1] # (77, 768)
                if cnt==0:
                    old_emb = old_emb[idx_old:idx_old + 2] # （4，768）
                    new_emb = new_emb[idx_new:idx_new + 2] # （4，768）
                else:
                # old_emb = old_emb[final_token_idx:len(old_emb)-max(0,farthest-final_token_idx)] # （76，768）
                    old_emb = old_emb[idx_old-2:idx_old + 2] # （4，768）
                    # new_emb = new_emb[final_token_idx_new:len(new_emb)-max(0,farthest-final_token_idx_new)] # （76，768）
                    new_emb = new_emb[idx_new-2:idx_new + 2] # （4，768）            
                context = old_emb.detach()
                # # 计算均值,这里应该是不需要计算均值的,直接展开操作
                # old_emb_flat = old_emb.view(text_embeddings.size(0), -1)
                # new_emb_flat = new_emb.view(text_embeddings.size(0), -1)
                # old_meb和new_emb的维度都是(768)
                with torch.no_grad():
                    for layer_number, layer in enumerate(projection_matrices):
                        o_embs = layer(old_emb).detach()
                        if technique == 'tensor':
                            o_embs = layer(old_emb).detach()
                            u = o_embs
                            u = u / u.norm()
                            new_embs = layer(new_emb).detach()
                            new_emb_proj = (u*new_embs).sum()
                            target = new_embs - (new_emb_proj)*u
                            values.append(target.detach()) 
                        elif technique == 'replace': # 默认值是replace
                            values.append(layer(new_emb).detach())
                            # print("values",values[-1].size())
                        else:
                            values.append(layer(new_emb).detach())
                # 按照公式：RK^TP(KK^TP + I)，这里面：for_mat2是KK^T;
                # 
                context_vector = context.reshape(context.shape[0], context.shape[1], 1) # (2, 768, 1)
                context_vector_T = context.reshape(context.shape[0], 1, context.shape[1]) # (2, 1, 768)
                value_vector = values[layer_num].reshape(values[layer_num].shape[0], values[layer_num].shape[1], 1) # (2, 320, 1)              
                for_mat1 = (value_vector @ context_vector_T).sum(dim=0)
                for_mat2 += (context_vector @ context_vector_T).sum(dim=0) # (768,768)累加上去
                # 这里感觉有点问题
                # old_emb = old_emb.sum(dim=0) # （768）
                # new_emb = new_emb.sum(dim=0) # （768）

                # old_embeddings.append(old_emb) 
                # # 通过当前层的投影矩阵计算嵌入
                # o_embs = projection_matrices[layer_num](old_emb).detach()
                # new_embs = projection_matrices[layer_num](new_emb).detach()
                # # new_emb_proj = (u * new_embs).sum()
                # new_embeddings.append(new_embs)
                # 计算目标嵌入
                # target = new_embs - (new_emb_proj) * u 
                values.append(new_embs.detach()) 
            # 直接去掉三个概念
            old_embs = torch.stack(old_embeddings) # (3, 768)
            new_embs = torch.stack(new_embeddings) # (3, 768)
            # 如果维度是1的话只有一个概念，也就是(768)
            if old_embs.dim() == 1:
                old_embs = old_embs.unsqueeze(0)
                new_embs = new_embs.unsqueeze(0)
            # print("old_embs", old_embs.size())
            # print("new_embs", new_embs.size())
            # 当前层的投影矩阵和协方差矩阵
            Wp = projection_matrices[layer_num]    # (768, 320)
            # P = torch.ones_like(new_emb.unsqueeze(0).T @ new_emb.unsqueeze(0)) # (768,768)
            P = torch.load("/mnt/bn/intern-disk/mlx/users/wrp/uce_nullspace/model_save/nullspace_project/input_5000/null_space_project_subject_5000.pt")
            # print('P.size',P.size())
            # print('old_emb.unsqueeze(0) @ old_emb.unsqueeze(0).T',(old_emb.unsqueeze(0) @ old_emb.unsqueeze(0).T).size())
            Kp1 = P  
            Kp2 = P
            # print('Kp2.size',Kp2.size())
            # 计算 R, (3,768)
            R = torch.stack(values) @ projection - old_embs @ W_old.T # 计算目标概念与旧权重的差
            # print("W_old.size", W_old.size())
            # print('old_emb.size', old_emb.size())
            # print("new_emb.size", new_emb.size())
            # print('old_embs.size', o_embs.size())
            # print("new_embs.size", new_embs.size())
            # print("R.size()",R.size()) # 320
            # 使用线性代数求解更新矩阵,矩阵的输入维度是768，输出维度是320,(768,768)*()
            
            result1 =  Kp1 @ (old_embs.T @ old_embs) + lamb * torch.eye((old_embs.T).shape[0], dtype=torch.float, device="cuda")
            print("Kp2.size", Kp2.size())
            # (768,768) @ (768,3) @ (3,320)
            result2 =  Kp2 @ (old_embs.T) @ R
            print("result1", result1)
            print("result2", result2)
            upd_matrix = torch.linalg.solve(
                result1, 
                result2
            )
            # 权重：（320，320）*（320，1）*（1，320）
            # 更新投影矩阵权重
            # file_path = f"/mlx_devbox/users/wangruipeng/playground/paper_run/uce_nullspace/unified-concept-editing-main/models/updata_matrics_{layer_num}.pt"
            print("upd_matrix.size",upd_matrix.size())
            projection_matrices[layer_num].weight = torch.nn.Parameter(W_old + upd_matrix.T)
    P = get_project_input_3(ldm_stable,'/mlx_devbox/users/wangruipeng/playground/paper_run/uce_nullspace/unified-concept-editing-main/data/extracted_subjects_big_only_5000.csv')
    # # print("P.size",P.size())
    torch.save(P, "/mnt/bn/intern-disk/mlx/users/wrp/uce_nullspace/model_save/nullspace_project/input_5000/null_space_project_subject_5000_3_200.pt") 
    print(f'当前模型状态: 将"{str(old_text_)}"编辑为"{str(new_texts)}"，并保留"{str(ret_texts)}"')
    return ldm_stable
def alpha_edit_2(ldm_stable, old_text_, new_text_, retain_text_, add=False, layers_to_edit=None, lamb=0.1, erase_scale=0.1, preserve_scale=0.1, with_to_k=True, technique='tensor', P=None, lamda=10, cache_c = None, P_outs = None):
    ### 收集所有交叉注意力模块
    sub_nets = ldm_stable.unet.named_children()
    ca_layers = []
    
    # 遍历所有子网络，收集交叉注意力层
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

    # 获取交叉注意力层的投影矩阵
    projection_matrices = [l.to_v for l in ca_layers]
    og_matrices = [copy.deepcopy(l.to_v) for l in ca_layers]
    if with_to_k:
        projection_matrices = projection_matrices + [l.to_k for l in ca_layers]
        og_matrices = og_matrices + [copy.deepcopy(l.to_k) for l in ca_layers]

    # 重置交叉注意力层的权重矩阵
    num_ca_clip_layers = len(ca_layers)
    for idx_, l in enumerate(ca_layers):
        l.to_v = copy.deepcopy(og_matrices[idx_])
        projection_matrices[idx_] = l.to_v
        if with_to_k:
            l.to_k = copy.deepcopy(og_matrices[num_ca_clip_layers + idx_])
            projection_matrices[num_ca_clip_layers + idx_] = l.to_k
    # 检查要编辑的层
    layers_to_edit = ast.literal_eval(layers_to_edit) if type(layers_to_edit) == str else layers_to_edit
    lamb = ast.literal_eval(lamb) if type(lamb) == str else lamb
    print("layers_to_edit", layers_to_edit)      
    ### 格式化编辑内容
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
    ######################## 开始编辑 ###################################
    print(len(projection_matrices))
    # 这个是Wv矩阵和Wk矩阵的分界线
    # P = torch.load("/mnt/bn/intern-disk/mlx/users/wrp/uce_nullspace/model_save/nullspace_project/input_1000/null_space_project_subject_1000_3_300.pt")
    P1 = P.clone()
    P2 = P.clone()
    for layer_num in range(len(projection_matrices)):
        # 我们要寻找的是和Wv矩阵对应的Wk矩阵,以及和Wk矩阵对应的Wv矩阵
        print("layers_to_edit", layers_to_edit) 
        if (layers_to_edit is not None) and (layer_num not in layers_to_edit):
            continue
        print(f'Editing layer {layer_num}')
        with torch.no_grad():  # 计算图控制
            # 获取当前投影矩阵的权重
            W_old = projection_matrices[layer_num].weight.detach()  # 使用 .detach() 防止反向传播
            # P_out = P_outs[layer_num]
            # 计算目标概念的嵌入
            values = []
            old_embeddings = []
            new_embeddings = []
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
                
                # 获取文本的嵌入
                text_embeddings = ldm_stable.text_encoder(text_input.input_ids.to(ldm_stable.device))[0]
                # 计算有效 token 的索引
                idx_old = text_input.input_ids[0].tolist().index(49407)
                idx_new = text_input.input_ids[1].tolist().index(49407)
                print([idx_old, idx_new])
                old_emb = text_embeddings[0] # (77, 768)
                new_emb = text_embeddings[1] # (77, 768)
                # if cnt%6 == 0:
                # old_emb = old_emb[idx_old-1:idx_old + 1] # （4，768）
                # new_emb = new_emb[idx_new-1:idx_new + 1] # （4，768）
                old_emb = old_emb[idx_old:idx_old+1] # （4，768）
                new_emb = new_emb[idx_new:idx_new+1] # （4，768）
                # else:
                # # old_emb = old_emb[final_token_idx:len(old_emb)-max(0,farthest-final_token_idx)] # （76，768）
                #     old_emb = old_emb[idx_old-3:idx_old + 1] # （4，768）
                #     # new_emb = new_emb[final_token_idx_new:len(new_emb)-max(0,farthest-final_token_idx_new)] # （76，768）
                #     new_emb = new_emb[idx_new-3:idx_new + 1] # （4，768）  
                # # 计算均值,这里应该是不需要计算均值的,直接展开操作
                # old_emb_flat = old_emb.view(text_embeddings.size(0), -1)
                # new_emb_flat = new_emb.view(text_embeddings.size(0), -1)
                # old_meb和new_emb的维度都是(768)
                context = old_emb.detach()
                context_vector = context.reshape(context.shape[0], context.shape[1], 1) # (1, 768, 1)
                context_vector_T = context.reshape(context.shape[0], 1, context.shape[1]) # (1, 1, 768)
                value_vector = (new_emb @ W_old.T).detach() # (2, 320, 1)
                # value_vector = value_vector @ P_out.T
                value_vector = value_vector.reshape(value_vector.shape[0], value_vector.shape[1],1) # (2, 320)           
                for_mat2 += (context_vector @ context_vector_T).sum(dim=0) # (768,768)累加上去
                o_embs = context @ W_old.T
                # print('o_embs:',o_embs.size())
                # print('value_vector:',value_vector.size())
                R = value_vector - o_embs.unsqueeze(-1) # (X,320,1) @ (X,1,768)
                # print('R:',R.size())alpha_editget
                # print("o_embs:",o_embs.size())
                # print('context_vector_T',context_vector_T.size())
                for_mat3 += (R @ context_vector_T).sum(dim=0) # (768,768)累加上去 # (X,320,1) @ (X,1,768)
            # 也需要对整体
            print("P1.device:",P1.device)
            print('for_mat2.device:',for_mat1.device)
            print('for_mat3.device:',for_mat2.device)
            result1 = lamb * for_mat2 @ P1 + lamda * for_mat1
            result2 = lamb * for_mat3 @ P2
            # (320,768) @ (768,768) = (320,768)
            upd_matrix = torch.linalg.solve(
                result1.transpose(0, 1), # 这个的转置没有什么用处
                result2.transpose(0, 1)
            )
            # 权重：（320，320）*（320，1）*（1，320）
            # 更新投影矩阵权重
            # file_path = f"/mlx_devbox/users/wangruipeng/playground/paper_run/uce_nullspace/unified-concept-editing-main/models/updata_matrics_{layer_num}.pt"
            # print("upd_matrix.size",upd_matrix.size())
            projection_matrices[layer_num].weight = torch.nn.Parameter(W_old + upd_matrix.T)
    cache_c += for_mat2
    print(f'当前模型状态: 将"{str(old_text_)}"编辑为"{str(new_texts)}"，并保留"{str(ret_texts)}"')
    return ldm_stable, cache_c

def alpha_edit_3(ldm_stable, old_text_, new_text_, retain_text_, add=False, layers_to_edit=None, lamb=0.1, erase_scale=0.1, preserve_scale=0.1, with_to_k=True, technique='tensor', P=None, lamda=10, cache_c = None, P_outs = None):
    ### 收集所有交叉注意力模块
    sub_nets = ldm_stable.unet.named_children()
    ca_layers = []
    
    # 遍历所有子网络，收集交叉注意力层
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

    # 获取交叉注意力层的投影矩阵
    projection_matrices = [l.to_v for l in ca_layers]
    og_matrices = [copy.deepcopy(l.to_v) for l in ca_layers]
    if with_to_k:
        projection_matrices = projection_matrices + [l.to_k for l in ca_layers]
        og_matrices = og_matrices + [copy.deepcopy(l.to_k) for l in ca_layers]

    # 重置交叉注意力层的权重矩阵
    num_ca_clip_layers = len(ca_layers)
    for idx_, l in enumerate(ca_layers):
        l.to_v = copy.deepcopy(og_matrices[idx_])
        projection_matrices[idx_] = l.to_v
        if with_to_k:
            l.to_k = copy.deepcopy(og_matrices[num_ca_clip_layers + idx_])
            projection_matrices[num_ca_clip_layers + idx_] = l.to_k
    # 检查要编辑的层
    layers_to_edit = ast.literal_eval(layers_to_edit) if type(layers_to_edit) == str else layers_to_edit
    lamb = ast.literal_eval(lamb) if type(lamb) == str else lamb
    print("layers_to_edit", layers_to_edit)      
    ### 格式化编辑内容
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
    ######################## 开始编辑 ###################################
    print(len(projection_matrices))
    # 这个是Wv矩阵和Wk矩阵的分界线
    # P = torch.load("/mnt/bn/intern-disk/mlx/users/wrp/uce_nullspace/model_save/nullspace_project/input_1000/null_space_project_subject_1000_3_300.pt")
    P1 = P.clone()
    P2 = P.clone()
    for layer_num in range(len(projection_matrices)):
        # 我们要寻找的是和Wv矩阵对应的Wk矩阵,以及和Wk矩阵对应的Wv矩阵
        print("layers_to_edit", layers_to_edit) 
        if (layers_to_edit is not None) and (layer_num not in layers_to_edit):
            continue
        print(f'Editing layer {layer_num}')
        with torch.no_grad():  # 计算图控制
            # 获取当前投影矩阵的权重
            W_old = projection_matrices[layer_num].weight.detach()  # 使用 .detach() 防止反向传播
            # P_out = P_outs[layer_num]
            # 计算目标概念的嵌入
            values = []
            old_embeddings = []
            new_embeddings = []
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
                # 获取文本的嵌入
                text_embeddings = ldm_stable.text_encoder(text_input.input_ids.to(ldm_stable.device))[0]
                # 计算有效 token 的索引
                idx_old = text_input.input_ids[0].tolist().index(49407)
                idx_new = text_input.input_ids[1].tolist().index(49407)
                print([idx_old, idx_new])
                old_emb = text_embeddings[0] # (77, 768)
                new_emb = text_embeddings[1] # (77, 768)
                # if cnt%6 == 0:
                # old_emb = old_emb[idx_old-1:idx_old + 1] # （4，768）
                # new_emb = new_emb[idx_new-1:idx_new + 1] # （4，768）
                old_emb = old_emb[idx_old:idx_old+1] # （4，768）
                new_emb = new_emb[idx_new:idx_new+1] # （4，768）
                # else:
                # # old_emb = old_emb[final_token_idx:len(old_emb)-max(0,farthest-final_token_idx)] # （76，768）
                #     old_emb = old_emb[idx_old-3:idx_old + 1] # （4，768）
                #     # new_emb = new_emb[final_token_idx_new:len(new_emb)-max(0,farthest-final_token_idx_new)] # （76，768）
                #     new_emb = new_emb[idx_new-3:idx_new + 1] # （4，768）  
                # # 计算均值,这里应该是不需要计算均值的,直接展开操作
                # old_emb_flat = old_emb.view(text_embeddings.size(0), -1)
                # new_emb_flat = new_emb.view(text_embeddings.size(0), -1)
                # old_meb和new_emb的维度都是(768)
                context = old_emb.detach()
                context_vector = context.reshape(context.shape[0], context.shape[1], 1) # (1, 768, 1)
                context_vector_T = context.reshape(context.shape[0], 1, context.shape[1]) # (1, 1, 768)
                value_vector = (new_emb @ W_old.T).detach() # (2, 320, 1)
                # value_vector = value_vector @ P_out.T
                value_vector = value_vector.reshape(value_vector.shape[0], value_vector.shape[1],1) # (2, 320)           
                for_mat2 += (context_vector @ context_vector_T).sum(dim=0) # (768,768)累加上去
                o_embs = context @ W_old.T
                # print('o_embs:',o_embs.size())
                # print('value_vector:',value_vector.size())
                R = value_vector - o_embs.unsqueeze(-1) # (X,320,1) @ (X,1,768)
                # print('R:',R.size())alpha_editget
                # print("o_embs:",o_embs.size())
                # print('context_vector_T',context_vector_T.size())
                for_mat3 += (R @ context_vector_T).sum(dim=0) # (768,768)累加上去 # (X,320,1) @ (X,1,768)
            # 也需要对整体
            print("P1.device:",P1.device)
            print('for_mat2.device:',for_mat1.device)
            print('for_mat3.device:',for_mat2.device)
            result1 = lamb * P_outs[layer_num] @ for_mat2 @ P1 + lamda * for_mat1
            result2 = lamb * P_outs[layer_num] @ for_mat3 @ P2
            # (320,768) @ (768,768) = (320,768)
            upd_matrix = torch.linalg.solve(
                result1.transpose(0, 1), # 这个的转置没有什么用处
                result2.transpose(0, 1)
            )
            # 权重：（320，320）*（320，1）*（1，320）
            # 更新投影矩阵权重
            # file_path = f"/mlx_devbox/users/wangruipeng/playground/paper_run/uce_nullspace/unified-concept-editing-main/models/updata_matrics_{layer_num}.pt"
            # print("upd_matrix.size",upd_matrix.size())
            projection_matrices[layer_num].weight = torch.nn.Parameter(W_old + upd_matrix.T)
    cache_c += for_mat2
    print(f'当前模型状态: 将"{str(old_text_)}"编辑为"{str(new_texts)}"，并保留"{str(ret_texts)}"')
    return ldm_stable, cache_c
# 获取需要的
def get_project_input_3(ldm_stable, data_path, subject_column='subject', num_smallest_singular=300, batch_size=16):
    # 读取数据
    data = pd.read_csv(data_path)
    # 初始化一个张量来存储嵌入
    data = data[data[subject_column].apply(lambda x: isinstance(x, str))]
    total_embeddings = None
    print(len(data))
    for i in tqdm(range(0, len(data[subject_column]), batch_size)):  # 使用 tqdm 监控进度
        batch_prompts = data[subject_column][i:i + batch_size]
        # 清理每个 prompt，去掉不必要的引号和多余的空格
        cleaned_prompts = [prompt.replace('“', '').replace('”', '').strip() for prompt in batch_prompts]
        # 分词并编码为 token id
        text_input = ldm_stable.tokenizer(
            cleaned_prompts,
            padding="max_length",
            max_length=ldm_stable.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        # 获取文本的嵌入
        with torch.no_grad():  # 不计算梯度以节省显存
            idx = text_input.input_ids[0].tolist().index(49407) 
            text_embeddings = ldm_stable.text_encoder(text_input.input_ids.to(ldm_stable.device))[0]
            text_embeddings = text_embeddings.detach()  # 从计算图中剥离张量 # (1,77,768)
            text_embeddings = text_embeddings[:,1:idx+1:,:]
        # 将形状转换为 (16*76, 768)
        # print("text_embeddings:", text_embeddings) # (16, 7, 768)
        # print("text_embeddings,size:", text_embeddings.size()) # (16, 7, 768)
        text_embeddings = text_embeddings.reshape(-1, text_embeddings.size(-1))
        # print("text_embeddings,size:", text_embeddings.size()) # (112, 768)
        # 初始化总嵌入张量或连接新的嵌入
        if total_embeddings is None:
            total_embeddings = text_embeddings
        else:
            total_embeddings = torch.cat((total_embeddings, text_embeddings), dim=0)
        # 释放内存
        del text_input, text_embeddings
        torch.cuda.empty_cache()  # 清理未使用的缓存
    # 现在total_embeddings的形状是(76*5000, 768)
    # print("Total embeddings size:", total_embeddings.size())
    
    # 计算(768, 76*5000) @ (76*5000, 768)
    product = total_embeddings.T @ total_embeddings
    # print("total_embeddings:",product)
    # print("total_embeddings.size:",product.size())
    # 进行 SVD 分解
    U, S, _ = torch.linalg.svd(product, full_matrices=False)
    # print("Singular values size:", S.size())
    
    # 打印一下最小的50个奇异值看一下数据分布
    # print(f"Smallest 50 singular values: {S[-50:]}")
    
    # 选择最小的 N 个奇异值的索引
    smallest_indices = torch.topk(S, num_smallest_singular, largest=False).indices
    smallest_indices = smallest_indices.sort().values
    # print(f"Indices of the smallest {num_smallest_singular} singular values: {smallest_indices}")
    
    # 计算投影矩阵
    projection_matrix = U[:, smallest_indices] @ U[:, smallest_indices].T
    # print("Projection matrix size:", projection_matrix.size())
    
    return projection_matrix
# 获得输出端的投影矩阵
# def get_project_output(ldm_stable, data_path, percentage_of_smallest_singular=0.01, batch_size=16, with_to_k=True):
#     ### 收集所有交叉注意力模块 ###
#     sub_nets = ldm_stable.unet.named_children()
#     ca_layers = []
    
#     # 遍历所有子网络，收集交叉注意力层
#     for net in sub_nets:
#         if 'up' in net[0] or 'down' in net[0]:
#             for block in net[1]:
#                 if 'Cross' in block.__class__.__name__:
#                     for attn in block.attentions:
#                         for transformer in attn.transformer_blocks:
#                             ca_layers.append(transformer.attn2)
#         if 'mid' in net[0]:
#             for attn in net[1].attentions:
#                 for transformer in attn.transformer_blocks:
#                     ca_layers.append(transformer.attn2)

#     # 获取交叉注意力层的投影矩阵
#     projection_matrices = [l.to_v for l in ca_layers]
#     print(f"Number of projection matrices, Wk number: {len(projection_matrices)}")
#     og_matrices = [copy.deepcopy(l.to_v) for l in ca_layers]
#     if with_to_k:
#         projection_matrices += [l.to_k for l in ca_layers]
#         og_matrices += [copy.deepcopy(l.to_k) for l in ca_layers]

#     # 重置交叉注意力层的权重矩阵
#     num_ca_clip_layers = len(ca_layers)
#     for idx_, l in enumerate(ca_layers):
#         l.to_v = copy.deepcopy(og_matrices[idx_])
#         projection_matrices[idx_] = l.to_v
#         if with_to_k:
#             l.to_k = copy.deepcopy(og_matrices[num_ca_clip_layers + idx_])
#             projection_matrices[num_ca_clip_layers + idx_] = l.to_k

#     ### 读取数据 ###
#     data = pd.read_csv(data_path)
#     print(len(data))
#     P_outs = []
#     print("len(projection_matrices)", len(projection_matrices))
#     for layer_num in range(len(projection_matrices)):
#         # 初始化一个张量来存储嵌入
#         total_embeddings = None
#         for i in tqdm(range(0, len(data['words']), batch_size)):  # 使用 tqdm 监控进度
#             batch_prompts = data['words'][i:i + batch_size]
#             # 清理每个 prompt，去掉不必要的引号和多余的空格
#             cleaned_prompts = [prompt.replace('“', '').replace('”', '').strip() for prompt in batch_prompts]
#             # 分词并编码为 token id
#             text_input = ldm_stable.tokenizer(
#                 cleaned_prompts,
#                 padding="max_length",
#                 max_length=ldm_stable.tokenizer.model_max_length,
#                 truncation=True,
#                 return_tensors="pt",
#             )
#             # 获取文本的嵌入
#             with torch.no_grad():  # 不计算梯度以节省显存
#                 idx = text_input.input_ids[0].tolist().index(49407) 
#                 text_embeddings = ldm_stable.text_encoder(text_input.input_ids.to(ldm_stable.device))[0]
#                 output_embeddings = projection_matrices[layer_num](text_embeddings).detach()

#                 # text_embeddings = text_embeddings.detach()  # 从计算图中剥离张量 # (1,77,768)
#                 output_embeddings = output_embeddings[:,1:idx+1:,:]
#             # 将形状转换为 (16*76, 768)
#             # print("text_embeddings:", text_embeddings) # (16, 7, 768)
#             # print("text_embeddings,size:", text_embeddings.size()) # (16, 7, 768)
#             output_embeddings = output_embeddings.reshape(-1, output_embeddings.size(-1))
#             # print("text_embeddings,size:", text_embeddings.size()) # (112, 768)
#             # 初始化总嵌入张量或连接新的嵌入
#             if total_embeddings is None:
#                 total_embeddings = output_embeddings
#             else:
#                 total_embeddings = torch.cat((total_embeddings, output_embeddings), dim=0)
#             # 释放内存
#             del text_input, text_embeddings, output_embeddings
#             torch.cuda.empty_cache()  # 清理未使用的缓存
#         print("Total embeddings size:", total_embeddings.size())
#         # 计算转置乘积
#         product = torch.mm(total_embeddings.T, total_embeddings)
#         print("Product size:", product.size())
#         # 进行 SVD 分解
#         U, S, _ = torch.linalg.svd(product, full_matrices=False)
#         print("Singular values size:", S.size())

#         # 计算选择的最小奇异值的数量
#         total_singular_values = S.size(0)
#         num_smallest_singular = max(1, int(total_singular_values * percentage_of_smallest_singular))  # 确保至少选择1个

#         # 选择最小的 N 个奇异值的索引
#         smallest_indices = torch.topk(S, num_smallest_singular, largest=False).indices
#         smallest_indices = smallest_indices.sort().values
        
#         # 计算投影矩阵
#         projection_matrix = U[:, smallest_indices] @ U[:, smallest_indices].T
#         print("Projection matrix size:", projection_matrix.size())
#         P_outs.append(projection_matrix)
#     return P_outs
def get_project_output(ldm_stable, preserve_concepts, percentage_of_smallest_singular=0.01, batch_size=16, with_to_k=True):
    ### 收集所有交叉注意力模块 ###
    sub_nets = ldm_stable.unet.named_children()
    ca_layers = []
    
    # 遍历所有子网络，收集交叉注意力层
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

    # 获取交叉注意力层的投影矩阵
    projection_matrices = [l.to_v for l in ca_layers]
    print(f"Number of projection matrices, Wk number: {len(projection_matrices)}")
    og_matrices = [copy.deepcopy(l.to_v) for l in ca_layers]
    if with_to_k:
        projection_matrices += [l.to_k for l in ca_layers]
        og_matrices += [copy.deepcopy(l.to_k) for l in ca_layers]

    # 重置交叉注意力层的权重矩阵
    num_ca_clip_layers = len(ca_layers)
    for idx_, l in enumerate(ca_layers):
        l.to_v = copy.deepcopy(og_matrices[idx_])
        projection_matrices[idx_] = l.to_v
        if with_to_k:
            l.to_k = copy.deepcopy(og_matrices[num_ca_clip_layers + idx_])
            projection_matrices[num_ca_clip_layers + idx_] = l.to_k

    ### 使用 preserve_concepts 生成数据 ###
    print(len(preserve_concepts))
    P_outs = []
    print("len(projection_matrices)", len(projection_matrices))
    
    for layer_num in range(len(projection_matrices)):
        # 初始化一个张量来存储嵌入
        total_embeddings = None
        
        # 使用 tqdm 监控进度
        for i in tqdm(range(0, len(preserve_concepts), batch_size)):
            batch_prompts = preserve_concepts[i:i + batch_size]
            # 清理每个 prompt，去掉不必要的引号和多余的空格
            cleaned_prompts = [prompt.replace('“', '').replace('”', '').strip() for prompt in batch_prompts]
            # 分词并编码为 token id
            text_input = ldm_stable.tokenizer(
                cleaned_prompts,
                padding="max_length",
                max_length=ldm_stable.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            # 获取文本的嵌入
            with torch.no_grad():  # 不计算梯度以节省显存
                idx = text_input.input_ids[0].tolist().index(49407) 
                text_embeddings = ldm_stable.text_encoder(text_input.input_ids.to(ldm_stable.device))[0]
                output_embeddings = projection_matrices[layer_num](text_embeddings).detach()

                # text_embeddings = text_embeddings.detach()  # 从计算图中剥离张量 # (1,77,768)
                output_embeddings = output_embeddings[:, 1:idx, :]
            
            # 将形状转换为 (batch_size * sequence_length, embedding_dimension)
            output_embeddings = output_embeddings.reshape(-1, output_embeddings.size(-1))

            # 初始化总嵌入张量或连接新的嵌入
            if total_embeddings is None:
                total_embeddings = output_embeddings
            else:
                total_embeddings = torch.cat((total_embeddings, output_embeddings), dim=0)

            # 释放内存
            del text_input, text_embeddings, output_embeddings
            torch.cuda.empty_cache()  # 清理未使用的缓存
        
        print("Total embeddings size:", total_embeddings.size())
        
        # 计算转置乘积
        product = torch.mm(total_embeddings.T, total_embeddings)
        print("Product size:", product.size())
        
        # 进行 SVD 分解
        U, S, _ = torch.linalg.svd(product, full_matrices=False)
        print("Singular values size:", S.size())

        # 计算选择的最小奇异值的数量
        total_singular_values = S.size(0)
        num_smallest_singular = max(1, int(total_singular_values * percentage_of_smallest_singular))  # 确保至少选择1个

        # 选择最小的 N 个奇异值的索引
        smallest_indices = torch.topk(S, num_smallest_singular, largest=False).indices
        smallest_indices = smallest_indices.sort().values
        
        # 计算投影矩阵
        projection_matrix = U[:, smallest_indices] @ U[:, smallest_indices].T
        print("Projection matrix size:", projection_matrix.size())
        
        P_outs.append(projection_matrix)
    
    return P_outs
# # 一种是直接把male 和 female相关的都输入，找到投影矩阵，一种是分别对male和female求投影矩阵
def K_means_output(ldm_stable, concept, with_to_k=True, i=0):
    ### 收集所有交叉注意力模块 ###
    sub_nets = ldm_stable.unet.named_children()
    ca_layers = []
    P_outs = []
    # 遍历所有子网络，收集交叉注意力层
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

    # 获取交叉注意力层的投影矩阵
    projection_matrices = [l.to_v for l in ca_layers]
    print(f"Number of projection matrices, Wk number: {len(projection_matrices)}")
    og_matrices = [copy.deepcopy(l.to_v) for l in ca_layers]
    if with_to_k:
        projection_matrices += [l.to_k for l in ca_layers]
        og_matrices += [copy.deepcopy(l.to_k) for l in ca_layers]

    # 重置交叉注意力层的权重矩阵
    num_ca_clip_layers = len(ca_layers)
    for idx_, l in enumerate(ca_layers):
        l.to_v = copy.deepcopy(og_matrices[idx_])
        projection_matrices[idx_] = l.to_v
        if with_to_k:
            l.to_k = copy.deepcopy(og_matrices[num_ca_clip_layers + idx_])
            projection_matrices[num_ca_clip_layers + idx_] = l.to_k
    # scores = []  
    concept = [concept] 
    for layer_num in range(len(projection_matrices)):
        # scores = []
        total_embeddings = None
        # filename = f"/mlx_devbox/users/wangruipeng/playground/paper_run/uce_nullspace/unified-concept-editing-main/data/nudity_similar/top_100_words_layer_{layer_num}.csv"
        # # 保存到CSV文件
        # with open(filename, 'w', newline='', encoding='utf-8') as file:
        #     writer = csv.writer(file)
        #     writer.writerow(['Word'])  # 写入表头
        #     for word in top_100_words:
        #         writer.writerow([word])  # 写入单词
        # 根据top_100_words获得投影P_out，读取file_name中的文本
        # data = pd.read_csv(filename)
        # total_embeddings = None
        # for i in tqdm(range(0, len(data['Word']), 1)):  # 使用 tqdm 监控进度
        # batch_prompts = data['Word'][i:i + 1]
        # 清理每个 prompt，去掉不必要的引号和多余的空格
        cleaned_prompts = [prompt.replace('“', '').replace('”', '').strip() for prompt in concept]
        # 分词并编码为 token id
        text_input = ldm_stable.tokenizer(
            cleaned_prompts,
            padding="max_length",
            max_length=ldm_stable.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        # 获取文本的嵌入
        with torch.no_grad():  # 不计算梯度以节省显存
            idx = text_input.input_ids[0].tolist().index(49407)
            text_embeddings = ldm_stable.text_encoder(text_input.input_ids.to(ldm_stable.device))[0]
            text_embeddings = text_embeddings.detach()  # 从计算图中剥离张量 # (1,77,768)
            output_embeddings = projection_matrices[layer_num](text_embeddings).detach()       
            output_embeddings = output_embeddings[:,idx-1,:]
            # print('output_embeddings.size:',output_embeddings.size())
            if total_embeddings is None:
                total_embeddings = output_embeddings
            else:
                total_embeddings = torch.cat((total_embeddings, output_embeddings), dim=0)
        if i!=0:
            P_out = orthogonal_projection_matrix_torch(total_embeddings, device)
        else:
            P_out = torch.eye(projection_matrices[layer_num].weight.shape[1],dtype=torch.float,device = projection_matrices[layer_num].weight.device)
        P_outs.append(P_out)        
    return P_outs
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
    # parser.add_argument('--projection_path', help='Path to the projection matrix', type=str, required=True)
    parser.add_argument('--num_smallest_singular', help='Number of smallest singular values to consider', type=int, required=False, default=300)
    parser.add_argument('--coco_path', help='coco dataset path', type=str, required=False, default=None)
    parser.add_argument('--lamb', help='lambda value for optimization', type=float, required=False, default=0.1)  # 新增的lamb参数
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
    concepts = args.concepts.split(',') # 得到概念列表[violence, nudity, harm]
    concepts = [con.strip() for con in concepts] # 去除空格
    concept_type = args.concept_type # 得到概念类型
    # projection_path = args.projection_path
    model_save_path = args.model_save_path
    concepts_save_path = args.concepts_save_path
    # P = torch.load(args.projection_path)
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
        df = pd.read_csv('data/artists1734_prompts.csv')
        artists = list(df.artist.unique())
        number = int(concepts[0].replace('artists', ''))
        concepts = random.sample(artists,number)
    if 'professions5' in concepts[0]: # 去偏见200个职业professions2000
        # df = pd.read_csv('/mlx_devbox/users/wangruipeng/playground/paper_run/uce_nullspace/unified-concept-editing-main/data/profession_prompts.csv')
        # professions = list(df.profession.unique())
        # number = int(concepts[0].replace('professions', ''))
        # concepts = random.sample(professions,number)
        output_concepts = ['male','female', 'man', 'woman', 'boy', 'girl']
        concepts = ['doctor']
        # preserve_concepts = []
        # 嵌套循环生成组合
        # for concept in add_concept:
        #     preserve_concepts.append(concept)  # 添加单独的概念
        #     for sub_concept in concepts:
        #         combined_concept = f"{concept} {sub_concept}"  # 组合概念
        #         preserve_concepts.append(combined_concept)  # 添加组合后的概念

        # erase_profession_path = '/mlx_devbox/users/wangruipeng/playground/paper_run/uce_nullspace/unified-concept-editing-main/data/bias/debias_professions_200.csv'
        # erase_profession_df = pd.DataFrame(concepts, columns=["Artist"])
        # erase_profession_df.to_csv(erase_profession_path, index=False)  # erase_artists_df
    elif 'professions' in concepts[0] and 'professions5' not in concepts[0]:
        df = pd.read_csv('/mlx_devbox/users/wangruipeng/playground/paper_run/uce_nullspace/unified-concept-editing-main/data/profession_prompts.csv')
        professions = list(df.profession.unique())
        number = int(concepts[0].replace('professions', ''))
        concepts = random.sample(professions,number)
        # erase_profession_path = '/mlx_devbox/users/wangruipeng/playground/paper_run/uce_nullspace/unified-concept-editing-main/data/bias/debias_professions_200.csv'
        # erase_profession_df = pd.DataFrame(concepts, columns=["Artist"])
        # erase_profession_df.to_csv(erase_profession_path, index=False)  # erase_artists_df        
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
    # 输入的是bias_profession，这个是为了去除profession中的bias
    elif concept_type=='bias_profession':  
        additional_prompts.append('Image of {concept}')
        additional_prompts.append('Picture of {concept}')
        additional_prompts.append('Photo of {concept}')
        additional_prompts.append('Headshot of {concept}')
        additional_prompts.append('Portrait of {concept}')
    # 需要我们添加add_prompts
    if not add_prompts:
        additional_prompts = []
    concepts_ = []
    for concept in concepts:
        old_texts.append(f'{concept}')
        for prompt in additional_prompts:
            old_texts.append(prompt.format(concept=concept)) # old_texts存储的是additional_prompts加上单独的concept结果
        length = 1 + len(additional_prompts)
        concepts_.extend([concept]*length) # concepts_存储的是concepts的重复length次
    # 这个guide_concepts是一个指导概念，指导概念是要将原来的概念编辑为新的概念
    if guided_concepts is None:
        new_texts = [' ' for _ in old_texts]
        print_text+=f'-towards_uncond'
    # bias的时候这里需要重新设置
    elif guided_concepts == 'bias':
        new_texts = []
        for item in old_texts:
            # 假设old_texts中的item已经是单数形式，直接使用item.lower()获取单数小写形式
            # 如果item以's'结尾，则已经是复数形式，否则添加's'来形成复数
            item_singular = item.lower()
            item_plural = item.lower() + 's' if not item.lower().endswith('s') else item.lower()
            # new_texts.append(f"Make sure the gender ratio is evenly distributed and avoid reinforcing gender bias in professions to achieve a equal number of male {item_singular} and female {item_singular}") # 1
            # new_texts.append(f"equal number of male {item_singular} and female {item_singular}, avoid gender bias in {item_plural}")
            new_texts.append(f"the number of male {item_singular} and female {item_singular} is equal, avoid unequal numbers of men and women in {item_singular}")
            # new_texts.append(f"avoid reinforcing gender bias in {item_singular} to achieve a equal number of male {item_singular} and female {item_singular}") # 2
        #     # new_texts.append(f"Create an image of a modern hospital scene featuring a balanced group of {item_plural}. There should be an equal number of male and female {item_plural}, precisely one male {item_plural} on the left side and one female {item_plural} on the right side, ensuring a focus on the diversity of {item_singular}")
        # #     new_texts.append(f"Make sure the gender ratio is evenly distributed among {item_plural} and avoid reinforcing gender bias in {item_plural} to achieve a one to one ratio of male and female {item_singular}")
        # #     # new_texts.append(f"Ensure an even gender distribution among {item_plural} to achieve a one-to-one ratio of male to female, avoiding the reinforcement of gender bias within the {item_singular}")
        # new_texts = ["Make sure the gender ratio is evenly distributed and avoid reinforcing gender bias in professions to achieve a one to one ratio of male and female " + item.lower() for item in old_texts]
        # new_texts = ["Ensure an even gender distribution without reinforcing professional gender bias to achieve a one-to-one ratio of male to female " + item.lower() + " " + item.lower() for item in old_texts]
        # new_texts = ["Make sure the gender ratio is evenly distributed and reinforcing gender bias in professions to achieve a one to one ratio of male and female " + item.lower() + " " + item.lower() for item in old_texts]
        # new_texts = ["The image have an equal number of male and female "+ item.lower() for item in old_texts]
        print_text+=f'-towards_bias'
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
            new_texts = reduce(operator.concat, new_texts) # 将二维的列表转化为一维的列表
            print_text+=f'-towards'
            for t in new_texts:
                if t not in print_text:
                    print_text+=f'-{t}'
    #   相等      
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
    # 先不保留知识，UCE
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
    # 其实是有1200个需要编辑的,所以
    # 模型ldm_stable
    # old_texts
    # new_texts
    # P_outs = get_project_output(ldm_stable, output_concepts, percentage_of_smallest_singular=0.5, batch_size=1)
    batch_size = 1
    print("len(old_texts):", len(old_texts))
    cache_c = torch.zeros(768, 768, device=device)
    # model_path = f'/mnt/bn/intern-disk/mlx/users/wrp/uce_nullspace/model_save/art_erase/unbias/debias_5_lamda_10_bias_batch_3_lamb_0.1_num_smallest_50.pt'
    # ldm_stable.unet.load_state_dict(torch.load(model_path, map_location=device))
    last_saved_model_state = ldm_stable.unet.state_dict()
    get_saved_model_state = None
    log_file_path = '/mlx_devbox/users/wangruipeng/playground/paper_run/uce_nullspace/unified-concept-editing-main/train_unbias/log_file_10_race.txt'  # 替换为您希望保存日志文件的路径
    # 打开文件以追加模式
    with open(log_file_path, 'a') as log_file:
        for i in tqdm(range(0, len(old_texts), batch_size)):
            old_text = old_texts[i:i + batch_size]
            new_text = new_texts[i:i + batch_size]
            # 保存当前模型状态
            current_model_state = ldm_stable.unet.state_dict()
            ori_path = '/mlx_devbox/users/wangruipeng/playground/paper_run/uce_nullspace/unified-concept-editing-main/data/bias/bias/bias/'
            counts_save = []
            for i in range(50):
                min_difference = 100
                # if i !=0:
                P_outs = K_means_output(ldm_stable, 'doctor', i)
                for num_smallest_singular in range(50, 500, 10):
                    P = get_project_input_3(ldm_stable, coco_path, num_smallest_singular=num_smallest_singular, batch_size=16)
                    batch_index = i // batch_size
                    print("old_text:", old_text[0])
                    generate_images(ldm_stable, concept=old_text[0], num_samples=50)
                    img_path = '/mlx_devbox/users/wangruipeng/playground/paper_run/uce_nullspace/unified-concept-editing-main/data/bias/bias/bias/class'
                    counts_indian,counts_asian,counts_african,counts_european,counts_latino = CLIP_classification_race(im_path=img_path, attributes='indian, asian, african, european, latino', prompts_path='/mlx_devbox/users/wangruipeng/playground/paper_run/uce_nullspace/unified-concept-editing-main/data/profession/Analyst/analyst_origin.csv', save_path=None, from_case=0, till_case=10000)
                    counts_dict = {
                        'indian doctor': counts_indian,
                        'asian doctor': counts_asian,
                        'african doctor': counts_african,
                        'european doctor': counts_european,
                        'latino doctor': counts_latino
                    }
                    # 使用sorted函数根据值进行排序，并返回一个包含元组的列表
                    # for i in range(50):
                    sorted_counts = sorted(counts_dict.items(), key=lambda item: item[1])                        
                    # 将上面的五个变量按照大小顺序排序
                    name = sorted_counts[0][0]
                    counts = sorted_counts[0][1]
                    print(f'name:{name}, counts:{counts}')
                    min_difference = min(min_difference, abs(counts - 10))
                    # if counts >= 11:
                    #     new_text = [f'female {old_text[0]}']
                    # elif counts <= 9:
                    #     new_text = [f'male {old_text[0]}']
                        
                    log_file.write(f"new_text old_text: {[old_text[0],name]}\n")  # 记录新旧文本
                    name = [name]
                    model_save_path_with_index = args.model_save_path.replace('.pt', f'_race_batch_{batch_index}_i_{i}_lamb_{lamb}_num_smallest_{num_smallest_singular}.pt')
                    concepts_save_path_with_index = args.concepts_save_path.replace('.txt', f'_race_batch_{batch_index}_i_{i}_lamb_{lamb}_num_smallest_{num_smallest_singular}.txt')
                    
                    ldm_stable, cache_c = alpha_edit_2(ldm_stable=ldm_stable, old_text_=old_text, new_text_=name, add=False, retain_text_=retain_texts, lamb=lamb, erase_scale=erase_scale, preserve_scale=preserve_scale, technique=technique, lamda=lamda, P=P, cache_c=cache_c, P_outs=P_outs)
                    generate_images(ldm_stable, concept=old_text[0], num_samples=50)
                    counts_indian,counts_asian,counts_african,counts_european,counts_latino = CLIP_classification_race(im_path=img_path, attributes='indian, asian, african, european, latino', prompts_path='/mlx_devbox/users/wangruipeng/playground/paper_run/uce_nullspace/unified-concept-editing-main/data/profession/Analyst/analyst_origin.csv', save_path=None, from_case=0, till_case=10000)
                    # counts_indian,counts_asian,counts_african,counts_european,counts_latino = CLIP_classification_race(im_path=img_path, attributes='indian, asian, african, european, latino', prompts_path='/mlx_devbox/users/wangruipeng/playground/paper_run/uce_nullspace/unified-concept-editing-main/data/profession/Analyst/analyst_origin.csv', save_path=None, from_case=0, till_case=10000)
                    counts_dict = {
                        'asian doctor': counts_asian,
                        'african doctor': counts_african,
                        'european doctor': counts_european,
                        'latino doctor': counts_latino,
                        'indian doctor': counts_indian,
                    }
                    print("counts_dict:", counts_dict)  
                    sorted_counts = sorted(counts_dict.items(), key=lambda item: item[1])  
                    # print("counts_dict:", counts_dict) 
                    counts = [count for _, count in sorted_counts]               
                    # counts_save.append(counts)
                    # log_file.write(f'counts_save: {counts_save}\n')  # 记录 counts_save  
                    # 直接找到最优点
                    # counts = list(sorted_counts.values())
                    average = sum(counts) / len(counts)
                    print(f'average:{average}')
                    sum_of_abs_diff = sum(abs(count - average) for count in counts)
                    if abs(sorted_counts[0][1]-sorted_counts[4][1])<5:
                        log_file.write("find all\n")  # 记录日志
                        torch.save(ldm_stable.unet.state_dict(), model_save_path_with_index)
                        # 保存概念
                        with open(concepts_save_path_with_index, 'w') as fp:
                            json.dump(concepts, fp)
                        sys.exit()
                        break
                    elif sum_of_abs_diff<=10:
                        log_file.write("find near all\n")  # 记录日志
                        torch.save(ldm_stable.unet.state_dict(), model_save_path_with_index)
                        # 保存概念
                        with open(concepts_save_path_with_index, 'w') as fp:
                            json.dump(concepts, fp)
                        break
                    if counts_dict[name[0]] == 10:
                        # 保存模型
                        # torch.save(ldm_stable.unet.state_dict(), model_save_path_with_index)
                        # # 保存概念
                        # with open(concepts_save_path_with_index, 'w') as fp:
                        #     json.dump(concepts, fp)
                        # # 更新最后一个保存点的模型状态
                        last_saved_model_state = ldm_stable.unet.state_dict()
                        # ldm_stable.unet.load_state_dict(last_saved_model_state)   
                        torch.cuda.empty_cache()
                        log_file.write("find one!\n")  # 记录日志
                        break  # 跳到下一个 i
                    # 找到一个新的比较好的点：get_saved_model_state记录下来模型的权重参数，但是还是加载最开始的模型权重，只有找到最优和执行完循环找最优才改变模型权重
                    elif abs(10 - counts_dict[name[0]]) < min_difference:
                        # 如果不满足条件，则恢复到上一个保存点的模型状态
                        min_difference = abs(10 - counts_dict[name[0]]) 
                        get_saved_model_state = ldm_stable.unet.state_dict()
                        ldm_stable.unet.load_state_dict(last_saved_model_state)
                        # ldm_stable.unet.load_state_dict(last_saved_model_state)
                        log_file.write("find a better one\n")  # 记录日志
                        torch.cuda.empty_cache()
                    # 不满足上面的条件的时候就直接加载最开始的模型就好了
                    else:
                        ldm_stable.unet.load_state_dict(last_saved_model_state)   
                    if num_smallest_singular == 500:
                        # rate = min(abs(10 - x) for x in counts_save)
                        # log_file.write(f'rate: {rate}\n')  # 记录 rate
                        ldm_stable.unet.load_state_dict(get_saved_model_state)
                        last_saved_model_state = ldm_stable.unet.state_dict()
                        torch.cuda.empty_cache()
                        log_file.write("only can do this\n")  # 记录日志
                        torch.save(ldm_stable.unet.state_dict(), model_save_path_with_index)
                        # 保存概念
                        with open(concepts_save_path_with_index, 'w') as fp:
                            json.dump(concepts, fp)
                    log_file.write(f'counts:{name} {min_difference}\n')