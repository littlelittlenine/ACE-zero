# 擦除, alphaedit编辑实现偏见消除
# 专注于nudity的擦除，我们可以这样，目前看20000个概念作为正交集是不正确的，我觉得可以对nudity加高斯噪声人工制作一个数据集去获得正交空间
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
# 擦除风格，首先是1000个擦除看
# 这里需要我人工加高斯噪声创造一下数据集比较好，然后再去做投影矩阵
# 找到和nudity在输出端最接近的99个单词，在ldm_stable的输出端
def get_top_100_indices_torch(lst):
    print('lst[0]:',lst[0])
    print('len(lst):',len(lst))
    tensor = torch.tensor(lst)
    tensor = tensor.unsqueeze(0)
    _, top_100_indices = torch.topk(tensor, 1)
    return top_100_indices.flatten().tolist()  # 确保返回扁平化的索引列表
def get_bottom_100_indices_torch(lst):
    tensor = torch.tensor(lst)
    tensor = tensor.unsqueeze(0)
    _, bottom_100_indices = torch.topk(tensor, 1, largest=False)
    return bottom_100_indices.flatten().tolist()
def cosine_similarity(a, b):
    """计算两个向量的余弦相似度"""
    # print('a.size:',a.size()) # (1,320)
    a_norm = a / a.norm(dim=-1, keepdim=True)  # 归一化向量 a
    b_norm = b / b.norm(dim=-1, keepdim=True)  # 归一化向量 b
    return (a_norm * b_norm).sum(dim=-1)  # 返回余弦相似度
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
def K_means(ldm_stable, vocab_path, batch_size, with_to_k=True):
    with open(vocab_path, 'r') as file:
        vocab_dict = json.load(file)
    extracted_texts_all = [key for key in vocab_dict.keys()]
    extracted_texts_all = [item.replace("</w>", "") for item in extracted_texts_all]
    extracted_texts_all = list(dict.fromkeys(extracted_texts_all))
    # 得到输出
    print("output_extracted:",extracted_texts_all[200:300])
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
    for layer_num in range(29,len(projection_matrices)):
        scores = []
        total_embeddings = None
        for i in tqdm(range(0, len(extracted_texts_all), 1)):  # 使用 tqdm 监控进度
            batch_prompts = extracted_texts_all[i:i + 1]
            # 清理每个 prompt，去掉不必要的引号和多余的空格
            cleaned_prompts = [prompt.replace('“', '').replace('”', '').strip() for prompt in batch_prompts]
            # 分词并编码为 token id
            # print('cleaned_prompts:',cleaned_prompts)
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
            # 获取文本的嵌入
            with torch.no_grad():  # 不计算梯度以节省显存
                idx = text_input.input_ids[0].tolist().index(49407)
                idx_nudity = nudity_input.input_ids[0].tolist().index(49407)
                text_embeddings = ldm_stable.text_encoder(text_input.input_ids.to(ldm_stable.device))[0]
                output_embeddings = projection_matrices[layer_num](text_embeddings).detach()
                nudity_embeddings = ldm_stable.text_encoder(nudity_input.input_ids.to(ldm_stable.device))[0]
                nudity_output_embeddings = projection_matrices[layer_num](nudity_embeddings).detach()
                # text_embeddings = text_embeddings.detach()  # 从计算图中剥离张量 # (1,77,768)
                output_embeddings = output_embeddings[:,idx-1,:]
                nudity_output_embeddings = nudity_output_embeddings[:,idx_nudity-1,:]
                # print('output_embeddings.size():',output_embeddings.size())
                # 计算相似度得分
                print('output_embeddings:',output_embeddings.size())
                score = cosine_similarity(output_embeddings, nudity_output_embeddings)
                scores.append(score)
        top_100_indices_torch = get_top_100_indices_torch(scores)
        print('top_100_indices_torch:',top_100_indices_torch)
        top_100_words = [extracted_texts_all[idx] for idx in top_100_indices_torch]        
        # 构建文件名
        filename = f"/mlx_devbox/users/wangruipeng/playground/paper_run/uce_nullspace/unified-concept-editing-main/data/nudity_similar/top_100_words_layer_{layer_num}.csv"
        # 保存到CSV文件
        with open(filename, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Word'])  # 写入表头
            for word in top_100_words:
                writer.writerow([word])  # 写入单词
        # 根据top_100_words获得投影P_out，读取file_name中的文本
        data = pd.read_csv(filename)
        total_embeddings = None
        for i in tqdm(range(0, len(data['Word']), 1)):  # 使用 tqdm 监控进度
            batch_prompts = data['Word'][i:i + 1]
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
                output_embeddings = projection_matrices[layer_num](text_embeddings).detach()       
                output_embeddings = output_embeddings[:,idx-1,:]
                # print('output_embeddings.size:',output_embeddings.size())
                if total_embeddings is None:
                    total_embeddings = output_embeddings
                else:
                    total_embeddings = torch.cat((total_embeddings, output_embeddings), dim=0)
        # print("Total embeddings size:", total_embeddings.size())
        # # 计算转置乘积
        # product = torch.mm(total_embeddings.T, total_embeddings)
        # print("Product size:", product.size())
        # # 进行 SVD 分解
        # U, S, _ = torch.linalg.svd(product, full_matrices=False)
        # print("Singular values size:", S.size())

        # # 计算选择的最小奇异值的数量
        # total_singular_values = S.size(0)
        # num_smallest_singular = max(1, int(total_singular_values * percentage_of_smallest_singular))  # 确保至少选择1个

        # # 选择最小的 N 个奇异值的索引
        # smallest_indices = torch.topk(S, num_smallest_singular, largest=False).indices
        # smallest_indices = smallest_indices.sort().values
        
        # # 计算投影矩阵
        # projection_matrix = U[:, smallest_indices] @ U[:, smallest_indices].T
        # print("Projection matrix size:", projection_matrix.size())
        # P_outs.append(projection_matrix)
        P_out = orthogonal_projection_matrix_torch(total_embeddings, device)
        P_outs.append(P_out)        
    return P_outs
def K_means_output(ldm_stable,num_smallest_singular=100, with_to_k=True):
    # with open(vocab_path, 'r') as file:
    #     vocab_dict = json.load(file)
    # extracted_texts_all = [key for key in vocab_dict.keys()]
    # extracted_texts_all = [item.replace("</w>", "") for item in extracted_texts_all]
    # extracted_texts_all = list(dict.fromkeys(extracted_texts_all))
    # # 得到输出
    # print("output_extracted:",extracted_texts_all[200:300])
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
    for layer_num in range(len(projection_matrices)):
        scores = []
        total_embeddings = None
        filename = f"/mlx_devbox/users/wangruipeng/playground/paper_run/uce_nullspace/unified-concept-editing-main/data/nudity_similar/top_100_words_layer_{layer_num}.csv"
        # # 保存到CSV文件
        # with open(filename, 'w', newline='', encoding='utf-8') as file:
        #     writer = csv.writer(file)
        #     writer.writerow(['Word'])  # 写入表头
        #     for word in top_100_words:
        #         writer.writerow([word])  # 写入单词
        # 根据top_100_words获得投影P_out，读取file_name中的文本
        data = pd.read_csv(filename)
        data = [artist for artist in data['Word'] if isinstance(artist, str)]
        total_embeddings = None
        for i in tqdm(range(0, 50, 1)):  # 使用 tqdm 监控进度
            batch_prompts = data[i:i + 1]
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
                output_embeddings = projection_matrices[layer_num](text_embeddings).detach()       
                output_embeddings = output_embeddings[:,idx-1,:]
                # print('output_embeddings.size:',output_embeddings.size())
                if total_embeddings is None:
                    total_embeddings = output_embeddings
                else:
                    total_embeddings = torch.cat((total_embeddings, output_embeddings), dim=0)
        # P_out = orthogonal_projection_matrix_torch(total_embeddings, device)
        # P_outs.append(P_out) 
        product = total_embeddings.T @ total_embeddings # (320,320)
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
        P_out = U[:, smallest_indices] @ U[:, smallest_indices].T 
        # print("P_out:",P_out) 
        P_outs.append(P_out)     
    return P_outs
def old_expand(old_emb):
    # 确定原始嵌入所在的设备
    device = old_emb.device
    print("old_emb:", old_emb.size())
    # 计算原始嵌入的标准差
    std_dev = torch.std(old_emb)

    # 确定噪声的标准差，这里我们将其设置为原始嵌入标准差的1/10
    noise_std_dev = std_dev / 10

    # 生成999个高斯噪声向量，形状为 (999, 768)，并确保它们在相同的设备上
    noise = torch.randn(49, old_emb.size(1), device=device) * noise_std_dev
    print("noise:", noise.size())
    # 将原始嵌入向量扩展为形状 (1, 768) 并复制999次，确保在相同的设备上
    old_emb_expanded = old_emb.repeat(49, 1)

    # 将噪声向量加到扩展的原始嵌入向量上
    new_embeddings = old_emb_expanded + noise

    # 将原始嵌入向量作为第一个元素，与新的嵌入向量堆叠起来，形成形状为 (1000, 768) 的张量
    new_embeddings = torch.vstack((old_emb, new_embeddings))
    print('new_embeddings.size():', new_embeddings.size())
    return new_embeddings

    
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
# 获得投影矩阵
# 可能是获得投影矩阵的问题
# 如何获得呢
def get_project_input(ldm_stable, data_path, subject_column='subject', num_smallest_singular=300, batch_size=16):
    # 读取数据
    data = pd.read_csv(data_path)
    # 初始化一个张量来存储嵌入
    total_embeddings = torch.zeros(768, 768, device='cuda')
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
            text_embeddings = ldm_stable.text_encoder(text_input.input_ids.to(ldm_stable.device))[0]
            text_embeddings = text_embeddings.detach()  # 从计算图中剥离张量 # (1,77,768)
            idx = text_input.input_ids[0].tolist().index(49407) 
            text_embeddings = text_embeddings[:,1:idx+1:,:]
        # print("text_embeddings:", text_embeddings)
        # print(text_embeddings.shape)
        context = text_embeddings
        print("context.size", context.size())
        text_embeddings = context.permute(1,2,0) # (77,768,16)
        text_embeddings_T = context.permute(1,0,2)
        print(text_embeddings[0]) 
        print(text_embeddings_T[0])
        # text_embeddings = context.reshape(context.shape[1], context.shape[2], context.shape[0]) # (77,768,16)
        # print("context.size", context.size())
        # text_embeddings_T = context.reshape(context.shape[1], context.shape[0], context.shape[2]) # (77,16,768)
        # 更新总嵌入张量
        # print("text_embeddings.shape:",text_embeddings.shape)
        # print("text_embeddings_T.shape:",text_embeddings_T.shape)
        # print("text_embeddings @ text_embeddings_T.shape:",(text_embeddings @ text_embeddings_T).shape)
        # print("text_embeddings @ text_embeddings_T:",(text_embeddings @ text_embeddings_T)[1])
        total_embeddings += (text_embeddings @ text_embeddings_T).sum(dim=0)  # (768,768)
        # 释放内存
        del text_input, text_embeddings
        # torch.cuda.empty_cache()  # 清理未使用的缓存
    print("total_embeddings:",total_embeddings)
    print("total_embeddings.size:",total_embeddings.size())
    # 计算(768, 76*5000) @ (76*5000, 768)
    # product = total_embeddings.T @ total_embeddings

    # 进行 SVD 分解
    U, S, _ = torch.linalg.svd(total_embeddings, full_matrices=False)
    print("Singular values size:", S.size())
    
    # 打印一下最小的50个奇异值看一下数据分布
    print(f"Smallest 50 singular values: {S[-50:]}")
    
    # 选择最小的 N 个奇异值的索引
    smallest_indices = torch.topk(S, num_smallest_singular, largest=False).indices
    smallest_indices = smallest_indices.sort().values
    print(f"Indices of the smallest {num_smallest_singular} singular values: {smallest_indices}")
    
    # 计算投影矩阵
    projection_matrix = U[:, smallest_indices] @ U[:, smallest_indices].T
    print("Projection matrix size:", projection_matrix.size())
    return projection_matrix
# 两个代码得到的结果输出来看确实是不同的
# 但是为什么不同呢 （16,77,768) 
def get_project_input_3(ldm_stable, data_path, subject_column='subject', num_smallest_singular=300, batch_size=16):
    # 读取数据
    data = pd.read_csv(data_path)
    # 初始化一个张量来存储嵌入
    data = [artist for artist in data[subject_column] if isinstance(artist, str)]
    # data = data[data[subject_column].apply(lambda x: isinstance(x, str))]
    print("data:",data[:20])
    total_embeddings = None
    print(len(data))
    for i in tqdm(range(0, len(data), batch_size)):  # 使用 tqdm 监控进度
        batch_prompts = data[i:i + batch_size]
        # print("input_batch_prompts:", batch_prompts)
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
            idx_list = [input_ids.tolist().index(49407) for input_ids in text_input.input_ids]
            text_embeddings = ldm_stable.text_encoder(text_input.input_ids.to(ldm_stable.device))[0]
            print("text_embeddings:", text_embeddings.size())
            text_embeddings = text_embeddings.detach()  # 从计算图中剥离张量 # (1,77,768)
            # text_embeddings = text_embeddings[:,1:idx+1:,:]
            batch_embeddings = []
            for j, idx in enumerate(idx_list):
                batch_embeddings.append(text_embeddings[j, 1:idx+1, :])
            # 将每个输入的嵌入拼接起来
            batch_embeddings = torch.cat(batch_embeddings, dim=0)
        # 将形状转换为 (16*76, 768)
        # print("text_embeddings:", text_embeddings) # (16, 7, 768)
        # print("text_embeddings,size:", text_embeddings.size()) # (16, 7, 768)
        text_embeddings = text_embeddings.reshape(-1, batch_embeddings.size(-1))
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
    print("Singular values size:", S.size())
    
    # 打印一下最小的50个奇异值看一下数据分布
    print(f"Smallest 50 singular values: {S[-50:]}")
    
    # 选择最小的 N 个奇异值的索引
    smallest_indices = torch.topk(S, num_smallest_singular, largest=False).indices
    smallest_indices = smallest_indices.sort().values
    print(f"Indices of the smallest {num_smallest_singular} singular values: {smallest_indices}")
    
    # 计算投影矩阵
    projection_matrix = U[:, smallest_indices] @ U[:, smallest_indices].T
    print("Projection matrix size:", projection_matrix.size())
    
    return projection_matrix

def get_project_input_expand(ldm_stable, data_path, subject_column='subject', num_smallest_singular=300, batch_size=1):
    # 读取数据
    data = pd.read_csv(data_path)
    # 过滤掉非字符串类型的行
    data = data[data[subject_column].apply(lambda x: isinstance(x, str))]
    total_embeddings = None
    print(len(data))
    
    for i in tqdm(range(0, len(data[subject_column]), batch_size)):  # 使用 tqdm 监控进度
        batch_prompts = data[subject_column][i:i + batch_size].tolist()  # 将选定的批次转换为列表

        # 生成额外提示并与原始提示结合
        additional_prompts = []
        for concept in batch_prompts:
            additional_prompts.append(f'painting by {concept}')
            additional_prompts.append(f'art by {concept}')
            additional_prompts.append(f'artwork by {concept}')
            additional_prompts.append(f'picture by {concept}')
            additional_prompts.append(f'style of {concept}')
        
        # 将原始提示与额外提示结合
        combined_prompts = batch_prompts + additional_prompts
        
        # 清理每个 prompt，去掉不必要的引号和多余的空格
        cleaned_prompts = [prompt.replace('“', '').replace('”', '').strip() for prompt in combined_prompts]

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
            # 初始化一个列表来存储每个文本嵌入
            batch_text_embeddings = []
            print(len(text_input.input_ids))
            # print("text_input.input_ids:",text_input.input_ids)
            for idx in range(len(text_input.input_ids)):  # 遍历每个 input_id
                input_id = text_input.input_ids[idx].tolist()
                if 49407 in input_id:  # 确保 49407 存在于 input_id 中
                    print(f"input_id: {input_id}")
                    idx_value = input_id.index(49407)
                    print(f"idx_value: {idx_value}")
                    text_embeddings = ldm_stable.text_encoder(text_input.input_ids[idx].unsqueeze(0).to(ldm_stable.device))[0]
                    text_embeddings = text_embeddings.detach()  # 从计算图中剥离张量
                    text_embeddings = text_embeddings[:, idx_value-1:idx_value:,:]  # 获取相关的嵌入
                    batch_text_embeddings.append(text_embeddings)
                else:
                    # 如果 49407 不在 input_ids 中，可以处理一下
                    batch_text_embeddings.append(torch.zeros(1, 1, ldm_stable.text_encoder.config.hidden_size).to(ldm_stable.device))  # 或者其他适当的替代值

            # 将所有的嵌入拼接成一个张量
            if batch_text_embeddings:
                batch_text_embeddings = torch.cat(batch_text_embeddings, dim=0)
            else:
                batch_text_embeddings = torch.zeros(0, ldm_stable.text_encoder.config.hidden_size).to(ldm_stable.device)  # 确保有值

        # 将形状转换为 (batch_size * 76, 768)
        text_embeddings = batch_text_embeddings.reshape(-1, batch_text_embeddings.size(-1))

        # 初始化总嵌入张量或连接新的嵌入
        if total_embeddings is None:
            total_embeddings = text_embeddings
        else:
            total_embeddings = torch.cat((total_embeddings, text_embeddings), dim=0)

        # 释放内存
        del text_input, batch_text_embeddings
        torch.cuda.empty_cache()  # 清理未使用的缓存

    # 计算(768, batch_size*76) @ (batch_size*76, 768)
    product = total_embeddings.T @ total_embeddings
    
    # 进行 SVD 分解
    U, S, _ = torch.linalg.svd(product, full_matrices=False)
    print("Singular values size:", S.size())
    
    # 打印一下最小的50个奇异值看一下数据分布
    print(f"Smallest 50 singular values: {S[-50:]}")
    
    # 选择最小的 N 个奇异值的索引
    smallest_indices = torch.topk(S, num_smallest_singular, largest=False).indices
    smallest_indices = smallest_indices.sort().values
    print(f"Indices of the smallest {num_smallest_singular} singular values: {smallest_indices}")
    
    # 计算投影矩阵
    projection_matrix = U[:, smallest_indices] @ U[:, smallest_indices].T
    print("Projection matrix size:", projection_matrix.size())
    
    return projection_matrix


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
    P = get_project_input(ldm_stable,'/mlx_devbox/users/wangruipeng/playground/paper_run/uce_nullspace/unified-concept-editing-main/data/extracted_subjects_big_only_5000.csv')
    # # print("P.size",P.size())
    torch.save(P, "/mnt/bn/intern-disk/mlx/users/wrp/uce_nullspace/model_save/nullspace_project/input_5000/null_space_project_subject_5000_3_200.pt") 
    print(f'当前模型状态: 将"{str(old_text_)}"编辑为"{str(new_texts)}"，并保留"{str(ret_texts)}"')
    return ldm_stable
# 我希望按照一次编辑100个实现目标
def alpha_edit_2(ldm_stable, old_text_, new_text_, retain_text_, add=False, layers_to_edit=None, lamb=0.1, erase_scale=0.1, preserve_scale=0.1, with_to_k=True, technique='tensor', P=None, lamda=10, cache_c = None):
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
    # print("layers_to_edit", layers_to_edit)      
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
    # print("layers_to_edit", layers_to_edit) 
    ######################## 开始编辑 ###################################
    print(len(projection_matrices))
    # 这个是Wv矩阵和Wk矩阵的分界线
    # P = torch.load("/mnt/bn/intern-disk/mlx/users/wrp/uce_nullspace/model_save/nullspace_project/input_1000/null_space_project_subject_1000_3_300.pt")
    P1 = P.clone()
    P2 = P.clone()
    for layer_num in range(len(projection_matrices)):
        # 我们要寻找的是和Wv矩阵对应的Wk矩阵,以及和Wk矩阵对应的Wv矩阵
        # print("layers_to_edit", layers_to_edit) 
        if (layers_to_edit is not None) and (layer_num not in layers_to_edit):
            continue
        # print(f'Editing layer {layer_num}')
        with torch.no_grad():  # 计算图控制
            # 获取当前投影矩阵的权重
            W_old = projection_matrices[layer_num].weight.detach()  # 使用 .detach() 防止反向传播

            # 计算目标概念的嵌入
            values = []
            old_embeddings = []
            new_embeddings = []
            # 每一层初始化
            for_mat1 = torch.eye(projection_matrices[layer_num].weight.shape[1],dtype=torch.float,device = projection_matrices[layer_num].weight.device)
            for_mat2 = torch.zeros(768,768, device=W_old.device)
            for_mat3 = torch.zeros(projection_matrices[layer_num].weight.shape[0],768, device=W_old.device)
            # 当输入是一个的时候，比如(Van Gao, art)
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
                # 计算有效 token 的索引, 这个其实是有问题的，因为final_token_idx
                final_token_idx = text_input.attention_mask[0].sum().item() - 2
                final_token_idx_new = text_input.attention_mask[1].sum().item() - 2
                farthest = max(final_token_idx_new, final_token_idx)
                
                old_emb = text_embeddings[0][final_token_idx:len(text_embeddings[0])-max(0, farthest-final_token_idx)].detach()
                new_emb = text_embeddings[1][final_token_idx_new:len(text_embeddings[1])-max(0, farthest-final_token_idx_new)].detach()  
                # 获取有效嵌入，去除特殊 token
                # 在这里我们得到的待擦除的概念和新的概念的嵌入，去掉的是SOS token但是EOS token还在的
                # 这里一个token对应多个是不是不太好
                # 加上代码
                # 但是这里艺术家的token数是不确定的，那么可以通过art填充是不是补充一下
                ######## 第一种方法
                idx_old = text_input.input_ids[0].tolist().index(49407)
                idx_new = text_input.input_ids[1].tolist().index(49407)
                farthest = max(idx_old, idx_new)

                old_emb = text_embeddings[0][1:farthest+1].detach()
                new_emb = text_embeddings[1][1:farthest+1].detach()               
                # 新增加的代码如上，和下面的代码是两种不同的选择方式
                ######## 第二种方法
                # old_emb = text_embeddings[0][final_token_idx:len(text_embeddings[0])-max(0, farthest-final_token_idx)].detach()
                # new_emb = text_embeddings[1][final_token_idx_new:len(text_embeddings[1])-max(0, farthest-final_token_idx_new)].detach()

                # idx_old = text_input.input_ids[0].tolist().index(25148)
                # idx_new = text_input.input_ids[1].tolist().index(25148)
                # print([idx_old, idx_new])
                # old_emb = text_embeddings[0] # (77, 768)
                # new_emb = text_embeddings[1] # (77, 768)
                # if cnt==0:
                #     old_emb = old_emb[idx_old:idx_old + 2] # （4，768）
                #     new_emb = new_emb[idx_new:idx_new + 2] # （4，768）
                # else:
                # # old_emb = old_emb[final_token_idx:len(old_emb)-max(0,farthest-final_token_idx)] # （76，768）
                #     old_emb = old_emb[idx_old-2:idx_old + 2] # （4，768）
                #     # new_emb = new_emb[final_token_idx_new:len(new_emb)-max(0,farthest-final_token_idx_new)] # （76，768）
                #     new_emb = new_emb[idx_new-2:idx_new + 2] # （4，768）  
                # # 计算均值,这里应该是不需要计算均值的,直接展开操作
                # old_emb_flat = old_emb.view(text_embeddings.size(0), -1)
                # new_emb_flat = new_emb.view(text_embeddings.size(0), -1)
                # old_meb和new_emb的维度都是(768)
                # X(K1K1TP + KpKpTP + I) = RK1TP
                context = old_emb.detach()
                context_vector = context.reshape(context.shape[0], context.shape[1], 1) # (75, 768, 1)
                context_vector_T = context.reshape(context.shape[0], 1, context.shape[1]) # (75, 1, 768)
                value_vector = (new_emb @ W_old.T).detach() # (2, 320, 1)
                value_vector = value_vector.reshape(value_vector.shape[0], value_vector.shape[1],1) # (75, 320, 1)           
                for_mat2 += (context_vector @ context_vector_T).sum(dim=0) # (768,768)累加上去
                o_embs = context @ W_old.T
                R = value_vector - o_embs.unsqueeze(-1) # (X,320,1) @ (X,1,768)
                # print('R:',R.size())
                # print("o_embs:",o_embs.size())
                # print('context_vector_T',context_vector_T.size())
                for_mat3 += (R @ context_vector_T).sum(dim=0) # (768,768)累加上去 # (X,320,1) @ (X,1,768)
                # print("for_mat2",for_mat2)
                # print("for_mat3",for_mat3)
        result1 = lamb * (for_mat2 @ P1 + cache_c @ P1) + lamba * for_mat1
        result2 = lamb * for_mat3 @ P2
                # print("result1.size()",result1.size()) # (768,768)
                # print("result2.size()",result2.size()) # (1280,768)
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
# 将（77，768）里面的每个概念都看作一个概念，相当于一维的张量
# 可以想一下如果是顺序编辑，不考虑保留知识的话
# 输入一个：（4，768)
# 编辑得到第一个deta=(320,768)
# 编辑第二个：（4，758）
# 编辑得到第二个deta=(320,768）
# batch_edit相当于直接将这两个deta相加
# (WK-V1)2 + (W-Wold)2
def get_project_output(ldm_stable, concept='nudity', percentage_of_smallest_singular=0.01, batch_size=1, with_to_k=True):
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

    ### 读取数据 ###
    # data = pd.read_csv(data_path)
    # data = data[data['Artist'].apply(lambda x: isinstance(x, str))]
    # print(len(data))
    P_outs = []
    print("len(projection_matrices)", len(projection_matrices))
    for layer_num in range(len(projection_matrices)):
        # 初始化一个张量来存储嵌入
        # filename = f"/mlx_devbox/users/wangruipeng/playground/paper_run/uce_nullspace/unified-concept-editing-main/data/nudity_similar/top_100_words_layer_{layer_num}.csv"
        filename = f"/mlx_devbox/users/wangruipeng/playground/paper_run/uce_nullspace/unified-concept-editing-main/data/extracted_subjects_big_only_2000.csv"
        data = pd.read_csv(filename)
        data = [artist for artist in data['subject'] if isinstance(artist, str)]
        print('data:',data[:20])
        total_embeddings = None
        for i in tqdm(range(0, len(data), batch_size)):  # 使用 tqdm 监控进度
            batch_prompts = data[i:i + batch_size]
            # 清理每个 prompt，去掉不必要的引号和多余的空格
            # print("output_batch_prompts",batch_prompts)
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
                # print("output_embeddings:", output_embeddings.shape)
                # text_embeddings = text_embeddings.detach()  # 从计算图中剥离张量 # (1,77,768)
                output_embeddings = output_embeddings[:,idx-1,:]
                # print("output_embeddings:", output_embeddings.shape)
            # 将形状转换为 (16*76, 768)
            # print("text_embeddings:", text_embeddings) # (16, 7, 768)
            # print("text_embeddings,size:", text_embeddings.size()) # (16, 7, 768)
            # output_embeddings = output_embeddings.reshape(-1, output_embeddings.size(-1))
            # print("text_embeddings,size:", text_embeddings.size()) # (112, 768)
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
def get_project_output_expand(ldm_stable, concept='nudity', percentage_of_smallest_singular=0.01, batch_size=16, with_to_k=True):
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

    ### 读取数据 ###
    # data = pd.read_csv(data_path)
    # data = data[data['Artist'].apply(lambda x: isinstance(x, str))]
    # print(len(data))
    P_outs = []
    print("len(projection_matrices)", len(projection_matrices))
    for layer_num in range(len(projection_matrices)):
        # 初始化一个张量来存储嵌入
        total_embeddings = None
        # 清理每个 prompt，去掉不必要的引号和多余的空格
        cleaned_prompts = [concept.replace('“', '').replace('”', '').strip()]
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
            print("output_embeddings:", output_embeddings.size())
            # text_embeddings = text_embeddings.detach()  # 从计算图中剥离张量 # (1,77,768)
            output_embeddings = output_embeddings[:,idx-1:idx:,:]
        # 将形状转换为 (16*76, 768)
        # print("text_embeddings:", text_embeddings) # (16, 7, 768)
        # print("text_embeddings,size:", text_embeddings.size()) # (16, 7, 768)
        output_embeddings = output_embeddings.reshape(-1, output_embeddings.size(-1))
        print("output_embeddings:", output_embeddings.size())
        # print("text_embeddings,size:", text_embeddings.size()) # (112, 768)
        # 初始化总嵌入张量或连接新的嵌入
        # if total_embeddings is None:
        #     total_embeddings = output_embeddings
        # else:
        #     total_embeddings = torch.cat((total_embeddings, output_embeddings), dim=0)
        # total_embeddings = output_embeddings
        total_embeddings = old_expand(output_embeddings)
        # 释放内存
        del text_input, text_embeddings, output_embeddings
        torch.cuda.empty_cache()  # 清理未使用的缓存
        # print("Total embeddings size:", total_embeddings.size())
        # # 计算转置乘积
        # product = torch.mm(total_embeddings.T, total_embeddings)
        # print("Product size:", product.size())
        # # 进行 SVD 分解
        # U, S, _ = torch.linalg.svd(product, full_matrices=False)
        # print("Singular values size:", S.size())

        # # 计算选择的最小奇异值的数量
        # total_singular_values = S.size(0)
        # num_smallest_singular = max(1, int(total_singular_values * percentage_of_smallest_singular))  # 确保至少选择1个

        # # 选择最小的 N 个奇异值的索引
        # smallest_indices = torch.topk(S, num_smallest_singular, largest=False).indices
        # smallest_indices = smallest_indices.sort().values
        
        # # 计算投影矩阵
        # projection_matrix = U[:, smallest_indices] @ U[:, smallest_indices].T
        # print("Projection matrix size:", projection_matrix.size())
        # P_outs.append(projection_matrix)
        P_out = orthogonal_projection_matrix_torch(total_embeddings, device)
        P_outs.append(P_out)
    return P_outs
def find_most_diff(ldm_stable, data_path, with_to_k=True):
    # 加载数据集
    df = pd.read_csv(data_path)
    # 选择特定概念的行
    concepts = [artist for artist in df['Artist'] if isinstance(artist, str)]
    print('concepts:',concepts[:20])    
    sub_nets = ldm_stable.unet.named_children()
    ca_layers = []
    # P_outs = []
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
    # 计算每个概念的平均分数
    layer_target = []
    print('concepts:',concepts[:20])
    for layer_num in range(0,len(projection_matrices)):
        scores = []
        total_embeddings = None
        for i in tqdm(range(0, len(concepts), 1)):  # 使用 tqdm 监控进度
            batch_prompts = concepts[i:i + 1]
            print('batch_prompts:',batch_prompts)
            # 清理每个 prompt，去掉不必要的引号和多余的空格
            cleaned_prompts = [prompt.replace('“', '').replace('”', '').strip() for prompt in batch_prompts]
            # 分词并编码为 token id
            # print('cleaned_prompts:',cleaned_prompts)
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
            # 获取文本的嵌入
            with torch.no_grad():  # 不计算梯度以节省显存
                idx = text_input.input_ids[0].tolist().index(49407)
                idx_nudity = nudity_input.input_ids[0].tolist().index(49407)
                text_embeddings = ldm_stable.text_encoder(text_input.input_ids.to(ldm_stable.device))[0]
                output_embeddings = projection_matrices[layer_num](text_embeddings).detach()
                nudity_embeddings = ldm_stable.text_encoder(nudity_input.input_ids.to(ldm_stable.device))[0]
                nudity_output_embeddings = projection_matrices[layer_num](nudity_embeddings).detach()
                # text_embeddings = text_embeddings.detach()  # 从计算图中剥离张量 # (1,77,768)
                output_embeddings = output_embeddings[:,idx-1,:]
                nudity_output_embeddings = nudity_output_embeddings[:,idx_nudity-1,:]
                # print('output_embeddings.size():',output_embeddings.size())
                # 计算相似度得分
                # print('output_embeddings:',output_embeddings.size())
                score = cosine_similarity(output_embeddings, nudity_output_embeddings)
                scores.append(score)
                # print('len(scores):',len(scores))
        bottom_1_indices_torch = get_bottom_100_indices_torch(scores)
        print('scores:',scores)
        print('bottom_1_indices_torch:',bottom_1_indices_torch)
        bottom_1_word = [concepts[idx] for idx in bottom_1_indices_torch]        
        # 构建文件名
        layer_target.append(bottom_1_word)
        filename = f"/mlx_devbox/users/wangruipeng/playground/paper_run/uce_nullspace/unified-concept-editing-main/data/nudity_bottom_similar/top_100_words_layer_{layer_num}.csv"
        # 保存到CSV文件
        with open(filename, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Word'])  # 写入表头
            for word in bottom_1_word:
                writer.writerow([word])  # 写入单词
    # concept_scores = concept_rows.groupby('prompt')['score'].mean()
    return layer_target
def alpha_edit_3(ldm_stable, old_text_, new_text_, retain_text_, add=False, layers_to_edit=None, lamb=0.1, erase_scale=0.1, preserve_scale=0.1, with_to_k=True, technique='tensor', P=None, P_nude=None, lamda=10, cache_c = None, P_outs=None):
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
    # print("layers_to_edit", layers_to_edit)      
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
    # print("layers_to_edit", layers_to_edit) 
    ######################## 开始编辑 ###################################
    print(len(projection_matrices))
    # 这个是Wv矩阵和Wk矩阵的分界线
    # P = torch.load("/mnt/bn/intern-disk/mlx/users/wrp/uce_nullspace/model_save/nullspace_project/input_1000/null_space_project_subject_1000_3_300.pt")
    P1 = P.clone()
    P2 = P.clone()
    for layer_num in range(len(projection_matrices)):
        # 我们要寻找的是和Wv矩阵对应的Wk矩阵,以及和Wk矩阵对应的Wv矩阵
        # print("layers_to_edit", layers_to_edit) 
        if (layers_to_edit is not None) and (layer_num not in layers_to_edit):
            continue
        # print(f'Editing layer {layer_num}')
        with torch.no_grad():  # 计算图控制
            # 获取当前投影矩阵的权重
            W_old = projection_matrices[layer_num].weight.detach()  # 使用 .detach() 防止反向传播

            # 计算目标概念的嵌入
            values = []
            old_embeddings = []
            new_embeddings = []
            # 每一层初始化
            for_mat1 = torch.eye(projection_matrices[layer_num].weight.shape[1],dtype=torch.float,device = projection_matrices[layer_num].weight.device)
            # for_mat2 = torch.zeros(768,768, device=W_old.device)
            # for_mat3 = torch.zeros(projection_matrices[layer_num].weight.shape[0],768, device=W_old.device)
            # 当输入是一个的时候，比如(Van Gao, art)
            context = None
            value_vector = None
            # 如果输入10个，相当于不保存的情况下顺序输入10个
            # 
            is_nude = 0
            for cnt, (old_text, new_text) in enumerate(zip(old_texts, new_texts)):
                text_input = ldm_stable.tokenizer(
                    [old_text, new_text],
                    padding="max_length",
                    max_length=ldm_stable.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                # 获取文本的嵌入
                # 获取nudity的cnt，然后将对应的代码做正交处理
                target_bool = 0
                if 'nudity' in old_text:
                    target_bool = 1
                    is_nude = 1
                text_embeddings = ldm_stable.text_encoder(text_input.input_ids.to(ldm_stable.device))[0]
                # print("text_embeddings",text_embeddings)
                # 计算有效 token 的索引, 这个其实是有问题的，因为final_token_idx
                final_token_idx = text_input.attention_mask[0].sum().item() - 2
                final_token_idx_new = text_input.attention_mask[1].sum().item() - 2
                farthest = max(final_token_idx_new, final_token_idx)
                
                old_emb = text_embeddings[0][final_token_idx:len(text_embeddings[0])-max(0, farthest-final_token_idx)].detach()
                new_emb = text_embeddings[1][final_token_idx_new:len(text_embeddings[1])-max(0, farthest-final_token_idx_new)].detach()  
                # 获取有效嵌入，去除特殊 token
                # 在这里我们得到的待擦除的概念和新的概念的嵌入，去掉的是SOS token但是EOS token还在的
                # 这里一个token对应多个是不是不太好
                # 加上代码
                # 但是这里艺术家的token数是不确定的，那么可以通过art填充是不是补充一下
                ######## 第一种方法
                idx_old = text_input.input_ids[0].tolist().index(49407)
                idx_new = text_input.input_ids[1].tolist().index(49407)
                farthest = max(idx_old, idx_new)

                old_emb = text_embeddings[0][1:farthest+1].detach()
                new_emb = text_embeddings[1][1:farthest+1].detach()
                print("####################################old_emb",old_emb.size()) # (X,768)          
                # 新增加的代码如上，和下面的代码是两种不同的选择方式
                ######## 第二种方法
                # old_emb = text_embeddings[0][final_token_idx:len(text_embeddings[0])-max(0, farthest-final_token_idx)].detach()
                # new_emb = text_embeddings[1][final_token_idx_new:len(text_embeddings[1])-max(0, farthest-final_token_idx_new)].detach()

                # idx_old = text_input.input_ids[0].tolist().index(25148)
                # idx_new = text_input.input_ids[1].tolist().index(25148)
                # print([idx_old, idx_new])
                # old_emb = text_embeddings[0] # (77, 768)
                # new_emb = text_embeddings[1] # (77, 768)
                # if cnt==0:
                #     old_emb = old_emb[idx_old:idx_old + 2] # （4，768）
                #     new_emb = new_emb[idx_new:idx_new + 2] # （4，768）
                # else:
                # # old_emb = old_emb[final_token_idx:len(old_emb)-max(0,farthest-final_token_idx)] # （76，768）
                #     old_emb = old_emb[idx_old-2:idx_old + 2] # （4，768）
                #     # new_emb = new_emb[final_token_idx_new:len(new_emb)-max(0,farthest-final_token_idx_new)] # （76，768）
                #     new_emb = new_emb[idx_new-2:idx_new + 2] # （4，768）  
                # # 计算均值,这里应该是不需要计算均值的,直接展开操作
                # old_emb_flat = old_emb.view(text_embeddings.size(0), -1)
                # new_emb_flat = new_emb.view(text_embeddings.size(0), -1)
                # old_meb和new_emb的维度都是(768)
                # X(K1K1TP + KpKpTP + I) = RK1TP
                if target_bool==1:                    
                    n_embs = new_emb @ W_old.T @ P_outs[(layer_num+16)%32]  # (4,768) @ (768,320) = (4,320)
                else:
                    n_embs = new_emb @ W_old.T # (4,768) @ (768,320) = (4,320)
                # 将new_emb投影到输出端词的正交空间上
                # new_emb = new_emb @ P_outs[layer_num]  
                if cnt == 0 and target_bool!=1:
                    context = old_emb.detach() # (4,768)
                    context_1 = old_emb.detach() 
                    value_vector = n_embs.detach() # (4,320)
                elif target_bool==1:
                    context = torch.cat((context_1,old_emb),dim=0)
                    value_vector = torch.cat((value_vector,n_embs.detach()),dim=0)
                else:
                    context = torch.cat((context,old_emb),dim=0)
                    context_1 = torch.cat((context_1,old_emb),dim=0)
                    value_vector = torch.cat((value_vector,n_embs.detach()),dim=0)
            # 10个样本的context:(40,768), value_vector:(40,320)
            # context = old_emb.detach()
            # context_vector = context.reshape(context.shape[0], context.shape[1], 1) # (75, 768, 1)
            # context_vector_T = context.reshape(context.shape[0], 1, context.shape[1]) # (75, 1, 768)
            # value_vector = (new_emb @ W_old.T).detach() # (2, 320, 1)
            # value_vector = value_vector.reshape(value_vector.shape[0], value_vector.shape[1],1) # (75, 320, 1)   
            # RKTP = X(K1K1TP + KpKpTP + I)  -> VKTP - WKKTP = X(K1K1TP + KpKpTP + I)     
            for_mat2 = (context.T @ context) # (768,768)累加上去
            for_mat3 = value_vector.T @ context - W_old @ for_mat2
            # R = value_vector - o_embs # (X,320,1) @ (X,1,768)
            # print('R:',R.size())
            # print("o_embs:",o_embs.size())
            # print('context_vector_T',context_vector_T.size())
            # for_mat3 += R.T @ context # (320,10) @ (10,768) = (320,768)
                # print("for_mat2",for_mat2)
                # print("for_mat3",for_mat3)
        if is_nude!=1:
            result1 = lamb * (for_mat2 @ P1 + cache_c @ P1) +  lamda * for_mat1
            result2 = lamb * for_mat3 @ P2
        else:
            result1 = lamb * (for_mat2 @ P_nude + cache_c @ P_nude) +  lamda * for_mat1
            result2 = lamb * for_mat3 @ P_nude
        # 上面的代码对应公式：RKTP = X(K1K1TP + KpKpTP + I)
        # 
        # print("result1.size()",result1.size()) # (768,768)
        # print("result2.size()",result2.size()) # (1280,768)
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
    cache_c += (context_1.T @ context_1)
    print(f'当前模型状态: 将"{str(old_text_)}"编辑为"{str(new_texts)}"，并保留"{str(ret_texts)}"')
    return ldm_stable, cache_c
def alpha_edit_4(ldm_stable, old_text_, new_text_, layer_target, retain_text_, add=False, layers_to_edit=None, lamb=0.1, erase_scale=0.1, preserve_scale=0.1, with_to_k=True, technique='tensor', P=None, P_nude=None, lamda=10, cache_c = None, P_outs=None):
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
    # print("layers_to_edit", layers_to_edit)      
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
    # print("layers_to_edit", layers_to_edit) 
    ######################## 开始编辑 ###################################
    print(len(projection_matrices))
    # 这个是Wv矩阵和Wk矩阵的分界线
    # P = torch.load("/mnt/bn/intern-disk/mlx/users/wrp/uce_nullspace/model_save/nullspace_project/input_1000/null_space_project_subject_1000_3_300.pt")
    P1 = P.clone()
    P2 = P.clone()
    for layer_num in range(len(projection_matrices)):
        # 我们要寻找的是和Wv矩阵对应的Wk矩阵,以及和Wk矩阵对应的Wv矩阵
        # print("layers_to_edit", layers_to_edit) 
        if (layers_to_edit is not None) and (layer_num not in layers_to_edit):
            continue
        # print(f'Editing layer {layer_num}')
        with torch.no_grad():  # 计算图控制
            # 获取当前投影矩阵的权重
            W_old = projection_matrices[layer_num].weight.detach()  # 使用 .detach() 防止反向传播

            # 计算目标概念的嵌入
            values = []
            old_embeddings = []
            new_embeddings = []
            # 每一层初始化
            for_mat1 = torch.eye(projection_matrices[layer_num].weight.shape[1],dtype=torch.float,device = projection_matrices[layer_num].weight.device)
            # for_mat2 = torch.zeros(768,768, device=W_old.device)
            # for_mat3 = torch.zeros(projection_matrices[layer_num].weight.shape[0],768, device=W_old.device)
            # 当输入是一个的时候，比如(Van Gao, art)
            context = None
            value_vector = None
            # 如果输入10个，相当于不保存的情况下顺序输入10个
            # 
            is_nude = 0
            for cnt, (old_text, new_text) in enumerate(zip(old_texts, new_texts)):
                target_bool = 0
                if 'nudity' in old_text:
                    print('yesyesyes')
                    target_bool = 1
                    is_nude = 1
                    print([old_text,new_text])
                    new_text = str(layer_target[layer_num][0]) 
                    print([old_text,new_text])               
                text_input = ldm_stable.tokenizer(
                    [old_text, new_text],
                    padding="max_length",
                    max_length=ldm_stable.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                # 获取文本的嵌入
                # 获取nudity的cnt，然后将对应的代码做正交处理
                text_embeddings = ldm_stable.text_encoder(text_input.input_ids.to(ldm_stable.device))[0]
                # print("text_embeddings",text_embeddings)
                # 计算有效 token 的索引, 这个其实是有问题的，因为final_token_idx
                final_token_idx = text_input.attention_mask[0].sum().item() - 2
                final_token_idx_new = text_input.attention_mask[1].sum().item() - 2
                farthest = max(final_token_idx_new, final_token_idx)
                
                old_emb = text_embeddings[0][final_token_idx:len(text_embeddings[0])-max(0, farthest-final_token_idx)].detach()
                new_emb = text_embeddings[1][final_token_idx_new:len(text_embeddings[1])-max(0, farthest-final_token_idx_new)].detach()  
                # 获取有效嵌入，去除特殊 token
                # 在这里我们得到的待擦除的概念和新的概念的嵌入，去掉的是SOS token但是EOS token还在的
                # 这里一个token对应多个是不是不太好
                # 加上代码
                # 但是这里艺术家的token数是不确定的，那么可以通过art填充是不是补充一下
                ######## 第一种方法
                idx_old = text_input.input_ids[0].tolist().index(49407)
                idx_new = text_input.input_ids[1].tolist().index(49407)
                farthest = max(idx_old, idx_new)

                old_emb = text_embeddings[0][1:farthest+1].detach()
                new_emb = text_embeddings[1][1:farthest+1].detach()
                # print("####################################old_emb",old_emb.size()) # (X,768)          
                # 新增加的代码如上，和下面的代码是两种不同的选择方式
                ######## 第二种方法
                # old_emb = text_embeddings[0][final_token_idx:len(text_embeddings[0])-max(0, farthest-final_token_idx)].detach()
                # new_emb = text_embeddings[1][final_token_idx_new:len(text_embeddings[1])-max(0, farthest-final_token_idx_new)].detach()

                # idx_old = text_input.input_ids[0].tolist().index(25148)
                # idx_new = text_input.input_ids[1].tolist().index(25148)
                # print([idx_old, idx_new])
                # old_emb = text_embeddings[0] # (77, 768)
                # new_emb = text_embeddings[1] # (77, 768)
                # if cnt==0:
                #     old_emb = old_emb[idx_old:idx_old + 2] # （4，768）
                #     new_emb = new_emb[idx_new:idx_new + 2] # （4，768）
                # else:
                # # old_emb = old_emb[final_token_idx:len(old_emb)-max(0,farthest-final_token_idx)] # （76，768）
                #     old_emb = old_emb[idx_old-2:idx_old + 2] # （4，768）
                #     # new_emb = new_emb[final_token_idx_new:len(new_emb)-max(0,farthest-final_token_idx_new)] # （76，768）
                #     new_emb = new_emb[idx_new-2:idx_new + 2] # （4，768）  
                # # 计算均值,这里应该是不需要计算均值的,直接展开操作
                # old_emb_flat = old_emb.view(text_embeddings.size(0), -1)
                # new_emb_flat = new_emb.view(text_embeddings.size(0), -1)
                # old_meb和new_emb的维度都是(768)
                # X(K1K1TP + KpKpTP + I) = RK1TP
                if target_bool==1:                    
                    n_embs = new_emb @ W_old.T @ P_outs[(layer_num+16)%32]  # (4,768) @ (768,320) = (4,320)
                else:
                    n_embs = new_emb @ W_old.T # (4,768) @ (768,320) = (4,320)
                # 将new_emb投影到输出端词的正交空间上
                # new_emb = new_emb @ P_outs[layer_num]  
                if cnt == 0 and target_bool!=1:
                    context = old_emb.detach() # (4,768)
                    context_1 = old_emb.detach() 
                    value_vector = n_embs.detach() # (4,320)
                elif target_bool==1:
                    context = torch.cat((context_1,old_emb),dim=0)
                    value_vector = torch.cat((value_vector,n_embs.detach()),dim=0)
                else:
                    context = torch.cat((context,old_emb),dim=0)
                    context_1 = torch.cat((context_1,old_emb),dim=0)
                    value_vector = torch.cat((value_vector,n_embs.detach()),dim=0)
            # 10个样本的context:(40,768), value_vector:(40,320)
            # context = old_emb.detach()
            # context_vector = context.reshape(context.shape[0], context.shape[1], 1) # (75, 768, 1)
            # context_vector_T = context.reshape(context.shape[0], 1, context.shape[1]) # (75, 1, 768)
            # value_vector = (new_emb @ W_old.T).detach() # (2, 320, 1)
            # value_vector = value_vector.reshape(value_vector.shape[0], value_vector.shape[1],1) # (75, 320, 1)   
            # RKTP = X(K1K1TP + KpKpTP + I)  -> VKTP - WKKTP = X(K1K1TP + KpKpTP + I)     
            for_mat2 = (context.T @ context) # (768,768)累加上去
            for_mat3 = value_vector.T @ context - W_old @ for_mat2
            # R = value_vector - o_embs # (X,320,1) @ (X,1,768)
            # print('R:',R.size())
            # print("o_embs:",o_embs.size())
            # print('context_vector_T',context_vector_T.size())
            # for_mat3 += R.T @ context # (320,10) @ (10,768) = (320,768)
                # print("for_mat2",for_mat2)
                # print("for_mat3",for_mat3)
        if is_nude!=1:
            result1 = lamb * (for_mat2 @ P1 + cache_c @ P1) +  lamda * for_mat1
            result2 = lamb * for_mat3 @ P2
        else:
            result1 = lamb * (for_mat2 @ P_nude + cache_c @ P_nude) +  lamda * for_mat1
            result2 = lamb * for_mat3 @ P_nude
        # 上面的代码对应公式：RKTP = X(K1K1TP + KpKpTP + I)
        # 
        # print("result1.size()",result1.size()) # (768,768)
        # print("result2.size()",result2.size()) # (1280,768)
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
    cache_c += (context_1.T @ context_1)
    print(f'当前模型状态: 将"{str(old_text_)}"编辑为"{str(new_texts)}"，并保留"{str(ret_texts)}"')
    return ldm_stable, cache_c
# 在结束端不投影
def alpha_edit_5(ldm_stable, old_text_, new_text_, layer_target, retain_text_, add=False, layers_to_edit=None, lamb=0.1, erase_scale=0.1, preserve_scale=0.1, with_to_k=True, technique='tensor', P=None, P_nude=None, lamda=10, cache_c = None, P_outs=None):
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
    # print("layers_to_edit", layers_to_edit)      
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
    # print("layers_to_edit", layers_to_edit) 
    ######################## 开始编辑 ###################################
    print(len(projection_matrices))
    for layer_num in range(len(projection_matrices)):
        W_old = projection_matrices[layer_num].weight.detach()  # 使用 .detach() 防止反向传播
        print('W_old:',W_old.size())        
    # 这个是Wv矩阵和Wk矩阵的分界线
    # P = torch.load("/mnt/bn/intern-disk/mlx/users/wrp/uce_nullspace/model_save/nullspace_project/input_1000/null_space_project_subject_1000_3_300.pt")
    P1 = P.clone()
    P2 = P.clone()
    for layer_num in range(len(projection_matrices)):
        # 我们要寻找的是和Wv矩阵对应的Wk矩阵,以及和Wk矩阵对应的Wv矩阵
        # print("layers_to_edit", layers_to_edit) 
        if (layers_to_edit is not None) and (layer_num not in layers_to_edit):
            continue
        # print(f'Editing layer {layer_num}')
        with torch.no_grad():  # 计算图控制
            # 获取当前投影矩阵的权重
            W_old = projection_matrices[layer_num].weight.detach()  # 使用 .detach() 防止反向传播
            print('W_old:',W_old.size())
            # 计算目标概念的嵌入
            values = []
            old_embeddings = []
            new_embeddings = []
            # 每一层初始化
            for_mat1 = torch.eye(projection_matrices[layer_num].weight.shape[1],dtype=torch.float,device = projection_matrices[layer_num].weight.device)
            # for_mat2 = torch.zeros(768,768, device=W_old.device)
            # for_mat3 = torch.zeros(projection_matrices[layer_num].weight.shape[0],768, device=W_old.device)
            # 当输入是一个的时候，比如(Van Gao, art)
            context = None
            value_vector = None
            # 如果输入10个，相当于不保存的情况下顺序输入10个
            # 
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
                # 获取文本的嵌入
                # 获取nudity的cnt，然后将对应的代码做正交处理
                text_embeddings = ldm_stable.text_encoder(text_input.input_ids.to(ldm_stable.device))[0]
                # print("text_embeddings",text_embeddings)
                # 计算有效 token 的索引, 这个其实是有问题的，因为final_token_idx
                final_token_idx = text_input.attention_mask[0].sum().item() - 2
                final_token_idx_new = text_input.attention_mask[1].sum().item() - 2
                farthest = max(final_token_idx_new, final_token_idx)
                
                old_emb = text_embeddings[0][final_token_idx:len(text_embeddings[0])-max(0, farthest-final_token_idx)].detach()
                new_emb = text_embeddings[1][final_token_idx_new:len(text_embeddings[1])-max(0, farthest-final_token_idx_new)].detach()  
                # 获取有效嵌入，去除特殊 token
                # 在这里我们得到的待擦除的概念和新的概念的嵌入，去掉的是SOS token但是EOS token还在的
                # 这里一个token对应多个是不是不太好
                # 加上代码
                # 但是这里艺术家的token数是不确定的，那么可以通过art填充是不是补充一下
                ######## 第一种方法
                idx_old = text_input.input_ids[0].tolist().index(49407)
                idx_new = text_input.input_ids[1].tolist().index(49407)
                farthest = max(idx_old, idx_new)

                old_emb = text_embeddings[0][1:farthest+1].detach()
                new_emb = text_embeddings[1][1:farthest+1].detach()
                # print("####################################old_emb",old_emb.size()) # (X,768)          
                # 新增加的代码如上，和下面的代码是两种不同的选择方式
                ######## 第二种方法
                # old_emb = text_embeddings[0][final_token_idx:len(text_embeddings[0])-max(0, farthest-final_token_idx)].detach()
                # new_emb = text_embeddings[1][final_token_idx_new:len(text_embeddings[1])-max(0, farthest-final_token_idx_new)].detach()

                # idx_old = text_input.input_ids[0].tolist().index(25148)
                # idx_new = text_input.input_ids[1].tolist().index(25148)
                # print([idx_old, idx_new])
                # old_emb = text_embeddings[0] # (77, 768)
                # new_emb = text_embeddings[1] # (77, 768)
                # if cnt==0:
                #     old_emb = old_emb[idx_old:idx_old + 2] # （4，768）
                #     new_emb = new_emb[idx_new:idx_new + 2] # （4，768）
                # else:
                # # old_emb = old_emb[final_token_idx:len(old_emb)-max(0,farthest-final_token_idx)] # （76，768）
                #     old_emb = old_emb[idx_old-2:idx_old + 2] # （4，768）
                #     # new_emb = new_emb[final_token_idx_new:len(new_emb)-max(0,farthest-final_token_idx_new)] # （76，768）
                #     new_emb = new_emb[idx_new-2:idx_new + 2] # （4，768）  
                # # 计算均值,这里应该是不需要计算均值的,直接展开操作
                # old_emb_flat = old_emb.view(text_embeddings.size(0), -1)
                # new_emb_flat = new_emb.view(text_embeddings.size(0), -1)
                # old_meb和new_emb的维度都是(768)
                # X(K1K1TP + KpKpTP + I) = RK1TP
                # if target_bool==1:                    
                #     n_embs = new_emb @ W_old.T @ P_outs[(layer_num+16)%32]  # (4,768) @ (768,320) = (4,320)
                # else:
                n_embs = new_emb @ W_old.T # (4,768) @ (768,320) = (4,320)
                # 将new_emb投影到输出端词的正交空间上
                # new_emb = new_emb @ P_outs[layer_num]  
                if cnt == 0 and target_bool!=1:
                    context = old_emb.detach() # (4,768)
                    context_1 = old_emb.detach() 
                    value_vector = n_embs.detach() # (4,320)
                elif target_bool==1 and cnt==0:
                    context = old_emb.detach() # (4,768)
                    context_1 = old_emb.detach()
                    value_vector = n_embs.detach() # (4,320)
                elif target_bool==1 and cnt!=0:
                    context = torch.cat((context_1,old_emb),dim=0)
                    value_vector = torch.cat((value_vector,n_embs.detach()),dim=0)
                else:
                    context = torch.cat((context,old_emb),dim=0)
                    context_1 = torch.cat((context_1,old_emb),dim=0)
                    value_vector = torch.cat((value_vector,n_embs.detach()),dim=0)
            # 10个样本的context:(40,768), value_vector:(40,320)
            # context = old_emb.detach()
            # context_vector = context.reshape(context.shape[0], context.shape[1], 1) # (75, 768, 1)
            # context_vector_T = context.reshape(context.shape[0], 1, context.shape[1]) # (75, 1, 768)
            # value_vector = (new_emb @ W_old.T).detach() # (2, 320, 1)
            # value_vector = value_vector.reshape(value_vector.shape[0], value_vector.shape[1],1) # (75, 320, 1)   
            # RKTP = X(K1K1TP + KpKpTP + I)  -> VKTP - WKKTP = X(K1K1TP + KpKpTP + I)     
            for_mat2 = (context.T @ context) # (768,768)累加上去
            for_mat3 = value_vector.T @ context - W_old @ for_mat2
            # R = value_vector - o_embs # (X,320,1) @ (X,1,768)
            # print('R:',R.size())
            # print("o_embs:",o_embs.size())
            # print('context_vector_T',context_vector_T.size())
            # for_mat3 += R.T @ context # (320,10) @ (10,768) = (320,768)
                # print("for_mat2",for_mat2)
                # print("for_mat3",for_mat3)
        # if is_nude!=1:
        result1 = lamb * (for_mat2 @ P1 + cache_c @ P1) +  lamda * for_mat1
        result2 = lamb * for_mat3 @ P2
        # else:
        #     result1 = lamb * (for_mat2 @ P_nude + cache_c @ P_nude) +  lamda * for_mat1
        #     result2 = lamb * for_mat3 @ P_nude
        # 上面的代码对应公式：RKTP = X(K1K1TP + KpKpTP + I)
        # 
        # print("result1.size()",result1.size()) # (768,768)
        # print("result2.size()",result2.size()) # (1280,768)
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
    cache_c += (context_1.T @ context_1)
    print(f'当前模型状态: 将"{str(old_text_)}"编辑为"{str(new_texts)}"，并保留"{str(ret_texts)}"')
    return ldm_stable, cache_c
# 加上垂直投影之后
def alpha_edit_6(ldm_stable, old_text_, new_text_, layer_target, retain_text_, add=False, layers_to_edit=None, lamb=0.1, erase_scale=0.1, preserve_scale=0.1, with_to_k=True, technique='tensor', P=None, P_nude=None, lamda=10, cache_c = None, P_outs=None):
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
    # print("layers_to_edit", layers_to_edit)      
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
    # print("layers_to_edit", layers_to_edit) 
    ######################## 开始编辑 ###################################
    print(len(projection_matrices))
    # 这个是Wv矩阵和Wk矩阵的分界线
    # P = torch.load("/mnt/bn/intern-disk/mlx/users/wrp/uce_nullspace/model_save/nullspace_project/input_1000/null_space_project_subject_1000_3_300.pt")
    P1 = P.clone()
    P2 = P.clone()
    for layer_num in range(len(projection_matrices)):
        # 我们要寻找的是和Wv矩阵对应的Wk矩阵,以及和Wk矩阵对应的Wv矩阵
        # print("layers_to_edit", layers_to_edit) 
        if (layers_to_edit is not None) and (layer_num not in layers_to_edit):
            continue
        # print(f'Editing layer {layer_num}')
        with torch.no_grad():  # 计算图控制
            # 获取当前投影矩阵的权重
            W_old = projection_matrices[layer_num].weight.detach()  # 使用 .detach() 防止反向传播
            # 计算目标概念的嵌入
            values = []
            old_embeddings = []
            new_embeddings = []
            # 每一层初始化
            for_mat1 = torch.eye(projection_matrices[layer_num].weight.shape[1],dtype=torch.float,device = projection_matrices[layer_num].weight.device)
            # for_mat2 = torch.zeros(768,768, device=W_old.device)
            # for_mat3 = torch.zeros(projection_matrices[layer_num].weight.shape[0],768, device=W_old.device)
            # 当输入是一个的时候，比如(Van Gao, art)
            context = None
            value_vector = None
            # 如果输入10个，相当于不保存的情况下顺序输入10个
            # 
            is_nude = 0
            for cnt, (old_text, new_text) in enumerate(zip(old_texts, new_texts)):
                target_bool = 0
                if 'nudity' in old_text:
                    target_bool = 1
                    print('yesyesyes')
                    is_nude = 1
                    new_text = str(layer_target[layer_num][0])              
                text_input = ldm_stable.tokenizer(
                    [old_text, new_text],
                    padding="max_length",
                    max_length=ldm_stable.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                # 获取文本的嵌入
                # 获取nudity的cnt，然后将对应的代码做正交处理
                text_embeddings = ldm_stable.text_encoder(text_input.input_ids.to(ldm_stable.device))[0]
                # print("text_embeddings",text_embeddings)
                # 计算有效 token 的索引, 这个其实是有问题的，因为final_token_idx
                final_token_idx = text_input.attention_mask[0].sum().item() - 2
                final_token_idx_new = text_input.attention_mask[1].sum().item() - 2
                farthest = max(final_token_idx_new, final_token_idx)
                
                old_emb = text_embeddings[0][final_token_idx:len(text_embeddings[0])-max(0, farthest-final_token_idx)].detach()
                new_emb = text_embeddings[1][final_token_idx_new:len(text_embeddings[1])-max(0, farthest-final_token_idx_new)].detach()  
                # 获取有效嵌入，去除特殊 token
                # 在这里我们得到的待擦除的概念和新的概念的嵌入，去掉的是SOS token但是EOS token还在的
                # 这里一个token对应多个是不是不太好
                # 加上代码
                # 但是这里艺术家的token数是不确定的，那么可以通过art填充是不是补充一下
                ######## 第一种方法
                idx_old = text_input.input_ids[0].tolist().index(49407)
                idx_new = text_input.input_ids[1].tolist().index(49407)
                farthest = max(idx_old, idx_new)

                old_emb = text_embeddings[0][1:farthest+1].detach()
                new_emb = text_embeddings[1][1:farthest+1].detach()
                # print("####################################old_emb",old_emb.size()) # (X,768)          
                # 新增加的代码如上，和下面的代码是两种不同的选择方式
                ######## 第二种方法
                # old_emb = text_embeddings[0][final_token_idx:len(text_embeddings[0])-max(0, farthest-final_token_idx)].detach()
                # new_emb = text_embeddings[1][final_token_idx_new:len(text_embeddings[1])-max(0, farthest-final_token_idx_new)].detach()

                # idx_old = text_input.input_ids[0].tolist().index(25148)
                # idx_new = text_input.input_ids[1].tolist().index(25148)
                # print([idx_old, idx_new])
                # old_emb = text_embeddings[0] # (77, 768)
                # new_emb = text_embeddings[1] # (77, 768)
                # if cnt==0:
                #     old_emb = old_emb[idx_old:idx_old + 2] # （4，768）
                #     new_emb = new_emb[idx_new:idx_new + 2] # （4，768）
                # else:
                # # old_emb = old_emb[final_token_idx:len(old_emb)-max(0,farthest-final_token_idx)] # （76，768）
                #     old_emb = old_emb[idx_old-2:idx_old + 2] # （4，768）
                #     # new_emb = new_emb[final_token_idx_new:len(new_emb)-max(0,farthest-final_token_idx_new)] # （76，768）
                #     new_emb = new_emb[idx_new-2:idx_new + 2] # （4，768）  
                # # 计算均值,这里应该是不需要计算均值的,直接展开操作
                # old_emb_flat = old_emb.view(text_embeddings.size(0), -1)
                # new_emb_flat = new_emb.view(text_embeddings.size(0), -1)
                # old_meb和new_emb的维度都是(768)
                # X(K1K1TP + KpKpTP + I) = RK1TP
                # if target_bool==1:                    
                #     n_embs = new_emb @ W_old.T @ P_outs[(layer_num+16)%32]  # (4,768) @ (768,320) = (4,320)
                # else:
                n_embs = new_emb @ W_old.T # (4,768) @ (768,320) = (4,320)
                # 将new_emb投影到输出端词的正交空间上
                # new_emb = new_emb @ P_outs[layer_num]  
                if cnt == 0 and target_bool!=1:
                    context = old_emb.detach() # (4,768)
                    context_1 = old_emb.detach() 
                    value_vector = n_embs.detach() # (4,320)
                elif target_bool==1:
                    context = torch.cat((context_1,old_emb),dim=0)
                    value_vector = torch.cat((value_vector,n_embs.detach()),dim=0)
                else:
                    context = torch.cat((context,old_emb),dim=0)
                    context_1 = torch.cat((context_1,old_emb),dim=0)
                    value_vector = torch.cat((value_vector,n_embs.detach()),dim=0)
            # 10个样本的context:(40,768), value_vector:(40,320)
            # context = old_emb.detach()
            # context_vector = context.reshape(context.shape[0], context.shape[1], 1) # (75, 768, 1)
            # context_vector_T = context.reshape(context.shape[0], 1, context.shape[1]) # (75, 1, 768)
            # value_vector = (new_emb @ W_old.T).detach() # (2, 320, 1)
            # value_vector = value_vector.reshape(value_vector.shape[0], value_vector.shape[1],1) # (75, 320, 1)   
            # RKTP = X(K1K1TP + KpKpTP + I)  -> VKTP - WKKTP = X(K1K1TP + KpKpTP + I)     
            for_mat2 = (context.T @ context) # (768,768)累加上去
            for_mat3 = value_vector.T @ context - W_old @ for_mat2
            # R = value_vector - o_embs # (X,320,1) @ (X,1,768)
            # print('R:',R.size())
            # print("o_embs:",o_embs.size())
            # print('context_vector_T',context_vector_T.size())
            # for_mat3 += R.T @ context # (320,10) @ (10,768) = (320,768)
                # print("for_mat2",for_mat2)
                # print("for_mat3",for_mat3)
        # P_outs是输出嵌入的投影矩阵
        if is_nude!=1:
            result1 = lamb * (for_mat2 @ P1 + cache_c @ P1) +  lamda * for_mat1
            result2 = lamb * P_outs[layer_num] @ for_mat3 @ P2
        else:
            result1 = lamb * (for_mat2 @ P_nude + cache_c @ P_nude) +  lamda * for_mat1
            result2 = lamb * for_mat3 @ P_nude
        # 上面的代码对应公式：RKTP = X(K1K1TP + KpKpTP + I)
        # 
        # print("result1.size()",result1.size()) # (768,768)
        # print("result2.size()",result2.size()) # (1280,768)
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
    cache_c += (context_1.T @ context_1)
    print(f'当前模型状态: 将"{str(old_text_)}"编辑为"{str(new_texts)}"，并保留"{str(ret_texts)}"')
    return ldm_stable, cache_c
# 单纯的垂直投影P_outs
def alpha_edit_6(ldm_stable, old_text_, new_text_, layer_target, retain_text_, add=False, layers_to_edit=None, lamb=0.1, erase_scale=0.1, preserve_scale=0.1, with_to_k=True, technique='tensor', P=None, P_nude=None, lamda=10, cache_c = None, P_outs=None):
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
    # print("layers_to_edit", layers_to_edit)      
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
    # print("layers_to_edit", layers_to_edit) 
    ######################## 开始编辑 ###################################
    print(len(projection_matrices))
    # 这个是Wv矩阵和Wk矩阵的分界线
    # P = torch.load("/mnt/bn/intern-disk/mlx/users/wrp/uce_nullspace/model_save/nullspace_project/input_1000/null_space_project_subject_1000_3_300.pt")
    P1 = P.clone()
    P2 = P.clone()
    for layer_num in range(len(projection_matrices)):
        # 我们要寻找的是和Wv矩阵对应的Wk矩阵,以及和Wk矩阵对应的Wv矩阵
        # print("layers_to_edit", layers_to_edit) 
        if (layers_to_edit is not None) and (layer_num not in layers_to_edit):
            continue
        # print(f'Editing layer {layer_num}')
        with torch.no_grad():  # 计算图控制
            # 获取当前投影矩阵的权重
            W_old = projection_matrices[layer_num].weight.detach()  # 使用 .detach() 防止反向传播

            # 计算目标概念的嵌入
            values = []
            old_embeddings = []
            new_embeddings = []
            # 每一层初始化
            for_mat1 = torch.eye(projection_matrices[layer_num].weight.shape[1],dtype=torch.float,device = projection_matrices[layer_num].weight.device)
            # for_mat2 = torch.zeros(768,768, device=W_old.device)
            # for_mat3 = torch.zeros(projection_matrices[layer_num].weight.shape[0],768, device=W_old.device)
            # 当输入是一个的时候，比如(Van Gao, art)
            context = None
            value_vector = None
            # 如果输入10个，相当于不保存的情况下顺序输入10个
            # 
            is_nude = 0
            for cnt, (old_text, new_text) in enumerate(zip(old_texts, new_texts)):
                target_bool = 0
                if 'nudity' in old_text:
                    target_bool = 1
                    print('yesyesyes')
                    is_nude = 1
                    new_text = str(layer_target[layer_num][0])              
                text_input = ldm_stable.tokenizer(
                    [old_text, new_text],
                    padding="max_length",
                    max_length=ldm_stable.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                # 获取文本的嵌入
                # 获取nudity的cnt，然后将对应的代码做正交处理
                text_embeddings = ldm_stable.text_encoder(text_input.input_ids.to(ldm_stable.device))[0]
                # print("text_embeddings",text_embeddings)
                # 计算有效 token 的索引, 这个其实是有问题的，因为final_token_idx
                final_token_idx = text_input.attention_mask[0].sum().item() - 2
                final_token_idx_new = text_input.attention_mask[1].sum().item() - 2
                farthest = max(final_token_idx_new, final_token_idx)
                
                old_emb = text_embeddings[0][final_token_idx:len(text_embeddings[0])-max(0, farthest-final_token_idx)].detach()
                new_emb = text_embeddings[1][final_token_idx_new:len(text_embeddings[1])-max(0, farthest-final_token_idx_new)].detach()  
                # 获取有效嵌入，去除特殊 token
                # 在这里我们得到的待擦除的概念和新的概念的嵌入，去掉的是SOS token但是EOS token还在的
                # 这里一个token对应多个是不是不太好
                # 加上代码
                # 但是这里艺术家的token数是不确定的，那么可以通过art填充是不是补充一下
                ######## 第一种方法
                idx_old = text_input.input_ids[0].tolist().index(49407)
                idx_new = text_input.input_ids[1].tolist().index(49407)
                farthest = max(idx_old, idx_new)

                old_emb = text_embeddings[0][1:farthest+1].detach()
                new_emb = text_embeddings[1][1:farthest+1].detach()
                # print("####################################old_emb",old_emb.size()) # (X,768)          
                # 新增加的代码如上，和下面的代码是两种不同的选择方式
                ######## 第二种方法
                # old_emb = text_embeddings[0][final_token_idx:len(text_embeddings[0])-max(0, farthest-final_token_idx)].detach()
                # new_emb = text_embeddings[1][final_token_idx_new:len(text_embeddings[1])-max(0, farthest-final_token_idx_new)].detach()

                # idx_old = text_input.input_ids[0].tolist().index(25148)
                # idx_new = text_input.input_ids[1].tolist().index(25148)
                # print([idx_old, idx_new])
                # old_emb = text_embeddings[0] # (77, 768)
                # new_emb = text_embeddings[1] # (77, 768)
                # if cnt==0:
                #     old_emb = old_emb[idx_old:idx_old + 2] # （4，768）
                #     new_emb = new_emb[idx_new:idx_new + 2] # （4，768）
                # else:
                # # old_emb = old_emb[final_token_idx:len(old_emb)-max(0,farthest-final_token_idx)] # （76，768）
                #     old_emb = old_emb[idx_old-2:idx_old + 2] # （4，768）
                #     # new_emb = new_emb[final_token_idx_new:len(new_emb)-max(0,farthest-final_token_idx_new)] # （76，768）
                #     new_emb = new_emb[idx_new-2:idx_new + 2] # （4，768）  
                # # 计算均值,这里应该是不需要计算均值的,直接展开操作
                # old_emb_flat = old_emb.view(text_embeddings.size(0), -1)
                # new_emb_flat = new_emb.view(text_embeddings.size(0), -1)
                # old_meb和new_emb的维度都是(768)
                # X(K1K1TP + KpKpTP + I) = RK1TP
                # if target_bool==1:                    
                #     n_embs = new_emb @ W_old.T @ P_outs[(layer_num+16)%32]  # (4,768) @ (768,320) = (4,320)
                # else:
                n_embs = new_emb @ W_old.T # (4,768) @ (768,320) = (4,320)
                # 将new_emb投影到输出端词的正交空间上
                # new_emb = new_emb @ P_outs[layer_num]  
                if cnt == 0 and target_bool!=1:
                    context = old_emb.detach() # (4,768)
                    context_1 = old_emb.detach() 
                    value_vector = n_embs.detach() # (4,320)
                elif target_bool==1:
                    context = torch.cat((context_1,old_emb),dim=0)
                    value_vector = torch.cat((value_vector,n_embs.detach()),dim=0)
                else:
                    context = torch.cat((context,old_emb),dim=0)
                    context_1 = torch.cat((context_1,old_emb),dim=0)
                    value_vector = torch.cat((value_vector,n_embs.detach()),dim=0)
            # 10个样本的context:(40,768), value_vector:(40,320)
            # context = old_emb.detach()
            # context_vector = context.reshape(context.shape[0], context.shape[1], 1) # (75, 768, 1)
            # context_vector_T = context.reshape(context.shape[0], 1, context.shape[1]) # (75, 1, 768)
            # value_vector = (new_emb @ W_old.T).detach() # (2, 320, 1)
            # value_vector = value_vector.reshape(value_vector.shape[0], value_vector.shape[1],1) # (75, 320, 1)   
            # RKTP = X(K1K1TP + KpKpTP + I)  -> VKTP - WKKTP = X(K1K1TP + KpKpTP + I)     
            for_mat2 = (context.T @ context) # (768,768)累加上去
            for_mat3 = value_vector.T @ context - W_old @ for_mat2
            # R = value_vector - o_embs # (X,320,1) @ (X,1,768)
            # print('R:',R.size())
            # print("o_embs:",o_embs.size())
            # print('context_vector_T',context_vector_T.size())
            # for_mat3 += R.T @ context # (320,10) @ (10,768) = (320,768)
                # print("for_mat2",for_mat2)
                # print("for_mat3",for_mat3)
        # P_outs是输出嵌入的投影矩阵
        if is_nude!=1:
            result1 = lamb * (for_mat2 @ P1 + cache_c @ P1) +  lamda * for_mat1
            result2 = lamb * for_mat3 @ P2
        else:
            result1 = lamb * (for_mat2 @ P_nude + cache_c @ P_nude) +  lamda * for_mat1
            result2 = lamb * P_outs[layer_num] @ for_mat3 @ P_nude
        # 上面的代码对应公式：RKTP = X(K1K1TP + KpKpTP + I)
        # 
        # print("result1.size()",result1.size()) # (768,768)
        # print("result2.size()",result2.size()) # (1280,768)
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
    cache_c += (context_1.T @ context_1)
    print(f'当前模型状态: 将"{str(old_text_)}"编辑为"{str(new_texts)}"，并保留"{str(ret_texts)}"')
    return ldm_stable, cache_c
def alpha_edit_7(ldm_stable, old_text_, new_text_, layer_target, retain_text_, add=False, layers_to_edit=None, lamb=0.1, erase_scale=0.1, preserve_scale=0.1, with_to_k=True, technique='tensor', P=None, P_nude=None, lamda=10, cache_c = None, P_outs=None):
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
    # print("layers_to_edit", layers_to_edit)      
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
    # print("layers_to_edit", layers_to_edit) 
    ######################## 开始编辑 ###################################
    print(len(projection_matrices))
    # 这个是Wv矩阵和Wk矩阵的分界线
    # P = torch.load("/mnt/bn/intern-disk/mlx/users/wrp/uce_nullspace/model_save/nullspace_project/input_1000/null_space_project_subject_1000_3_300.pt")
    P1 = P.clone()
    P2 = P.clone()
    for layer_num in range(len(projection_matrices)):
        # 我们要寻找的是和Wv矩阵对应的Wk矩阵,以及和Wk矩阵对应的Wv矩阵
        # print("layers_to_edit", layers_to_edit) 
        if (layers_to_edit is not None) and (layer_num not in layers_to_edit):
            continue
        # print(f'Editing layer {layer_num}')
        with torch.no_grad():  # 计算图控制
            # 获取当前投影矩阵的权重
            W_old = projection_matrices[layer_num].weight.detach()  # 使用 .detach() 防止反向传播

            # 计算目标概念的嵌入
            values = []
            old_embeddings = []
            new_embeddings = []
            # 每一层初始化
            for_mat1 = torch.eye(projection_matrices[layer_num].weight.shape[1],dtype=torch.float,device = projection_matrices[layer_num].weight.device)
            # for_mat2 = torch.zeros(768,768, device=W_old.device)
            # for_mat3 = torch.zeros(projection_matrices[layer_num].weight.shape[0],768, device=W_old.device)
            # 当输入是一个的时候，比如(Van Gao, art)
            context = None
            value_vector = None
            # 如果输入10个，相当于不保存的情况下顺序输入10个
            # 
            is_nude = 0
            for cnt, (old_text, new_text) in enumerate(zip(old_texts, new_texts)):
                target_bool = 0
                if 'nudity' in old_text:
                    target_bool = 1
                    print('yesyesyes')
                    is_nude = 1
                    # new_text = str(layer_target[layer_num][0])              
                text_input = ldm_stable.tokenizer(
                    [old_text, new_text],
                    padding="max_length",
                    max_length=ldm_stable.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                # 获取文本的嵌入
                # 获取nudity的cnt，然后将对应的代码做正交处理
                text_embeddings = ldm_stable.text_encoder(text_input.input_ids.to(ldm_stable.device))[0]
                # print("text_embeddings",text_embeddings)
                # 计算有效 token 的索引, 这个其实是有问题的，因为final_token_idx
                final_token_idx = text_input.attention_mask[0].sum().item() - 2
                final_token_idx_new = text_input.attention_mask[1].sum().item() - 2
                farthest = max(final_token_idx_new, final_token_idx)
                
                old_emb = text_embeddings[0][final_token_idx:len(text_embeddings[0])-max(0, farthest-final_token_idx)].detach()
                new_emb = text_embeddings[1][final_token_idx_new:len(text_embeddings[1])-max(0, farthest-final_token_idx_new)].detach()  
                # 获取有效嵌入，去除特殊 token
                # 在这里我们得到的待擦除的概念和新的概念的嵌入，去掉的是SOS token但是EOS token还在的
                # 这里一个token对应多个是不是不太好
                # 加上代码
                # 但是这里艺术家的token数是不确定的，那么可以通过art填充是不是补充一下
                ######## 第一种方法
                idx_old = text_input.input_ids[0].tolist().index(49407)
                idx_new = text_input.input_ids[1].tolist().index(49407)
                farthest = max(idx_old, idx_new)

                old_emb = text_embeddings[0][1:farthest+1].detach()
                new_emb = text_embeddings[1][1:farthest+1].detach()
                # print("####################################old_emb",old_emb.size()) # (X,768)          
                # 新增加的代码如上，和下面的代码是两种不同的选择方式
                ######## 第二种方法
                # old_emb = text_embeddings[0][final_token_idx:len(text_embeddings[0])-max(0, farthest-final_token_idx)].detach()
                # new_emb = text_embeddings[1][final_token_idx_new:len(text_embeddings[1])-max(0, farthest-final_token_idx_new)].detach()

                # idx_old = text_input.input_ids[0].tolist().index(25148)
                # idx_new = text_input.input_ids[1].tolist().index(25148)
                # print([idx_old, idx_new])
                # old_emb = text_embeddings[0] # (77, 768)
                # new_emb = text_embeddings[1] # (77, 768)
                # if cnt==0:
                #     old_emb = old_emb[idx_old:idx_old + 2] # （4，768）
                #     new_emb = new_emb[idx_new:idx_new + 2] # （4，768）
                # else:
                # # old_emb = old_emb[final_token_idx:len(old_emb)-max(0,farthest-final_token_idx)] # （76，768）
                #     old_emb = old_emb[idx_old-2:idx_old + 2] # （4，768）
                #     # new_emb = new_emb[final_token_idx_new:len(new_emb)-max(0,farthest-final_token_idx_new)] # （76，768）
                #     new_emb = new_emb[idx_new-2:idx_new + 2] # （4，768）  
                # # 计算均值,这里应该是不需要计算均值的,直接展开操作
                # old_emb_flat = old_emb.view(text_embeddings.size(0), -1)
                # new_emb_flat = new_emb.view(text_embeddings.size(0), -1)
                # old_meb和new_emb的维度都是(768)
                # X(K1K1TP + KpKpTP + I) = RK1TP
                # if target_bool==1:                    
                #     n_embs = new_emb @ W_old.T @ P_outs[(layer_num+16)%32]  # (4,768) @ (768,320) = (4,320)
                # else:
                n_embs = new_emb @ W_old.T # (4,768) @ (768,320) = (4,320)
                # 将new_emb投影到输出端词的正交空间上
                # new_emb = new_emb @ P_outs[layer_num]  
                if cnt == 0 and target_bool!=1:
                    context = old_emb.detach() # (4,768)
                    context_1 = old_emb.detach() 
                    value_vector = n_embs.detach() # (4,320)
                elif target_bool==1:
                    context = torch.cat((context_1,old_emb),dim=0)
                    value_vector = torch.cat((value_vector,n_embs.detach()),dim=0)
                else:
                    context = torch.cat((context,old_emb),dim=0)
                    context_1 = torch.cat((context_1,old_emb),dim=0)
                    value_vector = torch.cat((value_vector,n_embs.detach()),dim=0)
            # 10个样本的context:(40,768), value_vector:(40,320)
            # context = old_emb.detach()
            # context_vector = context.reshape(context.shape[0], context.shape[1], 1) # (75, 768, 1)
            # context_vector_T = context.reshape(context.shape[0], 1, context.shape[1]) # (75, 1, 768)
            # value_vector = (new_emb @ W_old.T).detach() # (2, 320, 1)
            # value_vector = value_vector.reshape(value_vector.shape[0], value_vector.shape[1],1) # (75, 320, 1)   
            # RKTP = X(K1K1TP + KpKpTP + I)  -> VKTP - WKKTP = X(K1K1TP + KpKpTP + I)     
            for_mat2 = (context.T @ context) # (768,768)累加上去
            for_mat3 = value_vector.T @ context - W_old @ for_mat2
            # R = value_vector - o_embs # (X,320,1) @ (X,1,768)
            # print('R:',R.size())
            # print("o_embs:",o_embs.size())
            # print('context_vector_T',context_vector_T.size())
            # for_mat3 += R.T @ context # (320,10) @ (10,768) = (320,768)
                # print("for_mat2",for_mat2)
                # print("for_mat3",for_mat3)
        # P_outs是输出嵌入的投影矩阵
        if is_nude!=1:
            result1 = lamb * (for_mat2 @ P1 + cache_c @ P1) +  lamda * for_mat1
            result2 = lamb * P_outs[layer_num] @ for_mat3 @ P2
        else:
            result1 = lamb * (for_mat2 @ P_nude + cache_c @ P_nude) +  lamda * for_mat1
            result2 = lamb * for_mat3 @ P_nude
        # 上面的代码对应公式：RKTP = X(K1K1TP + KpKpTP + I)
        # 
        # print("result1.size()",result1.size()) # (768,768)
        # print("result2.size()",result2.size()) # (1280,768)
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
    cache_c += (context_1.T @ context_1)
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
    parser.add_argument('--num_smallest_singular', help='Number of smallest singular values to consider', type=int, required=False, default=300)
    parser.add_argument('--coco_path', help='coco dataset path', type=str, required=False, default=None)
    parser.add_argument('--lamb', help='lambda value for optimization', type=float, required=False, default=0.1)  # 新增的lamb参数
    parser.add_argument('--lamda', help='Lambda value for scaling the regularization term', type=float, required=False, default=20.0)
    parser.add_argument('--percentage_of_smallest_singular', help='Lambda value for scaling the regularization term', type=float, required=False, default=0.01)

    # parser.add_argument('--projection_path', help='Path to the projection matrix', type=str, required=True)

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
    # 输入artists1000的时候就是提取随机提取1000个艺术家的名称，但是这里是不是可复现的
    if 'artists' in concepts[0]:
        df = pd.read_csv('/mlx_devbox/users/wangruipeng/playground/paper_run/uce_nullspace/unified-concept-editing-main/data/erase_artist/erase_nude_artists_1000.csv')    
        # 提取唯一艺术家的名字
        artists = list(df.Artist.unique())   
        # 得到需要保留的艺术家的数量
        concepts = artists[:999]
        print('concepts:',concepts[:20])
        print('concepts:', 'nudity' in concepts)
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
    # old_texts存储的是1000个artists concepts
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
        new_texts = ["Make sure the gender ratio is evenly distributed and achieve a balanced ratio of male and female analyst " + item for item in old_texts]
        print_text+=f'-towards_bias'
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
    data_path = '/mlx_devbox/users/wangruipeng/playground/paper_run/uce_nullspace/unified-concept-editing-main/data/extracted_subjects_big_only_2000.csv'
    if args.base=='1.4':
        model_version = sd14
    elif args.base=='2.1':
        model_version = sd21
    else:
        model_version = sd14
    print("model_version:", model_version)
    ldm_stable = StableDiffusionPipeline.from_pretrained(model_version).to(device)
    # model_path = f'/mnt/bn/intern-disk/mlx/users/wrp/uce_nullspace/model_save/art_erase/unbias/debias_5_lamda_10_bias_batch_4_lamb_0.1_num_smallest_295.pt'
    # ldm_stable.unet.load_state_dict(torch.load(model_path, map_location=device))
    # ldm_stable.unet.load_state_dict(torch.load(model_path, map_location=device))
    # model_path = '/mnt/bn/intern-disk/mlx/users/wrp/uce_nullspace/model_save/art_erase/500_preserve_1000_erase/'
    # ldm_stable.unet.load_state_dict(torch.load(model_path, map_location=device)) 
    # 获得对应的保护集的投影矩阵   
    if preserve_concepts is None:
        if concept_type == 'art':
            df = pd.read_csv('data/artists1734_prompts.csv')
            retain_texts = list(df.artist.unique())
            old_texts_lower = [text.lower() for text in old_texts]
            preserve_concepts = [text for text in retain_texts if text.lower() not in old_texts_lower]
            if preserve_number is not None:
                print_text+=f'-preserving_{preserve_number}artists'
                preserve_concepts = random.sample(preserve_concepts, preserve_number)
            # 将上面的preserve_concepts存放到csv文件中形成新的data_path
            # preserve_artists_df = pd.DataFrame(preserve_concepts, columns=["Artist"])
            # csv_file_path = '/mlx_devbox/users/wangruipeng/playground/paper_run/uce_nullspace/unified-concept-editing-main/data/preserve_artists/preserve_artists_1000_50edit.csv'
            # preserve_artists_df.to_csv(csv_file_path, index=False)
            # P = get_project_input(ldm_stable, csv_file_path, subject_column='Artist', num_smallest_singular=num_smallest_singular, batch_size=16)
            # torch.save(P,'/mnt/bn/intern-disk/mlx/users/wrp/uce_nullspace/model_save/art_erase/10_erase-artists_output/input.pt')
            # coco_path需要把preserve_artists_path里面标题Artist下的艺术家加入coco数据集中,coco_path路径下的文件
            # preserve_artists_path里面标题Artist下的艺术家
            # 读取艺术家信息
            artists_df = pd.read_csv(preserve_artists_path)

            # 读取另一个CSV文件，它只包含一列，比如'subject'
            subjects_df = pd.read_csv(coco_path)

            subjects_df.columns = ['Artist']

            # 使用concat函数将两个DataFrame拼接在一起
            # 注意：这里我们假设两个CSV文件中的行数相同
            merged_df = pd.concat([artists_df, subjects_df], ignore_index=True)
            with open('/mlx_devbox/users/wangruipeng/playground/paper_run/uce_nullspace/unified-concept-editing-main/train_art/length_of_merged_df.txt', 'w') as file:
                file.write(f"Length of merged DataFrame: {len(merged_df)}\n")
            # 保存合并后的DataFrame到新的CSV文件
            merged_df.to_csv(preserve_artists_path, index=False)

            P = get_project_input_3(ldm_stable, preserve_artists_path, subject_column='Artist', num_smallest_singular=num_smallest_singular, batch_size=16)
            
        elif concept_type == 'coco':
            # coco_path
            layer_target = []
            # layer_target = find_most_diff(ldm_stable, coco_path)
            layer_target = get_new_concept()
            print('len(layer_target):',len(layer_target))
            print('done!!!')
            # P_outs = get_project_output(ldm_stable,percentage_of_smallest_singular=percentage_of_smallest_singular, batch_size=1)
            P = get_project_input_3(ldm_stable, coco_path, subject_column='Artist', num_smallest_singular=num_smallest_singular, batch_size=16)
            # P_nude = get_project_input_3(ldm_stable, coco_path, subject_column='Artist', num_smallest_singular=nude_erase_num_smallest_singular, batch_size=16)
            # P_outs = K_means_output(ldm_stable,num_smallest_singular=150)
            # P_outs = get_project_output_expand(ldm_stable, 'nudity', percentage_of_smallest_singular=percentage_of_smallest_singular, batch_size=1)
            # P_outs = 
            # P_outs = get_project_output(ldm_stable,percentage_of_smallest_singular=percentage_of_smallest_singular, batch_size=1)
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
    # 模型ldm_stable
    # old_texts
    # new_texts
    # get_similar_token(ldm_stable, path='/mlx_devbox/users/wangruipeng/playground/paper_run/uce_nullspace/unified-concept-editing-main/data/nude_NOUN_VERB_ADJ.csv', layers_to_edit=None, lamb=0.1, with_to_k=True)
    # 输入为1的时候没问题，输入为10的时候第一次编辑就有问题
    # 编辑到
    with open('/mlx/mlx_devbox/users/wangruipeng/playground/paper_run/uce_nullspace/unified-concept-editing-main/train_nude_new/SD21_alpha.txt', 'a') as time_log:
        batch_size = 100
        cache_c = torch.zeros(768, 1024, device=device)
        for i in tqdm(range(0, len(old_texts), batch_size)): # 方便修改！！！！！！！！！！！！！！！！！
            start_time = time.time()
            old_text = old_texts[i:i + batch_size]
            new_text = new_texts[i:i + batch_size]
            # 使用批次索引更新模型保存路径
            batch_index = i // batch_size  # 计算批次索引
            model_save_path_with_index = args.model_save_path.replace('.pt', f'_batch_{batch_index}_lamb_{lamb}_num_smallest_{args.num_smallest_singular}_nudity.pt')  # 添加lamb到文件名
            concepts_save_path_with_index = args.concepts_save_path.replace('.txt', f'_batch_{batch_index}_lamb_{lamb}_num_smallest_{args.num_smallest_singular}_nudity.txt')  # 添加lamb到文件名
            
            ldm_stable, cache_c = alpha_edit_5(ldm_stable=ldm_stable, old_text_=old_text, new_text_=new_text, layer_target = layer_target, add=False, retain_text_=retain_texts, lamb=lamb, erase_scale=erase_scale, preserve_scale=preserve_scale, technique=technique, lamda=lamda, P=P, P_nude=None, cache_c=cache_c)
            
            # 保存模型，文件名包含批次索引
            torch.save(ldm_stable.unet.state_dict(), model_save_path_with_index)
            # 保存概念，文件名包含批次索引
            with open(concepts_save_path_with_index, 'w') as fp:
                json.dump(concepts, fp)
            end_time = time.time()
            elapsed_time = end_time - start_time
            time_log.write(f"Batch {batch_index} processed in {elapsed_time:.2f} seconds.\n")
            time_log.flush()  # 确保立即写入文件