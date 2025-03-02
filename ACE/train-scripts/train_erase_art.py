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
import pandas as pd
from tqdm import tqdm
import pdb
# 接受一个图像列表，并将它们排列成一个网格布局。网格具有指定的行数 (num_rows)，并且图像之间有一定的偏移量 (offset_ratio)
# images是输入，可能是单个图像，也可能是多个图像，也可能是一个四维的numpy数组
# 功能：在紧凑的网格格式中可视化一批图像
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
# 获得协方差矩阵，获取某一层的协方差函数，但是在stable diffusion里面的每个layer的协方差矩阵是相同的
# 输入数据 .csv文件：/mlx_devbox/users/wangruipeng/playground/paper_run/uce_nullspace/unified-concept-editing-main/data/coco_30k.csv
# coco数据集作为保留的概念，作为我们的保留概念
def get_project(ldm_stable, data_path, num_smallest_singular=20, batch_size=16):
    # 读取数据
    data = pd.read_csv(data_path)
    # 初始化一个张量来存储嵌入
    total_embeddings = None
    print(len(data))

    for i in tqdm(range(0, len(data['subject']), batch_size)):  # 使用 tqdm 监控进度
        batch_prompts = data['subject'][i:i + batch_size]
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

        text_embeddings = text_embeddings[1:].sum(dim=1)  # 将形状转换为 (N, 768)
        
        # 初始化总嵌入张量
        if total_embeddings is None:
            total_embeddings = text_embeddings
        else:
            total_embeddings = torch.cat((total_embeddings, text_embeddings), dim=0)

        # 释放内存
        del text_input, text_embeddings
        torch.cuda.empty_cache()  # 清理未使用的缓存

    print("Total embeddings size:", total_embeddings.size())
    
    # 计算转置乘积
    product = torch.mm(total_embeddings.T, total_embeddings)
    print("Product size:", product.size())
    
    # 进行 SVD 分解
    U, S, _ = torch.linalg.svd(product, full_matrices=False)
    print("Singular values size:", S.size())
    # 打印一下最小的50个奇异值看一下数据分布
    print(f"Smallest 50 singular values: {S[-200:]}")
    # 选择最小的 N 个奇异值的索引
    smallest_indices = torch.topk(S, num_smallest_singular, largest=False).indices
    smallest_indices = smallest_indices.sort().values
    print(f"Indices of the smallest {num_smallest_singular} singular values: {smallest_indices}")
    # 计算投影矩阵
    projection_matrix = U[:, smallest_indices] @ U[:, smallest_indices].T
    print("Projection matrix size:", projection_matrix.size())
    
    return projection_matrix
# 编辑模型，首先这个P矩阵的形状还是(768,768)
# 相比较之前而言，这里面没有了erase_scale和preserver_scale的限制。
# cache_c就是一个编辑过的知识的协方差
def edit_model(ldm_stable, old_text_, new_text_, retain_text_, add=False, layers_to_edit=None, lamb=0.1, erase_scale=0.1, preserve_scale=0.1, with_to_k=True, technique='tensor', P=None, lamda=10, cache_c = None):
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
    ######################## 持续编辑 ###################################
    artist_path = '/mlx_devbox/users/wangruipeng/playground/paper_run/uce_nullspace/unified-concept-editing-main/data/artist'
    artist_file = glob.glob(artist_path + '/*')
    
    for layer_num in range(len(projection_matrices)):
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
            for old_text, new_text in zip(old_texts, new_texts):
                text_input = ldm_stable.tokenizer(
                    [old_text, new_text],
                    padding="max_length",
                    max_length=ldm_stable.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                # 获取文本的嵌入
                text_embeddings = ldm_stable.text_encoder(text_input.input_ids.to(ldm_stable.device))[0]
                # print("text_embeddings",text_embeddings.size())
                # 计算有效 token 的索引
                final_token_idx = text_input.attention_mask[0].sum().item() - 2
                final_token_idx_new = text_input.attention_mask[1].sum().item() - 2
                farthest = max(final_token_idx_new, final_token_idx)
                # print("final_token_idx::::",final_token_idx,final_token_idx_new,farthest)
                # 获取有效嵌入，去除特殊 token
                # 在这里我们得到的待擦除的概念和新的概念的嵌入，去掉的是SOS token但是EOS token还在的
                old_emb = text_embeddings[0][final_token_idx:len(text_embeddings[0])-max(0, farthest-final_token_idx)].detach()
                new_emb = text_embeddings[1][final_token_idx_new:len(text_embeddings[1])-max(0, farthest-final_token_idx_new)].detach()
                # print("old_emb_1", old_emb.size())
                # # 计算均值,这里应该是不需要计算均值的,直接展开操作
                # old_emb_flat = old_emb.view(text_embeddings.size(0), -1)
                # new_emb_flat = new_emb.view(text_embeddings.size(0), -1)
                # old_meb和new_emb的维度都是(768)
                old_emb = old_emb.sum(dim=0)
                new_emb = new_emb.sum(dim=0)
                old_embeddings.append(old_emb)
                # print("old_emb_2", old_emb.size())
                # 通过当前层的投影矩阵计算嵌入
                o_embs = projection_matrices[layer_num](old_emb).detach()
                new_embs = projection_matrices[layer_num](new_emb).detach()
                # new_emb_proj = (u * new_embs).sum()
                new_embeddings.append(new_embs)
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
            P = torch.load("/mlx_devbox/users/wangruipeng/playground/paper_run/uce_nullspace/unified-concept-editing-main/models/null_space_project_subject_5000.pt")
            # print('P.size',P.size())
            # print('old_emb.unsqueeze(0) @ old_emb.unsqueeze(0).T',(old_emb.unsqueeze(0) @ old_emb.unsqueeze(0).T).size())
            Kp1 = P  
            Kp2 = P
            # print('Kp2.size',Kp2.size())
            # 计算 R, (3,768)
            R = torch.stack(values)- old_embs @ W_old.T # 计算目标概念与旧权重的差
            # print("W_old.size", W_old.size())
            # print('old_emb.size', old_emb.size())
            # print("new_emb.size", new_emb.size())
            # print('old_embs.size', o_embs.size())
            # print("new_embs.size", new_embs.size())
            # print("R.size()",R.size()) # 320
            # 使用线性代数求解更新矩阵,矩阵的输入维度是768，输出维度是320,(768,768)*()
            
            result1 =  Kp1 @ (old_embs.T @ old_embs + cache_c.cuda()) + 10 * torch.eye((old_embs.T).shape[0], dtype=torch.float, device="cuda")
            print("Kp2.size", Kp2.size())
            # (768,768) @ (768,3) @ (3,320)
            result2 =  Kp2 @ (old_embs.T) @ R
            upd_matrix = torch.linalg.solve(
                result1, 
                result2
            )
            # 权重：（320，320）*（320，1）*（1，320）
            # 更新投影矩阵权重
            file_path = f"/mlx_devbox/users/wangruipeng/playground/paper_run/uce_nullspace/unified-concept-editing-main/models/updata_matrics_{layer_num}.pt"
            print("upd_matrix.size",upd_matrix.size())
            projection_matrices[layer_num].weight = torch.nn.Parameter(W_old + 0.1 * upd_matrix.T)
            torch.save(upd_matrix, file_path)
            # 在这里构建一下投影矩阵
    #         P = torch.zeros((len(hparams.layers), W_out.shape[0], W_out.shape[0]), device="cpu")
    # for i, layer in enumerate(len(projection_matrices)):
    #     P[i,:,:] = get_project(model,tok,layer,hparams)
    # torch.save(P, "null_space_project.pt") 
    # P = torch.zeros(W_old.shape[1], W_old.shape[1]) # (768,768)
    # P = get_project(ldm_stable,'/mlx_devbox/users/wangruipeng/playground/paper_run/uce_nullspace/unified-concept-editing-main/data/extracted_subjects_big_only_5000.csv')
    # # print("P.size",P.size())
    # torch.save(P, "/mlx_devbox/users/wangruipeng/playground/paper_run/uce_nullspace/unified-concept-editing-main/models/null_space_project_subject_5000.pt") 
    print(f'当前模型状态: 将"{str(old_text_)}"编辑为"{str(new_texts)}"，并保留"{str(ret_texts)}"')
    cache_c += old_embs.cpu().T @ old_embs.cpu()
    return ldm_stable
# 第二部通过保留的概念来训练模型



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

    args = parser.parse_args()
    technique = args.technique # replace
    device = f'cuda:{args.device}' # cuda:0
    preserve_scale = args.preserve_scale # 0.1
    erase_scale = args.erase_scale # 0.1
    add_prompts = args.add_prompts # False
    guided_concepts = args.guided_concepts # art
    preserve_concepts = args.preserve_concepts # None
    preserve_number = args.preserve_number # None
    concepts = args.concepts.split(',') # [Kelly Mckernan, Sarah Anderson]
    concepts_args = [con.strip() for con in concepts] # ['Kelly Mckernan', 'Sarah Anderson']
    concept_type = args.concept_type # 如果是art的话会起到什么作用
    # print_text=''
    # for txt in concepts:
    #     print_text += txt.lower()+'_'
    # print_text  = print_text[:-1]
    # PROMPT CLEANING
    if concepts[0] == 'allartist':
        concepts = ["Kelly Mckernan", "Thomas Kinkade", "Pablo Picasso", "Tyler Edlin", "Kilian Eng"]
    if concepts[0] == 'i2g':
        concepts = ['hate', 'harassment', 'violence', 'suffering', 'humiliation', 'harm', 'suicide', 'sexual', 'nudity', 'bodily fluids', 'blood']
    if concepts[0] == '10artists':
        concepts = ["Asger Jorn", "Eric Fischl", "Johannes Vermeer", "Apollinary Vasnetsov", "Naoki Urasawa", "Nicolas Mignard", "John Whitcomb", "John Constable", "Warwick Globe", "Albert Marquet"]
    if concepts[0] == 'imagenette':
        concepts = ['Cassette Player', 'Chain Saw', 'Church', 'Gas Pump', 'Tench', 'Garbage Truck', 'English Springer', 'Golf Ball', 'Parachute', 'French Horn']
    # 读取cvs文件，将artist列中的内容读取出来并去重
    # 去除artist的风格，需要我们将构建一个以删除的artist概念，每次从总共的artist数据集中去重这些已经被删除的artist
    # 按照1，5，10，50，100，200，300，400，500，600，700，800，900，1000这样编辑下去，100开始每次加编辑的数量是100
    if 'artists' in concepts[0]:
        df = pd.read_csv('data/artists1734_prompts.csv')
        artists = list(df.artist.unique())
        number = int(concepts[0].replace('artists', '')) # 一般是Xartists这样的，X是数字
        concepts = random.sample(artists,number) 
    # 存储的要持续编辑的文件夹和对应的csv文件，实现持续编辑
    if 'continuous' in concepts[0]:
        artist_path = '/mlx_devbox/users/wangruipeng/playground/paper_run/uce_nullspace/unified-concept-editing-main/data/artist'
        artist_files = glob.glob(artist_path + '/*')
    # 先初始化SD模型，因为后面的修改都是在上次修改的基础上进行的，所以在循环开始ian初始化SD模型
    sd14="/mnt/bn/intern-disk/mlx/users/wrp/ecu-nullspace/SDv1.4"
    sd21='/mnt/bn/intern-disk/mlx/users/wrp/ecu-nullspace/SDv2.1'
    if args.base=='1.4':
        model_version = sd14
    elif args.base=='2.1':
        model_version = sd21
    else:
        model_version = sd14
    ldm_stable = StableDiffusionPipeline.from_pretrained(model_version).to(device)
    print("Loaded model successfully.")
    cache_c = torch.zeros(ldm_stable.unet.config.cross_attention_dim, ldm_stable.unet.config.cross_attention_dim)
    #### 开始持续编辑
    for index, artist_file in enumerate(artist_files):
        print_text=''
        for txt in concepts_args:
            print_text += txt.lower()+'_'
        print_text  = print_text[:-1]
        old_texts = []
        df = pd.read_csv(artist_file)
        # 提取artist列下的单词
        concepts = [con.strip() for con in df['artist'].tolist()]
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
        elif index == 0:
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
        else:
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
        
        # 这个在null space这里是不是不应该被保护
        # if preserve_concepts is None:
        #     if concept_type == 'art':
        #         df = pd.read_csv('data/artists1734_prompts.csv')

        #         retain_texts = list(df.artist.unique())
        #         old_texts_lower = [text.lower() for text in old_texts]
        #         preserve_concepts = [text for text in retain_texts if text.lower() not in old_texts_lower]
        #         if preserve_number is not None:
        #             print_text+=f'-preserving_{preserve_number}artists'
        #             preserve_concepts = random.sample(preserve_concepts, preserve_number)
        #     else:
        #         preserve_concepts = []
        # retain_texts = ['']+preserve_concepts   
        retain_texts = ['']
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
        # 先随机初始化一个P
        ldm_stable = edit_model(ldm_stable= ldm_stable, old_text_= old_texts, new_text_=new_texts, add=False, retain_text_= retain_texts, lamb=0.1, erase_scale = erase_scale, preserve_scale = preserve_scale,  technique=technique, cache_c=cache_c)
        # 在这里构建输入知识的零空间的投影矩阵
        # 因为文件路径太长导致的错误
        torch.save(ldm_stable.unet.state_dict(), f'/mnt/bn/intern-disk/mlx/users/wrp/uce_nullspace/model_save/erase_art/erased-{print_text}_nullspace_0.1_{index}.pt') # 这里需要与编辑的次数有关，P与编辑的次数无关。但是代码需要重新编写了，因为需要保留知识
        with open(f'/mlx_devbox/users/wangruipeng/playground/paper_run/uce_nullspace/unified-concept-editing-main/data/info_copy/erased-{print_text}_nullspace_P_0.1.txt', 'w') as fp:
            json.dump(concepts,fp)
###### 举例
# old_texts
# [
#     'Kelly Mckernan', 
#     'painting by Kelly Mckernan', 
#     'art by Kelly Mckernan', 
#     'artwork by Kelly Mckernan', 
#     'picture by Kelly Mckernan', 
#     'style of Kelly Mckernan',
#     'Sarah Anderson', 
#     'painting by Sarah Anderson', 
#     'art by Sarah Anderson', 
#     'artwork by Sarah Anderson', 
#     'picture by Sarah Anderson', 
#     'style of Sarah Anderson'
# ]
# concepts_
# [
#     'Kelly Mckernan', 'Kelly Mckernan', 'Kelly Mckernan', 'Kelly Mckernan', 'Kelly Mckernan', 'Kelly Mckernan', 
#     'Sarah Anderson', 'Sarah Anderson', 'Sarah Anderson', 'Sarah Anderson', 'Sarah Anderson', 'Sarah Anderson'
# ]
# new_texts
# [
#     ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '
# ]