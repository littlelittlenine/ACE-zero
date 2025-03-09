# 去除多个概念的prompts, 上面的职业
# 擦除, alphaedit编辑实现偏见消除
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
clip_model = CLIPModel.from_pretrained("/mnt/bn/intern-disk/mlx/users/wrp/ecu-nullspace/clip-vit")
clip_processor = CLIPProcessor.from_pretrained("/mnt/bn/intern-disk/mlx/users/wrp/ecu-nullspace/clip-vit")

with_to_k = False
with_augs = True
train_func = "train_closed_form"

### load model
LOW_RESOURCE = False
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77
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
    print(old_text_, new_text_)
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
    print([old_texts, new_texts])
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
                # idx_old = text_input.input_ids[0].tolist().index(49407)
                # idx_new = text_input.input_ids[1].tolist().index(49407)
                # print([idx_old, idx_new])
                # old_emb = text_embeddings[0] # (77, 768)
                # new_emb = text_embeddings[1] # (77, 768)
   
                # old_emb = old_emb[idx_old:idx_old+1] # （4，768）
                # new_emb = new_emb[idx_new:idx_new+1] # （4，768）
                final_token_idx = text_input.attention_mask[0].sum().item() - 2
                final_token_idx_new = text_input.attention_mask[1].sum().item() - 2
                farthest = max(final_token_idx_new, final_token_idx)
                # 获取有效嵌入，去除特殊 token
                # 在这里我们得到的待擦除的概念和新的概念的嵌入，去掉的是SOS token但是EOS token还在的
                old_emb = text_embeddings[0][final_token_idx:len(text_embeddings[0])-max(0, farthest-final_token_idx)].detach()
                new_emb = text_embeddings[1][final_token_idx_new:len(text_embeddings[1])-max(0, farthest-final_token_idx_new)].detach()

                context = old_emb.detach()
                context_vector = context.reshape(context.shape[0], context.shape[1], 1) # (1, 768, 1)
                context_vector_T = context.reshape(context.shape[0], 1, context.shape[1]) # (1, 1, 768)
                value_vector = (new_emb @ W_old.T).detach() # (2, 320, 1)
                # value_vector = value_vector @ P_out.T
                value_vector = value_vector.reshape(value_vector.shape[0], value_vector.shape[1],1) # (2, 320)           
                for_mat2 += (context_vector @ context_vector_T).sum(dim=0) # (768,768)累加上去
                o_embs = context @ W_old.T

                R = value_vector - o_embs.unsqueeze(-1) # (X,320,1) @ (X,1,768)

                for_mat3 += (R @ context_vector_T).sum(dim=0) # (768,768)累加上去 # (X,320,1) @ (X,1,768)

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

            projection_matrices[layer_num].weight = torch.nn.Parameter(W_old + upd_matrix.T)
    cache_c += for_mat2
    print(f'当前模型状态: 将"{str(old_text_)}"编辑为"{str(new_texts)}"，并保留"{str(ret_texts)}"')
    return ldm_stable, cache_c
# 获取需要的
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
            # print("text_embeddings:", text_embeddings.size())
            text_embeddings = text_embeddings.detach()  
            # text_embeddings = text_embeddings[:,1:idx+1:,:]
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
def get_project_output(ldm_stable, preserve_concepts, percentage_of_smallest_singular=0.01, batch_size=16, with_to_k=True):
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

    print(len(preserve_concepts))
    P_outs = []
    print("len(projection_matrices)", len(projection_matrices))
    
    for layer_num in range(len(projection_matrices)):

        total_embeddings = None

        for i in tqdm(range(0, len(preserve_concepts), batch_size)):
            batch_prompts = preserve_concepts[i:i + batch_size]
            cleaned_prompts = [prompt.replace('“', '').replace('”', '').strip() for prompt in batch_prompts]
            text_input = ldm_stable.tokenizer(
                cleaned_prompts,
                padding="max_length",
                max_length=ldm_stable.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            with torch.no_grad():  
                idx = text_input.input_ids[0].tolist().index(49407) 
                text_embeddings = ldm_stable.text_encoder(text_input.input_ids.to(ldm_stable.device))[0]
                output_embeddings = projection_matrices[layer_num](text_embeddings).detach()
                output_embeddings = output_embeddings[:, 1:idx, :]
            output_embeddings = output_embeddings.reshape(-1, output_embeddings.size(-1))

            if total_embeddings is None:
                total_embeddings = output_embeddings
            else:
                total_embeddings = torch.cat((total_embeddings, output_embeddings), dim=0)
            del text_input, text_embeddings, output_embeddings
            torch.cuda.empty_cache()  
        
        print("Total embeddings size:", total_embeddings.size())
        
        product = torch.mm(total_embeddings.T, total_embeddings)
        print("Product size:", product.size())
        
        U, S, _ = torch.linalg.svd(product, full_matrices=False)
        print("Singular values size:", S.size())

        total_singular_values = S.size(0)
        num_smallest_singular = max(1, int(total_singular_values * percentage_of_smallest_singular))  # 确保至少选择1个

        smallest_indices = torch.topk(S, num_smallest_singular, largest=False).indices
        smallest_indices = smallest_indices.sort().values
        
        projection_matrix = U[:, smallest_indices] @ U[:, smallest_indices].T
        print("Projection matrix size:", projection_matrix.size())
        
        P_outs.append(projection_matrix)
    
    return P_outs
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
    concepts = args.concepts.split(',') 
    concepts = [con.strip() for con in concepts] 
    concept_type = args.concept_type
    # projection_path = args.projection_path
    model_save_path = args.model_save_path
    concepts_save_path = args.concepts_save_path
    # P = torch.load(args.projection_path)
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
        output_concepts = ['male','female', 'man', 'woman', 'boy', 'girl']
        concepts = ['sheriff', 'librarian', 'teacher','analyst', 'doctor']

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
            old_texts.append(prompt.format(concept=concept)) 
        length = 1 + len(additional_prompts)
        concepts_.extend([concept]*length) 
    print('old_texts:',old_texts)
    if guided_concepts is None:
        new_texts = [' ' for _ in old_texts]
        print_text+=f'-towards_uncond'
    elif guided_concepts == 'bias':
        new_texts = []
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
            new_texts = reduce(operator.concat, new_texts) 
            print_text+=f'-towards'
            for t in new_texts:
                if t not in print_text:
                    print_text+=f'-{t}'      
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
            preserve_concepts = []
        else:
            preserve_concepts = []
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
    log_file_path = '/mlx_devbox/users/wangruipeng/playground/paper_run/uce_nullspace/unified-concept-editing-main/train_unbias/log_file_100.txt'  
    with open(log_file_path, 'a') as log_file:
        for i in tqdm(range(0, len(old_texts), batch_size)):
            old_text = old_texts[i:i + batch_size]
            new_text = new_texts[i:i + batch_size]
            current_model_state = ldm_stable.unet.state_dict()
            counts_save = []
            generate_images_1(ldm_stable, concept=old_text[0], num_samples=20)
            img_path = '/mlx_devbox/users/wangruipeng/playground/paper_run/uce_nullspace/unified-concept-editing-main/data/bias/bias/bias3/class'
            count0 = CLIP_classification(im_path=img_path, attributes='a man,a woman', prompts_path='/mlx/mlx_devbox/users/wangruipeng/playground/paper_run/uce_nullspace/unified-concept-editing-main/data/profession3.csv', save_path=None, from_case=0, till_case=10000)
            if count0 >= 11:
                print([old_text[0], concepts[i//6]])
                new_text = [old_text[0].replace(concepts[i//6], f'female {concepts[i//6]}')]
            elif count0 <= 9:
                new_text = [old_text[0].replace(concepts[i//6], f'male {concepts[i//6]}')]
            print("old_text:", old_text[0])
            print("new_text:", new_text)    
            log_file.write(f"old_text new_text: {[old_text[0], new_text]}\n")  
            log_file.flush()
            for num_smallest_singular in range(10, 500, 10):
                P = get_project_input_3(ldm_stable, coco_path, num_smallest_singular=num_smallest_singular, batch_size=16)
                batch_index = i // batch_size
                ldm_stable, cache_c = alpha_edit_2(ldm_stable=ldm_stable, old_text_=old_text, new_text_=new_text, add=False, retain_text_=retain_texts, lamb=lamb, erase_scale=erase_scale, preserve_scale=preserve_scale, technique=technique, lamda=lamda, P=P, cache_c=cache_c)
                generate_images_1(ldm_stable, concept=old_text[0], num_samples=20)
                counts = CLIP_classification(im_path=img_path, attributes='a man,a woman', prompts_path='/mlx/mlx_devbox/users/wangruipeng/playground/paper_run/uce_nullspace/unified-concept-editing-main/data/profession3.csv', save_path=None, from_case=0, till_case=10000)
                counts_save.append(counts)
                
                min_difference = min(abs(10 - x) for x in counts_save)
                if counts == 10:
                    model_save_path_with_index = args.model_save_path.replace('.pt', f'_bias_batch_{batch_index}_lamb_{lamb}_num_smallest_{num_smallest_singular}.pt')
                    concepts_save_path_with_index = args.concepts_save_path.replace('.txt', f'_bias_batch_{batch_index}_lamb_{lamb}_num_smallest_{num_smallest_singular}.txt')
                    if (i+1)%6==0:
                        torch.save(ldm_stable.unet.state_dict(), model_save_path_with_index)
                        with open(concepts_save_path_with_index, 'w') as fp:
                            json.dump(concepts, fp)
                    last_saved_model_state = ldm_stable.unet.state_dict()
                    ldm_stable.unet.load_state_dict(last_saved_model_state)
                    cache_c_last = cache_c   
                    torch.cuda.empty_cache()
                    log_file.write("yes, find it!\n")  
                    log_file.flush()
                    break  
                elif abs(10 - counts) <= min_difference:
                    model_save_path_with_index = args.model_save_path.replace('.pt', f'_bias_batch_{batch_index}_lamb_{lamb}_num_smallest_{num_smallest_singular}.pt')
                    concepts_save_path_with_index = args.concepts_save_path.replace('.txt', f'_bias_batch_{batch_index}_lamb_{lamb}_num_smallest_{num_smallest_singular}.txt')
                    get_saved_model_state = ldm_stable.unet.state_dict()
                    ldm_stable.unet.load_state_dict(last_saved_model_state)
                    cache_c = cache_c_last
                    # log_file.write("find a better one\n")
                    log_file.write(f'counts: {counts}\n')  
                    log_file.flush()
                    torch.cuda.empty_cache()
                else:
                    ldm_stable.unet.load_state_dict(last_saved_model_state)   
                    cache_c = cache_c_last
                if num_smallest_singular == 490 or (count0>10 and counts<=9) or (count0<10 and counts>=11):
                    rate = min(abs(10 - x) for x in counts_save)
                    log_file.write(f'rate: {rate}\n')  
                    log_file.flush()
                    ldm_stable.unet.load_state_dict(get_saved_model_state)
                    generate_images_1(ldm_stable, concept=old_text[0], num_samples=20)
                    count1 = CLIP_classification(im_path=img_path, attributes='a man,a woman', prompts_path='/mlx/mlx_devbox/users/wangruipeng/playground/paper_run/uce_nullspace/unified-concept-editing-main/data/profession3.csv', save_path=None, from_case=0, till_case=10000)
                    print("count1:", count1)
                    last_saved_model_state = ldm_stable.unet.state_dict()
                    cache_c_last = cache_c 
                    torch.cuda.empty_cache()
                    log_file.write("only can do this\n")  # 记录日志
                    log_file.flush()
                    if (i+1)%6==0:
                        torch.save(ldm_stable.unet.state_dict(), model_save_path_with_index)
                        with open(concepts_save_path_with_index, 'w') as fp:
                            json.dump(concepts, fp)
                        break
            log_file.write(f'counts_save: {counts_save}\n')
            log_file.flush()
