# CLIP（Contrastive Language-Image Pretraining）模型计算图像与其对应文本描述（提示）之间的相似度得分
from PIL import Image
import requests
import os, glob
import pandas as pd
import numpy as np
import re
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("/mnt/bn/intern-disk/mlx/users/wrp/ecu-nullspace/clip-vit")
processor = CLIPProcessor.from_pretrained("/mnt/bn/intern-disk/mlx/users/wrp/ecu-nullspace/clip-vit")
# 这个排序函数的特别之处在于它能够按照人类直觉的顺序（也称为“自然排序”）对字符串进行排序，即数字部分会被当作数值来比较，而不是字符顺序。
def sorted_nicely( l ):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)


path = '/mlx_devbox/users/wangruipeng/playground/paper_run/uce_nullspace/unified-concept-editing-main/data/results/intro_coco'  # 不同模型生成的图像文件夹
model_names = os.listdir(path)                 # 获取文件夹下所有文件夹名称
model_names = [m for m in model_names] # 不包含那些包含.的文件。应该是.pt这种
csv_path = '/mlx_devbox/users/wangruipeng/playground/paper_run/uce_nullspace/unified-concept-editing-main/data/concat_data/coco250_prompts.csv' # 是包含图像描述的 CSV 文件的路径。
save_path = '/mlx_devbox/users/wangruipeng/playground/paper_run/uce_nullspace/unified-concept-editing-main/data/output_score/clip_score' # 保存的路径
print("model_names",model_names)
model_names.sort() # 排序
if 'original' in model_names:
    model_names.remove('original')
print("model_names",model_names)
# model_names = [m for m in model_names if 'single50_50' in m or ('10a' in m or '50a' in m)]
print("model_names",model_names)
#model_names = ['original'] + model_names
#model_names = [m for m in model_names if 'i2p' in m]
for model_name in model_names:
    print(model_name)
    # csv_path = f'/share/u/rohit/erase-closed/data/coco_30k.csv'
    im_folder = os.path.join(path, model_name)
    df = pd.read_csv(csv_path)
    images = os.listdir(im_folder)
    images = sorted_nicely(images)
    ratios = {}
    df['clip'] = np.nan
    print("yes")
    for image in images:
        try:
            # print("ohhhhhhh")
            case_number = int(image.split('_')[0].replace('.png',''))
            # print("case_number",case_number)
            # print("case_number",list(df['case_number']))
            if case_number not in list(df['case_number']):
                print('yesyesyes')
                continue
            caption = df.loc[df.case_number==case_number]['prompt'].item()
            # print("caption",caption)
            im = Image.open(os.path.join(im_folder, image))
            inputs = processor(text=[caption], images=im, return_tensors="pt", padding=True)
            outputs = model(**inputs)
            clip_score = outputs.logits_per_image[0][0].detach().cpu() # this is the image-text similarity score
            # print("clip_score",clip_score)
            ratios[case_number] = ratios.get(case_number, []) + [clip_score]
        except:
            pass
    # print("ratios",ratios)
    for key in ratios.keys():
        df.loc[key,'clip'] = np.mean(ratios[key])
    df = df.dropna(axis=0)
    print(f"Mean CLIP score: {df['clip'].mean()}")
    print('-------------------------------------------------')
    print('\n')
