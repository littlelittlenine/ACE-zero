from PIL import Image
import requests
import os, glob
import pandas as pd
import numpy as np
import re
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("/mnt/bn/intern-disk/mlx/users/wrp/ecu-nullspace/clip-vit")
processor = CLIPProcessor.from_pretrained("/mnt/bn/intern-disk/mlx/users/wrp/ecu-nullspace/clip-vit")

def sorted_nicely( l ):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

path = '/mlx/mlx_devbox/users/wangruipeng/playground/paper_run/uce_nullspace/unified-concept-editing-main/data/rich/1' 
model_names = os.listdir(path)                
model_names = [m for m in model_names] 
csv_path = '/mlx/mlx_devbox/users/wangruipeng/playground/paper_run/uce_nullspace/unified-concept-editing-main/data/coco_2014_random_100.csv' 
save_path = '/mlx_devbox/users/wangruipeng/playground/paper_run/uce_nullspace/unified-concept-editing-main/data/output_score/clip_score' 
print("model_names",model_names)
model_names.sort() # 排序
if 'original' in model_names:
    model_names.remove('original')

for model_name in model_names:
    print(model_name)
    im_folder = os.path.join(path, model_name)
    df = pd.read_csv(csv_path)
    images = os.listdir(im_folder)
    images = sorted_nicely(images)
    ratios = {}
    df['clip'] = np.nan
    print("yes")
    for image in images:
        try:
            case_number = int(image.split('_')[0].replace('.png',''))
            if case_number not in list(df['case_number']):
                print('yesyesyes')
                continue
            caption = df.loc[df.case_number==case_number]['prompt'].item()
            im = Image.open(os.path.join(im_folder, image))
            inputs = processor(text=[caption], images=im, return_tensors="pt", padding=True)
            outputs = model(**inputs)
            clip_score = outputs.logits_per_image[0][0].detach().cpu() # this is the image-text similarity score
            ratios[case_number] = ratios.get(case_number, []) + [clip_score]
        except:
            pass
    for key in ratios.keys():
        df.loc[key,'clip'] = np.mean(ratios[key])
    df = df.dropna(axis=0)
    print(f"Mean CLIP score: {df['clip'].mean()}")
    print('-------------------------------------------------')
    print('\n')
