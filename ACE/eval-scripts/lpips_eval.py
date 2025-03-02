# 它同时支持单张和多张图像的LPIPS得分计算，并尝试实现样式和内容损失的计算（虽然实现不完整）。如果需要完整的功能，可能需要定义缺失的函数。整体上，这段代码提供了更多的灵活性和功能，但也引入了更多的复杂性。
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import copy
import os
import pandas as pd
import argparse
import lpips
from styleloss import get_style_content_loss

# desired size of the output image
imsize = 64
loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor


def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    image = (image-0.5)*2
    return image.to(torch.float)


if __name__=='__main__':
    parser = argparse.ArgumentParser(
                    prog = 'LPIPS',
                    description = 'Takes the path to two images and gives LPIPS')
    parser.add_argument('--original_path', help='path to original image', type=str, required=True)
    parser.add_argument('--edited_path', help='path to edited image', type=str, required=True)
    parser.add_argument('--csv_path', help='path to csv prompts', type=str, required=True)
    parser.add_argument('--save_path', help='path to save results', type=str, required=False, default=None)
    parser.add_argument(
        "--image",
        action="store_true",
        help="Whether it is a single image path",
    )
    print(lpips)
    loss_fn_alex = lpips.LPIPS(net='alex')
    args = parser.parse_args()
    print('args.image:',args.image)
    if args.image:
        original = image_loader(args.original_path)
        edited = image_loader(args.edited_path)

        style_score, content_score, total_score = get_style_content_loss(cnn, cnn_normalization_mean, cnn_normalization_std,
                                    content_img=original, style_img=original, input_img=edited)
        if args.save_path is not None:
            df = pd.DataFrame({'filename': [args.edited_path.split('/')[-1]], 'Style_Loss': [style_score], 'Content_Loss': [content_score], 'Total_Loss': [total_score]})
            df.to_csv(args.save_path)
    else:
        file_names = os.listdir(args.original_path)
        file_names = [name for name in file_names if '.png' in name] # 列举路径下的所有文件
        df_prompts = pd.read_csv(args.csv_path) # 读取csv文件
        print('file_names:',len(file_names))
        df_prompts['lpips_loss'] = df_prompts['case_number'] *0 # 初始化Lpips损失列
        for index, row in df_prompts.iterrows(): # 这行代码开始遍历CSV文件中的每一行
            case_number = row.case_number # 获取当前行的case_number列的值
            files = [file for file in file_names if file.startswith(f'{case_number}_')] # 以case_number开头的文件
            lpips_scores = [] # 分数
            for file in files: # 遍历文件列表
                print('file:',file)
                # 对于每个匹配的文件对（原始和编辑后的图像），加载图像并计算LPIPS分数。如果遇到任何异常（例如文件不存在），则捕获异常并打印消息。
                try:
                    original = image_loader(os.path.join(args.original_path,file))
                    edited = image_loader(os.path.join(args.edited_path,file))

                    l = loss_fn_alex(original, edited)
                    print(f'LPIPS score: {l.item()}')
                    lpips_scores.append(l.item())
                except Exception:
                    print('No File')
                    pass
            df_prompts.loc[index,'lpips_loss'] = np.mean(lpips_scores)
        if args.save_path is not None:
            if len(os.path.basename(args.edited_path).strip()) == 0:
                basename = args.edited_path.split('/')[-2]
            else:
                basename = args.edited_path.split('/')[-1]
            df_prompts.to_csv(os.path.join(args.save_path, f'{basename}_lpipsloss.csv'))
# python eval-scripts/lpips_eval.py --original_path '/share/u/rohit/www/closed_form/niche_short/original/' --csv_path '/share/u/rohit/erase-closed/data/short_niche_art_prompts.csv' --save_path '/share/u/rohit/www/closed_form/niche_short/' --edited_path '/share/u/rohit/www/closed_form/niche_short/erasing-ThomasKinkade-with-preservation/'
