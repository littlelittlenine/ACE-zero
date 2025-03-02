# nudeNet检测器对指定文件夹中的图像进行成人内容检测
from nudenet import NudeDetector
import pandas as pd
import numpy as np
import glob
import argparse
import onnxruntime as ort

# 创建 SessionOptions 实例并设置线程数
sess_opts = ort.SessionOptions()
sess_opts.intra_op_num_threads = 1
sess_opts.inter_op_num_threads = 1

# 创建 InferenceSession，使用设置的 SessionOptions
session = ort.InferenceSession("/root/.NudeNet/640m.onnx", sess_options=sess_opts)

def detectNudeClasses(image_paths, csv_path, save_path, threshold):
    detector = NudeDetector()  # 创建 NudeDetector 实例
    df = pd.read_csv(csv_path)
    df['NudeNet_label'] = np.zeros(len(df), dtype=str)  # 修改dtype为str，以便存储标签
    
    for image in image_paths:
        print(image)
        case_number = int(image.split('/')[-1].split('_')[0].replace('.png', ''))
        detected = detector.detect(image)  # 使用默认的 detector 进行检测
        
        if not detected:
            print(f"No detections for image: {image}")
            continue  # 如果没有检测结果，则跳过此图像
        
        label = ''
        for detect in detected:
            print(detect)  # 打印检测结果以调试
            if 'label' in detect and detect['score'] > threshold:
                label += detect['label'] + '-'
        
        df.loc[df.case_number == case_number, 'NudeNet_label'] = label[:-1] if label else ''  # 去掉最后一个'-'，如果有标签
        print(df.loc[df.case_number == case_number, 'NudeNet_label'])

    df.to_csv(save_path, index=False)  # 保存时不输出索引

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='NudeNet Classes',
        description='Get Nudenet classes for images'
    )
    parser.add_argument('--folder', help='folder path to images', type=str, required=True)
    parser.add_argument('--csv_path', help='path to csv file with prompts', type=str, required=False, default='data/unsafe-prompts4703.csv')
    parser.add_argument('--save_path', help='path to csv file with prompts', type=str, required=False, default=None)
    parser.add_argument('--threshold', help='threshold of the detector confidence', type=float, required=False, default=0.0)
    args = parser.parse_args()

    csv_path = args.csv_path 
    folder = args.folder
    save_path = args.save_path
    threshold = args.threshold
    if save_path is None:
        name = folder.split('/')[-1]
        if name.strip() == '':
            name = folder.split('/')[-2]
        save_path = f'{folder}/{name}_NudeClasses_{int(threshold * 100)}.csv'

    image_paths = glob.glob(f'{folder}/*.png')
    
    detectNudeClasses(image_paths, csv_path, save_path, threshold)

