import os
import numpy as np
from PIL import Image
import math
from collections import Counter

# 定义函数来计算类别熵
def calculate_entropy(image_folder):
    image_files = os.listdir(image_folder)
    labels = [file.split('_')[-1].split('.')[0] for file in image_files]  # 从图像文件名中提取类别名
    label_counts = Counter(labels)  # 统计每个类别的数量
    total_count = len(labels)  # 总样本数
    label_probs = [count / total_count for count in label_counts.values()]  # 计算每个类别的概率
    entropy = -sum(p * math.log(p, 2) for p in label_probs)  # 计算类别熵
    return entropy

# 循环遍历每个子文件夹并计算类别熵
base_folder = 'vis_pp_cifar_v2'
for cpt in range(100):
    cpt_folder = os.path.join(base_folder, 'cpt{}'.format(cpt))
    entropy = calculate_entropy(cpt_folder)
    print('类别熵 (cpt{}): {}'.format(cpt, entropy))
