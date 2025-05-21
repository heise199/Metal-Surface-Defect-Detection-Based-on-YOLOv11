# -*- coding:utf-8 -*-
# @author: 牧锦程
# @微信公众号: AI算法与电子竞赛
# @Email: m21z50c71@163.com
# @VX：fylaicai

import os
import random
import shutil

from tqdm import tqdm

train_percent = 0.8  # 训练集占总比例的比例
file_path = 'GC10-DET'
save_train_path = "train/images"
save_val_path = "val/images"
total_file = os.listdir(file_path)[:-2]

if os.path.exists(save_train_path) or os.path.exists(save_val_path):
    shutil.rmtree(save_train_path)
    shutil.rmtree(save_val_path)

os.makedirs(save_train_path)
os.makedirs(save_val_path)

for name in tqdm(total_file, desc="划分图片数据"):
    total_num = os.listdir(os.path.join(file_path, name))
    random.shuffle(total_num)

    train_file = total_num[:int(len(total_num) * train_percent)]
    val_file = total_num[int(len(total_num) * train_percent):]

    for train in train_file:
        shutil.copy(os.path.join(file_path, name, train), os.path.join(save_train_path, train))
    for val in val_file:
        shutil.copy(os.path.join(file_path, name, val), os.path.join(save_val_path, val))
