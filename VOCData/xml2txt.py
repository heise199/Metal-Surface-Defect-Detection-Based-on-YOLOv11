# -*- coding:utf-8 -*-
# @author: 牧锦程
# @微信公众号: AI算法与电子竞赛
# @Email: m21z50c71@163.com
# @VX：fylaicai

import shutil
import xml.etree.ElementTree as ET
import os
from tqdm import tqdm


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return [x, y, w, h]


def convert_annotation(xml_path, save_path):
    in_file = open(xml_path, encoding='utf-8')
    tree = ET.parse(in_file)
    root = tree.getroot()
    for size in root.iter("size"):
        w = int(size.find("width").text)
        h = int(size.find("height").text)

    all_boxs = []
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        b1, b2, b3, b4 = b
        b = (b1, b2, b3, b4)
        bb = convert((w, h), b)
        bb.insert(0, cls_id)
        all_boxs.append(bb)

    out_file = open(f'{save_path}/{os.path.split(xml_path)[-1].replace(".xml", ".txt")}', 'w', encoding="utf-8")
    for box in all_boxs:
        out_file.write(" ".join([str(i) for i in box]) + '\n')
    out_file.close()


if __name__ == "__main__":
    file_path = 'GC10-DET'
    sets = ['train', 'val']
    # 改成自己的类别
    classes = ['1_chongkong', '2_hanfeng', '3_yueyawan', '4_shuiban', '5_youban', '6_siban', '7_yiwu', '8_yahen', '9_zhehen', '10_yaozhe']
    abs_path = os.getcwd()
    print(abs_path)

    if os.path.exists('./train/labels') or os.path.exists("./val/labels"):
        shutil.rmtree('./train/labels/')
        shutil.rmtree('./val/labels/')

    os.makedirs('./train/labels/')
    os.makedirs('./val/labels/')

    for image_set in sets:
        for i in tqdm(os.listdir(f"./{image_set}/images"), desc=f"{image_set}"):
            xml_path = f"./{file_path}/label/{i.replace('.jpg', '.xml')}"
            try:
                convert_annotation(xml_path, f"./{image_set}/labels")
            except:
                os.remove(f"./{image_set}/images/" + i)
                