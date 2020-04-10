# -*- coding:utf-8 -*-
# !/usr/bin/env python

import argparse
import json
import matplotlib.pyplot as plt
# import skimage.io as io
import cv2
# from labelme import utils
import numpy as np
import glob
from PIL import Image
import os, sys
import random
import shutil
from collections import Counter
from tqdm import tqdm

image_size = [(586, 480), (704, 576)]

try:
    import xml.etree.cElementTree as ET  # 解析xml的c语言版的模块
except ImportError:
    import xml.etree.ElementTree as ET

class PascalVOC2coco(object):
    def __init__(self, xml=[], save_json_path=None, img=None, save_pic_path=None):

        '''
        :param xml: 所有Pascal VOC的xml文件路径组成的列表
        :param save_json_path: json保存位置
        '''
        self.xml = xml
        self.save_json_path = save_json_path
        self.save_pic_path = save_pic_path
        self.img = img
        self.images = []
        self.categories = []
        self.cls = ['holothurian', 'echinus', 'scallop', 'starfish']
        # * means the number of your dataset category
        self.id = dict(zip(self.cls, range(1, 5)))
        print(self.id)
        self.annotations = []
        self.label = []
        self.annID = 1
        self.height = 0
        self.width = 0
        self.ob = []
        self.save_json()

    def data_transfer(self):

        for num, json_file in enumerate(self.xml):
            # 进度输出
            sys.stdout.write('\r>> Converting image %d/%d' % (
                num + 1, len(self.xml)))
            sys.stdout.flush()

            self.json_file = json_file
            self.num = num
            with open(json_file, 'r') as fp:

                self.filename = json_file.split('\\')[-1].split('.')[0] + '.jpg'
                file_path = os.path.join(self.img, self.filename)
                # 划分训练集，测试集
                new_path = os.path.join(self.save_pic_path, self.filename)
                shutil.copyfile(file_path, new_path)
                img = Image.open(file_path)
                self.width = img.size[0]
                self.height = img.size[1]
                self.images.append(self.image())
                flag = 0
                for p in fp:

                    # if 'width' in p:
                    #     self.width = int(p.split('>')[1].split('<')[0])
                    #
                    # if 'height' in p:
                    #     self.height = int(p.split('>')[1].split('<')[0])
                    if flag == 1:
                        for i in range(1, 5):

                            if i == self.id[self.ob[0]]:
                                self.supercategory = self.ob[0]

                                if self.supercategory not in self.label:

                                    self.categories.append(self.categorie())
                                    self.label.append(self.supercategory)

                        # 边界框
                        x1 = int(self.ob[1])
                        y1 = int(self.ob[2])
                        x2 = int(self.ob[3])
                        if y2 == 1081:
                            y2 = int(self.ob[4]) - 1
                        else:
                            y2 = int(self.ob[4])

                        self.rectangle = [x1, y1, x2, y2]
                        self.bbox = [x1, y1, x2 - x1, y2 - y1]  # COCO 对应格式[x,y,w,h]
                        self.area = (x2 - x1) * (y2 - y1)
                        self.annotations.append(self.annotation())
                        self.annID += 1

                        self.ob = []
                        flag = 0
                    elif flag == 0:
                        if 'name' in p:
                            if p.split('>')[1].split('<')[0] != 'waterweeds': # 去除waterweeds类
                                self.ob.append(p.split('>')[1].split('<')[0])

                        if 'xmin' in p:
                            self.ob.append(p.split('>')[1].split('<')[0])

                        if 'ymin' in p:
                            self.ob.append(p.split('>')[1].split('<')[0])

                        if 'xmax' in p:
                            self.ob.append(p.split('>')[1].split('<')[0])

                        if 'ymax' in p:
                            self.ob.append(p.split('>')[1].split('<')[0])

                            flag = 1
                            if len(self.ob) == 4:
                                self.ob = []
                                flag = 0

                            elif len(self.ob) == 5:
                                # 去除标注bbox过小的样本类别
                                x1 = int(self.ob[1])
                                y1 = int(self.ob[2])
                                x2 = int(self.ob[3])
                                y2 = int(self.ob[4])

                                if (x2 - x1) * (y2 - y1) < 20:
                                    self.ob = []
                                    flag = 0

        sys.stdout.write('\n')
        sys.stdout.flush()

    def image(self):
        image = {}
        image['height'] = self.height
        image['width'] = self.width
        image['id'] = self.num + 1
        image['file_name'] = self.filename
        return image

    def categorie(self):
        categorie = {}
        categorie['supercategory'] = self.supercategory
        categorie['id'] = self.id[self.supercategory]
        categorie['name'] = self.supercategory
        return categorie

    def annotation(self):
        annotation = {}
        annotation['area'] = self.area
        annotation['segmentation'] = []
        annotation['iscrowd'] = 0
        annotation['image_id'] = self.num + 1
        annotation['bbox'] = self.bbox
        annotation['category_id'] = self.getcatid(self.supercategory)
        annotation['id'] = self.annID
        return annotation

    def getcatid(self, label):
        for categorie in self.categories:
            if label == categorie['name']:
                return categorie['id']
        return -1

    def data2coco(self):
        data_coco = {}
        data_coco['images'] = self.images
        data_coco['categories'] = self.categories
        data_coco['annotations'] = self.annotations
        return data_coco

    def save_json(self):
        self.data_transfer()
        self.data_coco = self.data2coco()
        # 保存json文件
        json.dump(self.data_coco, open(self.save_json_path, 'w'), indent=4)  # indent=4 更加美观显示


def clean_data(xml_path, num=1):
    '''
    可用于统计和筛选图片标注数量样本
    :param xml_path:
    :return:
    '''
    count_xml = []
    new_xml = []
    for xml in tqdm(xml_path):
        tree = ET.ElementTree(file=xml)  # 打开文件，解析成一棵树型结构
        root = tree.getroot()  # 获取树型结构的根
        ObjectSet = root.findall('object')  # 找到文件中所有含有object关键字的地方，这些地方含有标注目标

        if len(ObjectSet) > num: # 默认选取至少标注一个样本的图片
            new_xml.append(xml)
            count_xml.append(len(ObjectSet))
    print(Counter(count_xml))
    # print(new_xml)
    print('new_xml', len(new_xml))
    return new_xml


def ana_data(json_path):
    '''
    分析数据集样本及bbox面积分布
    :param json_path:
    :return:
    '''
    with open(json_path, 'r') as js:
        data = json.load(js)

        area = [a['area'] for a in data['annotations']]
        cat = [c['category_id'] for c in data['annotations']]
        image = [(i['height'], i['width']) for i in data['images']]
    # print(area, cat)
    area.sort()
    print(area[:15])
    new_area = []
    for a in area:
        if a < 20:
            new_area.append(a)

    area_count = Counter(new_area)
    cat_count = Counter(cat)
    img_count = Counter(image)
    print(area_count)
    print(cat_count)
    print(img_count)


if __name__ == '__main__':

    xml_path1 = 'box'
    img_path = 'image'
    save_json_dir = 'data/coco/annotations'
    save_pic_dir = 'data/coco'
    xml1 = [os.path.join(xml_path1, file) for file in os.listdir(xml_path1)]
    # clean data
    xml1 = clean_data(xml1)

    random.shuffle(xml1)

    train_xml = xml1[:5000]
    val_xml = xml1[5000:-1]

    splits = ['train', 'val']
    for split in splits:
        random.shuffle(train_xml)
        random.shuffle(val_xml)
        # save_path = os.path.join(save_json_dir, 'instances_' + split + '2017.json') # linux适用
        save_path = save_json_dir + '/' +  'instances_' + split + '2017.json' # windows适用
        if split == 'train':
            random.shuffle(train_xml)
            PascalVOC2coco(train_xml, save_path, img_path)
        if split == 'val':
            random.shuffle(val_xml)
            PascalVOC2coco(val_xml, save_path, img_path)

    ana_data(os.path.join(save_json_dir, 'instances_train2017.json'))
    ana_data(os.path.join(save_json_dir, 'instances_val2017.json'))
