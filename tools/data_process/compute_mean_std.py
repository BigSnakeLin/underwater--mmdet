# _*_ coding: utf-8 _*_
# @Time    : 2020/3/26 9:35
# @Author  : LinQH
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
# from scipy.misc import imread
from imageio import imread
import random
image_size = [(586, 480), (704, 576), (1920, 1080), (3840, 2160), (720, 405)]

def generate_sample_data(xmlpath, imgpath):
    new_data = {}
    for size in image_size:
        new_xml = []
        for xml in xmlpath:
            pic = imgpath + '/' + xml.split('\\')[-1].split('.')[0] + '.jpg'
            img = Image.open(pic)
            img_size = img.size
            # print(img_size)
            if img_size == size:
                new_xml.append(xml)
        new_data[size] = new_xml
    return new_data

xml_path1 = 'box'
img_path = 'image'
save_json_dir = 'annotations'

xml1 = [os.path.join(xml_path1, file) for file in os.listdir(xml_path1)]


# generate sample data
new_xml_data = generate_sample_data(xml1, img_path)
print(len(new_xml_data))

sample_data = []
sample_num = 800 # 根据尺寸分布选取800张样本图片
for nx in new_xml_data:
    num = int(len(new_xml_data[nx])/len(xml1) * sample_num)
    sample_data += random.sample(new_xml_data[nx], num)
print(len(sample_data))
#
random.shuffle(sample_data)

print(sample_data)
R_channel = 0
G_channel = 0
B_channel = 0
R_channel_square = 0
G_channel_square = 0
B_channel_square = 0
pixels_num = 0

imgs = []
for i in range(len(sample_data)):
    pic = img_path + '/' + sample_data[i].split('\\')[-1].split('.')[0] + '.jpg'
    img = imread(pic)
    h, w, _ = img.shape
    pixels_num += h * w  # 统计单个通道的像素数量

    R_temp = img[:, :, 0]
    R_channel += np.sum(R_temp)
    R_channel_square += np.sum(np.power(R_temp, 2.0))
    G_temp = img[:, :, 1]
    G_channel += np.sum(G_temp)
    G_channel_square += np.sum(np.power(G_temp, 2.0))
    B_temp = img[:, :, 2]
    B_channel = B_channel + np.sum(B_temp)
    B_channel_square += np.sum(np.power(B_temp, 2.0))

R_mean = R_channel / pixels_num
G_mean = G_channel / pixels_num
B_mean = B_channel / pixels_num

"""
S^2
= sum((x-x')^2 )/N = sum(x^2+x'^2-2xx')/N
= {sum(x^2) + sum(x'^2) - 2x'*sum(x) }/N
= {sum(x^2) + N*(x'^2) - 2x'*(N*x') }/N
= {sum(x^2) - N*(x'^2) }/N
= sum(x^2)/N - x'^2
"""

R_std = np.sqrt(R_channel_square / pixels_num - R_mean * R_mean)
G_std = np.sqrt(G_channel_square / pixels_num - G_mean * G_mean)
B_std = np.sqrt(B_channel_square / pixels_num - B_mean * B_mean)

print("R_mean is %f, G_mean is %f, B_mean is %f" % (R_mean, G_mean, B_mean))
print("R_std is %f, G_std is %f, B_std is %f" % (R_std, G_std, B_std))


# 64.212456, 151.620467, 83.109372
# 27.647789, 42.758793, 44.519865
