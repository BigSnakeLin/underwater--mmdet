# _*_ coding: utf-8 _*_
# @Time    : 2020/3/23 15:31
# @Author  : LinQH
import _pickle as pk
import numpy as np
import json
import os
import cv2

headers = ['name', 'image_id', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax']
class_name = ['holothurian', 'echinus', 'scallop', 'starfish']

with open('result_pkl/cascade_x101_64x4d_fpn_1x.pkl', 'rb') as f:
    result_pkl = pk.load(f)
with open('data/coco/annotations/testA.json', 'r') as js:
    json_file = json.load(js)
image_list = json_file['images']
# print(image_list,len(image_list))
results = []
# print(len(result_pkl[:1][0]))
for k, res in enumerate(result_pkl):
    if image_list[k]['file_name'] in os.listdir('sample_result'): # 创建一个sample_result文件，自行挑选一些图片放入进行可视化观察
        bboxes = np.vstack(res)

        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(res)
        ]
        labels = np.concatenate(labels)

        scores = bboxes[:, -1]
        '''
        自定义阈值，方便可视化展示
        # inds = scores > 0.01
        # bboxes = bboxes[inds, :]
        # labels = labels[inds]
        # scores = scores[inds]
        '''
        # print(image_list[k]['file_name'])

        pic = os.path.join('sample_result', image_list[k]['file_name'])
        # pic = 'test-A-image{}'.format(image_list[k]['file_name'])
        img = cv2.imread(pic)
        
        for i in range(len(labels)):
            
            cv2.rectangle(img, (int(bboxes[i][0]), int(bboxes[i][1])), (int(bboxes[i][2]), int(bboxes[i][3])),
                          (0, 0, 255), 2)

            cv2.putText(img, class_name[int(labels[i])], (int(bboxes[i][0]), int(bboxes[i][1])),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)

            cv2.putText(img, str(scores[i]), (int(bboxes[i][2]), int(bboxes[i][3])), cv2.FONT_HERSHEY_COMPLEX, 1,
                        (0, 255, 255), 1)

        cv2.imwrite('result_vision/{}'.format(image_list[k]['file_name']), img)
