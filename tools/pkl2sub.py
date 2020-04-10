import _pickle as pk
import numpy as np
import pandas
import json
from ensemble_boxes import *
import os
from tqdm import tqdm
headers = ['name', 'image_id', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax']
class_name = ['holothurian', 'echinus', 'scallop', 'starfish']

with open('result/cascade_x101_64x4d_fpn_1x.pkl', 'rb') as f:
  result_pkl = pk.load(f)
with open('annotations/testB.json', 'r') as js:
  json_file = json.load(js)
image_list = json_file['images']

results = []

for k, res in enumerate(result_pkl):
  bboxes = np.vstack(res)
  labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(res)
        ]
  labels = np.concatenate(labels)

  scores = bboxes[:, -1]
  """
  # inds = scores > 0.00099 # 自定义阈值
  # bboxes = bboxes[inds, :]
  # labels = labels[inds]
  # scores = scores[inds]
  """
  print(len(labels))

  for i in range(len(labels)):
    new_res = {}

    new_res['name'] = class_name[int(labels[i])]
    new_res['xmin'] = int(bboxes[i][0])
    new_res['ymin'] = int(bboxes[i][1])
    new_res['xmax'] = int(bboxes[i][2])
    new_res['ymax'] = int(bboxes[i][3])
    new_res['confidence'] = scores[i]
    new_res['image_id'] = image_list[k]['file_name'].split('.')[0] + '.xml'

    results.append(new_res)

sub = {}
name = [result['name'] for result in results]
image_id = [result['image_id'] for result in results]
confidence = [result['confidence'] for result in results]
xmin = [result['xmin'] for result in results]
ymin = [result['ymin'] for result in results]
xmax = [result['xmax'] for result in results]
ymax = [result['ymax'] for result in results]
sub['name'] = name
sub['image_id'] = image_id
sub['confidence'] = confidence
sub['xmin'] = xmin
sub['ymin'] = ymin
sub['xmax'] = xmax
sub['ymax'] = ymax
df = pandas.DataFrame(sub)
df.to_csv("result_sub/sub_cascade_x101_64x4d_fpn_1x_B.csv", index=False, sep=',') # input your filename

