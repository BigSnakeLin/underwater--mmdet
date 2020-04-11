import _pickle as pk
import numpy as np
import pandas
import json
import os
from tqdm import tqdm
from ensemble_boxes import *
import cv2
from PIL import Image

headers = ['name', 'image_id', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax']
class_name = ['holothurian', 'echinus', 'scallop', 'starfish']
# cx101 72 cascade_r101_fpn_20e_nms.pkl 70
pkl_file = ['cx101.pkl',  'cascade_r101_fpn_20e_nms.pkl']

weights = [1, 1]
iou_thr = 0.7
skip_box_thr = 0.0001


with open('data/coco/annotations/testA.json', 'r') as js:
    json_file = json.load(js)
image_list = json_file['images']

results = []
for k in tqdm(range(len(image_list))):
    bboxes = []
    scores = []
    labels = []
    for pkl in pkl_file:
        with open(os.path.join('result', pkl), 'rb') as f:
            result_pkl = pk.load(f)

        res = result_pkl[k]

        box = np.vstack(res)
        label = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(res)
        ]
        label = np.concatenate(label)

        bboxes.append(box[:, :4])
        scores.append(box[:, 4])
        labels.append(label)

    pic = os.path.join('data/coco/test-A-image', image_list[k]['file_name'])
    img = Image.open(pic)

    labels = [l.tolist() for l in labels]
    bboxes = [b.tolist() for b in bboxes]
    scores = [s.tolist() for s in scores]

    bboxes, scores, labels = weighted_boxes_fusion(bboxes, scores, labels, weights=weights, iou_thr=iou_thr,
                                                   skip_box_thr=skip_box_thr)
    # bboxes, scores, labels = nms(bboxes, scores, labels, weights=weights, iou_thr=iou_thr)
    # bboxes, scores, labels = non_maximum_weighted(bboxes, scores, labels, weights=weights, iou_thr=iou_thr,
    #                                                skip_box_thr=skip_box_thr)
    print(len(labels))

    # inds = scores > 0.001
    #
    # bboxes = bboxes[inds, :]
    # labels = labels[inds]
    # scores = scores[inds]
    ##  可视化融合结果
    #     pic = os.path.join('data/coco/test-A-image', image_list[k]['file_name'])
    #     # pic = 'test-A-image{}'.format(image_list[k]['file_name'])
    #     img = cv2.imread(pic)
    #     # print(img,pic)
    #     for i in range(len(labels)):
    #         # print((int(bboxes[i][0]), int(bboxes[i][1])), (int(bboxes[i][2]), int(bboxes[i][3])))
    #         cv2.rectangle(img, (int(bboxes[i][0]), int(bboxes[i][1])), (int(bboxes[i][2]), int(bboxes[i][3])),
    #                       (0, 0, 255), 2)
    #
    #         cv2.putText(img, class_name[int(labels[i])], (int(bboxes[i][0]), int(bboxes[i][1])),
    #                     cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)
    #
    #         cv2.putText(img, str(scores[i]), (int(bboxes[i][2]), int(bboxes[i][3])), cv2.FONT_HERSHEY_COMPLEX, 1,
    #                     (0, 255, 255), 1)
    #
    #     cv2.imwrite('result_vision_fusion/{}'.format(image_list[k]['file_name']), img)

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
df.to_csv("submit/cx101-cr101-wbs.csv", index=False, sep=',')
