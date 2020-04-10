# under-water-uodac-mmdet

**News**: We released the technical report on [ArXiv](https://arxiv.org/abs/1906.07155).

Documentation: https://mmdetection.readthedocs.io/

## 2020 水下目标检测大赛 个人参赛心得
**赛址**：https://www.kesci.com/home/competition/5e535a612537a0002ca864ac

**使用框架**：主要使用mmdetection开源检测工具，前期也尝试使用centernet开源算法。

## 数据集处理及分析
**1**： 赛方给的是xml格式的标注文件，共5543张，这里通过脚本xml2coco.py转换成标准coco格式
**2**:  数据分析得知，图像共有有效标注5462张，没有标注的负样本图像达到81张，且图像尺寸不一，主要有（3840, 2160）--> 1072张，（1920, 1080）--> 477张，（720， 405）--> 2350张，（704, 576）--> 31张，（586， 480）--> 36张，
**3**：  标注高宽比分析--> 通过脚本bbox2analysis.py, bbox高宽比主要分布在1.0附近，少量2.0附近
**4**：  标注类别分布--> Counter({2: 22340, 4: 6841, 3: 6713, 1: 5536})---->(1:holothurian, 2:echinus, 3:scallop, 4:starfish)--->可以明显看出2类远高于其他类别，另外有一些标注框框存在标注错误，比如标注越界或者bbox面积过小，这里均可以通过xml2coco.py脚本进行处理
**5**：  数据集图像均值和方差分析-->通过脚本compute_mean_std.py获得--->(可用于改善图像归一化分布)
**6**：  通过观察训练集图样，可以明显看出水下采集图像具有运动模糊（机器运动采集），水中杂质散射，图像清晰度和对比度较低等特性，因此前期在训练过程中考虑加入颜色抖动、随机色度变化、模糊算子等数据增强手段（似乎对于这个比赛测试集并不work）

## 选用模型
**01**：cascade_r50_fpn_1x.py ---> 线上map44+
**02**：cascade_r101_fpn_1x.py --> 线上map46+
**03**：cascade_x101_32x4d_fpn_16e.py --> 线上map48+
**04**：cascade_x101_64x4d_fpn_16e.py --> 线上map49+
**其他一些模型试验**： CenterNet--> 线上map最高44+，Atss_x101_32x4d_fpn_16e.py--> 线上map最高45+，grid_rcnn_gn_head_x101_32x4d_fpn_2x.py-->线上map最高43+， cascade_rcnn_hrnetv2p_w32.py-->线上最高map42+，这些都是实验过程中验证针对该数据集并不work的模型，当然也可能是我训练的tricks不行，自行参考。
## Train
  python tools/train.py waterconfigs/{your_config_file} --gpus {int nums}
## Test
  python tools/test.py waterconfigs/{your_config_file} {your_epoch_file} --eval bbox
## Generate Submission File
  **这里提供两种方法**：
  *1*.借鉴斩风大佬的转换方法：
  首先产生一个测试集对应的testA.json文件，方便使用mmdet自带的测试管道进行测试，斩风大佬是产生一个json文件，命令如下：
      python tools/test.py waterconfigs/{your_config_file} {your_epoch_file} --json_out result/json_filename
      python tools/json2sub.py
  *2*.自己的转换方法：
      python tools/test.py waterconfigs/{your_config_file} {your_epoch_file} --out result/pkl_filename.pkl
      python tools/pkl2sub.py
## Citation

If you use this toolbox or benchmark in your research, please cite this project.

```
@article{mmdetection,
  title   = {{MMDetection}: Open MMLab Detection Toolbox and Benchmark},
  author  = {Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and
             Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and
             Liu, Ziwei and Xu, Jiarui and Zhang, Zheng and Cheng, Dazhi and
             Zhu, Chenchen and Cheng, Tianheng and Zhao, Qijie and Li, Buyu and
             Lu, Xin and Zhu, Rui and Wu, Yue and Dai, Jifeng and Wang, Jingdong
             and Shi, Jianping and Ouyang, Wanli and Loy, Chen Change and Lin, Dahua},
  journal= {arXiv preprint arXiv:1906.07155},
  year={2019}
}
```


## Contact
```
  email: 21860137@zju.edu.cn
```
