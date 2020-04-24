
# coding: utf-8

# In[1]:


import sys
from detection.MtcnnDetector import MtcnnDetector
from detection.detector import Detector
from detection.fcn_detector import FcnDetector
from train.model import P_Net,R_Net,O_Net,P_Net_v1,R_Net_v1,R_Net_fcn,R_Net_fcn_v1,O_Net_v1,O_Net_fcn_v1
import cv2
import os
import numpy as np
import train.config_eval as config

from train.model import P_Net_8,P_Net_9

import pdb
import time
import os.path as osp
from tqdm import tqdm
import tensorflow as tf


# In[ ]:


def cal_scale(img_size):

    bandwidth = 1024**2
    channel = 10    # PNet第一个卷积层的通道数

    h, w = img_size

    long_e = np.max(img_size)
    short_e = np.min(img_size)
    ratio = short_e / long_e

    long_e_resized = (bandwidth / (channel * ratio))**0.5 - 1   #减一是为了预留一定的带宽

    return long_e_resized / long_e

# pdb.set_trace()


def resize_image(img, scale):
    """
        按照给定的比例resize图片
    """

    height, width, channels = img.shape
    new_height = int(height * scale)  
    new_width = int(width * scale)  
    new_dim = (new_width, new_height)
    img_resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_LINEAR) 

    return img_resized


# pdb.set_trace()


test_mode=config.test_mode
thresh=config.thresh
min_face_size=config.min_face
stride=config.stride
suffix=config.suffix
path=config.test_dir
src_img_root=config.img_root
save_img=config.save_img
save_tensorboard=config.save_tensorboard
obj_name=config.obj_name
resize_input=config.resize_input
# model_path=config.model_path
sliding_win=config.sliding_win
win_size=config.win_size
batch_size=config.batches
net_name=config.net_name

# 确保resize_input和sliding_win至多只能有一个为True
assert not (resize_input and sliding_win)


detectors=[None,None,None]
# 模型放置位置
# if model_path is None:  ####
#     if suffix is None:
#         model_path=['model/PNet/','model/RNet/','model/ONet/']
#     else:
#         if obj_name == 'head' and suffix == 'all':
#             model_path=['model/PNet/','model/{}/RNet/'.format(suffix),'model/{}/ONet/'.format(suffix)]
#         else:
#             model_path=['model/{}/PNet/'.format(suffix),'model/{}/RNet/'.format(suffix),'model/{}/ONet/'.format(suffix)]
# else:
#     assert isinstance(model_path, list)

if net_name is not None:
    net_num = len(net_name)
    model_path = [None, None, None]
    assert net_num == (('PNet', 'RNet', 'ONet').index(test_mode) + 1)

    if net_num == 1:
        model_path[0] = '../model_mine/{}/{}/'.format(args.suffix, net_name[0])
    elif net_num == 2:
        model_path[0] = '../model_mine/{}/{}/'.format(args.suffix, net_name[0])
        model_path[1] = '../model_mine/{}/{}/'.format(args.suffix, net_name[1])
    else:
        model_path[0] = '../model_mine/{}/{}/'.format(args.suffix, net_name[0])
        model_path[1] = '../model_mine/{}/{}/'.format(args.suffix, net_name[1])
        model_path[2] = '../model_mine/{}/{}/'.format(args.suffix, net_name[2])
else:
    if args.suffix is None:
        model_path=['../model_mine/PNet/','../model_mine/RNet/','../model_mine/ONet']
    else:
        model_path=['../model_mine/{}/PNet/'.format(args.suffix),'../model_mine/{}/RNet/'.format(args.suffix),'../model_mine/{}/ONet'.format(args.suffix)]






# if 'PNet_8' in model_path[0]:   ####
#     PNet=FcnDetector(P_Net_8,model_path[0])
# elif 'PNet_9' in model_path[0]:
#     PNet=FcnDetector(P_Net_9,model_path[0])
# else:
#     PNet=FcnDetector(P_Net,model_path[0])
# detectors[0]=PNet

# if test_mode in ["RNet", "ONet"]:
#     RNet = Detector(R_Net, 24, batch_size[1], model_path[1])
#     detectors[1] = RNet

# if test_mode == "ONet":
#     ONet = Detector(O_Net, 48, batch_size[2], model_path[2])
#     detectors[2] = ONet


if net_name is None:
    PNet=FcnDetector(P_Net,model_path[0])
    if test_mode in ["RNet", "ONet"]:
        RNet = Detector(R_Net, 24, batch_size[1], model_path[1])
        detectors[1] = RNet
    if test_mode == "ONet":
        ONet = Detector(O_Net, 48, batch_size[2], model_path[2])
        detectors[2] = ONet
else:
    if test_mode == "PNet":
        PNet=FcnDetector(eval(net_name[0]),model_path[0])
    elif test_mode == "RNet":
        PNet=FcnDetector(eval(net_name[0]),model_path[0])
        RNet=Detector(eval(net_name[1]),24,batch_size[1],model_path[1])
        detectors[1]=RNet
    else:
        PNet=FcnDetector(eval(net_name[0]), model_path[0])
        RNet=Detector(eval(net_name[1]), 24, batch_size[1], model_path[1])
        ONet = Detector(eval(net_name[2]), 48, batch_size[2], model_path[2])
        detectors[1]=RNet 
        detectors[2]=ONet 
detectors[0]=PNet




# if 'PNet_8' in model_path[0]:   ####
#     mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
#                                    stride=stride, threshold=thresh, pnet_size=8, 
#                                    sliding_win=sliding_win, win_size=win_size)
#     out_path=osp.join(config.out_path, 'PNet_8-' + test_mode) if suffix is None else \
#                     osp.join(config.out_path, suffix, 'PNet_8-' + test_mode)
# elif 'PNet_9' in model_path[0]:
#     mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
#                                    stride=stride, threshold=thresh, pnet_size=9,
#                                    sliding_win=sliding_win, win_size=win_size)
#     out_path=osp.join(config.out_path, 'PNet_9-' + test_mode) if suffix is None else \
#                     osp.join(config.out_path, suffix, 'PNet_9-' + test_mode)
# else:

mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                               stride=stride, threshold=thresh,
                               sliding_win=sliding_win, win_size=win_size)
if net_name is None:
    out_path=osp.join(config.out_path, test_mode) if suffix is None else \
                    osp.join(config.out_path, suffix, test_mode)
else:
    out_path=osp.join(config.out_path, net_name[('PNet', 'RNet', 'ONet').index(test_mode)]) if suffix is None else \
                    osp.join(config.out_path, suffix, net_name[('PNet', 'RNet', 'ONet').index(test_mode)])
out_txt_root = osp.join(out_path, 'txt')
out_img_root = osp.join(out_path, 'im', str(thresh[['PNet', 'RNet', 'ONet'].index(test_mode)]))
out_tb_root = osp.join(out_path, 'tb', str(thresh[['PNet', 'RNet', 'ONet'].index(test_mode)]))


###
if not osp.isdir(out_path):
    os.makedirs(out_path)
if not osp.isdir(out_txt_root):
    os.makedirs(out_txt_root) 
if not osp.isdir(out_img_root):
    os.makedirs(out_img_root)
if not osp.isdir(out_tb_root):
    os.makedirs(out_tb_root)


if save_tensorboard:

    tb_size = (500, 500)

    tb_img = tf.placeholder(tf.float32, [tb_size[0], tb_size[1], 3], name='image')
    tb_img_reshaped = tf.reshape(tb_img, [-1, tb_size[0], tb_size[1], 3])
    tf.summary.image('input', tb_img_reshaped, max_outputs=100)
    merged = tf.summary.merge_all()

    sess = tf.InteractiveSession()
    # pdb.set_trace()
    tf.global_variables_initializer().run()
    writer = tf.summary.FileWriter(out_tb_root)


if path is not None:
    with open(path, 'r') as f:
        lines = f.readlines()

    relative_paths = [x.strip().split()[0] for x in lines]
else:
    relative_paths = os.listdir(src_img_root)

#print(path)
for k, relative_path in tqdm(enumerate(relative_paths)):
    # start = time.time()

    img_path=os.path.join(src_img_root, relative_path)
    img=cv2.imread(img_path)
    src_img = img.copy()

    # pdb.set_trace()

    if resize_input:
        ratio = cal_scale(img.shape[:2])
        img = resize_image(img, ratio)

    # pdb.set_trace()

    boxes_c,landmarks=mtcnn_detector.detect(img)

    # end = time.time()
    # print("Inference time: {}s".format(end - start))

    img_name = osp.basename(img_path)
    txt_name = "{}.txt".format(img_name[:-4])
    out_txt_dir = osp.join(out_txt_root, txt_name)

    f = open(out_txt_dir, 'w')


    for i in range(boxes_c.shape[0]):
        bbox=boxes_c[i,:4]
        score=boxes_c[i,4]

        # 如果输入图片经过resize，要把坐标放缩回原图大小
        if resize_input:
            bbox = bbox / ratio

        corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]

        if save_img or save_tensorboard:

            if resize_input:
                img = src_img

            #画人脸框
            cv2.rectangle(img, (corpbbox[0], corpbbox[1]),
                          (corpbbox[2], corpbbox[3]), (255, 0, 0), 1)
            #判别为人脸的置信度
            cv2.putText(img, '{:.2f}'.format(score), 
                       (corpbbox[0], corpbbox[1] - 2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 0, 255), 2)

        line = ' '.join([obj_name, str(score)] + [str(x) for x in corpbbox]) + '\n'

        f.write(line)
  
    if save_img:
        out_img_dir = osp.join(out_img_root, img_name)
        cv2.imwrite(out_img_dir, img)

    if save_tensorboard:
        img = cv2.resize(img, tb_size)[:,:,::-1] # rgb 2 bgr
        summary = sess.run(merged, feed_dict={tb_img: img})
        writer.add_summary(summary, k)

    f.close()


if save_tensorboard:

    writer.close()
    sess.close()
