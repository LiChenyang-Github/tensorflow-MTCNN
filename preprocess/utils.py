
# coding: utf-8

# In[1]:

import pdb

import numpy as np

import os.path as osp
import numpy.random as npr
# In[2]:

from glob import glob
from collections import defaultdict

npr.seed(333)



def calculate_iou(dets_a, dets_b):
    """
        Calculate the overlap of two np.array.
        - inputs:
            dets_a: (N, 4)
            dets_b: (M, 4)
        - outputs:
            res: (N, M)
    """

    N = dets_a.shape[0]
    M = dets_b.shape[0]


    dets_a = np.tile(dets_a[:, np.newaxis, :], (1, M, 1))
    dets_b = np.tile(dets_b[np.newaxis, :, :], (N, 1, 1))

    # pdb.set_trace()
    aw = dets_a[:,:,2] - dets_a[:,:,0]
    ah = dets_a[:,:,3] - dets_a[:,:,1]
    bw = dets_b[:,:,2] - dets_b[:,:,0]
    bh = dets_b[:,:,3] - dets_b[:,:,1]

    area_a = aw * ah
    area_b = bw * bh

    iw = np.minimum(dets_a[:,:,2], dets_b[:,:,2]) - \
            np.maximum(dets_a[:,:,0], dets_b[:,:,0])
    iw = np.maximum(iw, 0)

    ih = np.minimum(dets_a[:,:,3], dets_b[:,:,3]) - \
            np.maximum(dets_a[:,:,1], dets_b[:,:,1])
    ih = np.maximum(ih, 0)

    # pdb.set_trace()

    inter = iw * ih

    iou = inter / (area_a + area_b - inter)

    return iou



def IOU(box,boxes):
    '''裁剪的box和图片所有人脸box的iou值
    参数：
      box：裁剪的box,当box维度为4时表示box左上右下坐标，维度为5时，最后一维为box的置信度
      boxes：图片所有人脸box,[n,4]
    返回值：
      iou值，[n,]
    '''
    #box面积
    box_area=(box[2]-box[0]+1)*(box[3]-box[1]+1)
    #boxes面积,[n,]
    area=(boxes[:,2]-boxes[:,0]+1)*(boxes[:,3]-boxes[:,1]+1)
    #重叠部分左上右下坐标
    xx1=np.maximum(box[0],boxes[:,0])
    yy1=np.maximum(box[1],boxes[:,1])
    xx2=np.minimum(box[2],boxes[:,2])
    yy2=np.minimum(box[3],boxes[:,3])
    
    #重叠部分长宽
    w=np.maximum(0,xx2-xx1+1)
    h=np.maximum(0,yy2-yy1+1)
    #重叠部分面积
    inter=w*h
    return inter/(box_area+area-inter+1e-10)


# In[3]:

def read_annotation(base_dir, label_path):
    '''读取文件的image，box'''
    data = dict()
    images = []
    bboxes = []
    labelfile = open(label_path, 'r')
    while True:
        # 图像地址
        imagepath = labelfile.readline().strip('\n')
        if not imagepath:
            break
        imagepath = base_dir + '/images/' + imagepath
        images.append(imagepath)
        # 人脸数目
        nums = labelfile.readline().strip('\n')
     
        one_image_bboxes = []
        for i in range(int(nums)):
           
            bb_info = labelfile.readline().strip('\n').split(' ')
            #人脸框
            face_box = [float(bb_info[i]) for i in range(4)]
            
            xmin = face_box[0]
            ymin = face_box[1]
            xmax = xmin + face_box[2]
            ymax = ymin + face_box[3]
            
            one_image_bboxes.append([xmin, ymin, xmax, ymax])
           
        bboxes.append(one_image_bboxes)


    data['images'] = images
    data['bboxes'] = bboxes
    return data

def read_annotation_xiao(base_dir, label_path, img_num=None, add_img_suffix=False):
    '''读取西奥数据的image，box'''
    data = dict()
    images = []
    bboxes = []

    with open(label_path, 'r') as f:

        lines = f.readlines()

    npr.shuffle(lines)

    # pdb.set_trace()

    if img_num is not None:
        assert img_num <= len(lines)
        lines = lines[:img_num]


    for line in lines:
        conts = line.strip().split()
        if not add_img_suffix:
            imagepath = osp.join(base_dir, conts[0])
        else:
            imagepath = glob(osp.join(base_dir, conts[0]+'.*'))[0]
            # pdb.set_trace()

        one_image_bboxes = np.array([float(x) for x in conts[1:]]).reshape(-1, 4).tolist()

        images.append(imagepath)
        bboxes.append(one_image_bboxes)


    data['images'] = images
    data['bboxes'] = bboxes
    return data


def read_annotation_2cls(base_dir, label_path_list, img_num=None):
    '''
        读取人头和上半身两个类型数据的image，box
        - outputs:
            data: dict
                data['images']: list
                data['bboxes']: list(list)
    '''

    assert len(label_path_list) == 2

    data = dict()
    images = []
    bboxes = []

    data_dict = defaultdict(int)

    for i, label_path in enumerate(label_path_list):
        """ 对gt txt训练，先人头，后上半身"""

        with open(label_path, 'r') as f:

            lines = f.readlines()

        for line in lines:
            conts = line.strip().split()
            imagepath = osp.join(base_dir, conts[0])

            one_image_bboxes = np.array([float(x) for x in conts[1:]]).reshape(-1, 4).tolist()

            if i == 0:
                data_dict[imagepath] = list((one_image_bboxes, None))
            else:
                if data_dict[imagepath] == 0:
                    data_dict[imagepath] = list((None, one_image_bboxes))
                else:
                    data_dict[imagepath][1] = one_image_bboxes

    keys_list = list(data_dict.keys())

    npr.shuffle(lines)

    if img_num is not None:
        assert img_num <= len(keys_list)
        keys_list = keys_list[:img_num]

    for k in keys_list:
        images.append(k)
        bboxes.append(data_dict[k])

    data['images'] = images
    data['bboxes'] = bboxes
    return data


def convert_to_square(box):
    '''将box转换成更大的正方形
    参数：
      box：预测的box,[n,5]
    返回值：
      调整后的正方形box，[n,5]
    '''
    square_box=box.copy()
    h=box[:,3]-box[:,1]+1
    w=box[:,2]-box[:,0]+1
    #找寻正方形最大边长
    max_side=np.maximum(w,h)
    
    square_box[:,0]=box[:,0]+w*0.5-max_side*0.5
    square_box[:,1]=box[:,1]+h*0.5-max_side*0.5
    square_box[:,2]=square_box[:,0]+max_side-1
    square_box[:,3]=square_box[:,1]+max_side-1
    return square_box

