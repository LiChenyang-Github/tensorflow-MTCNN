
import sys
import cv2
import pdb
import json
import random
random.seed(333)

import os
import shutil
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt

from glob import glob
from tqdm import tqdm
from collections import defaultdict, Counter

import xml.etree.ElementTree as ET
import time

from math import *
from PIL import Image



def gen_origin_train_val_txt_multi_scenes():
    """
        generate original train/val txt for xiao 6 scenes.
    """

    anno_file_list = [#'/disk3/hjy/work/data/aic_upper_head/WFP.json',
                    '/disk3/hjy/work/data/aic_upper_head/head_annotation_2.json',
                    '/disk3/hjy/work/data/aic_upper_head/subway.json',
                    '/disk3/hjy/work/data/aic_upper_head/upperBody_annotation_2.json',
                    '/disk3/hjy/work/data/aic_upper_head/upperBody_annotation_3.json',
                    #'/disk3/hjy/work/data/aic_upper_head/upperBody_annotation_4.json',
                    '/disk3/hjy/work/data/aic_upper_head/zhan_ting_qiu_ji_2.json',
                    '/disk3/hjy/work/data/aic_upper_head/xiao_outside.json']

    image_folder_list = [#'/disk3/hjy/work/data/aic_upper_head/WFP',
                    '/disk3/hjy/work/data/aic_upper_head/head_annotation_2',
                    '/disk3/hjy/work/data/aic_upper_head/subway',
                    '/disk3/hjy/work/data/aic_upper_head/upperBody_annotation_2',
                    '/disk3/hjy/work/data/aic_upper_head/upperBody_annotation_3',
                    #'/disk3/hjy/work/data/aic_upper_head/upperBody_annotation_4',
                    '/disk3/hjy/work/data/aic_upper_head/zhan_ting_qiu_ji_2',
                    '/disk3/hjy/work/data/aic_upper_head/xiao_outside']



    train_all_images = []
    train_all_annos = []
    
    test_all_images = []
    test_all_annos = []
    
    for i in range(len(anno_file_list)):
        train_images, val_images, train_annos, val_annos = aic_format_to_image_based_anno(anno_file_list[i], image_folder_list[i])
        # print(train_annos)
        train_all_images.extend(train_images)
        train_all_annos.extend(train_annos)

        test_all_images.extend(val_images)
        test_all_annos.extend(val_annos)

    print(len(train_all_images))
    print(len(test_all_images))

    # pdb.set_trace()


    train_face_txt = "./data_mine/xiao_6_scenes/xiao_head_train.txt"
    val_face_txt = "./data_mine/xiao_6_scenes/xiao_head_val.txt"
    train_upperbody_txt = "./data_mine/xiao_6_scenes/xiao_upperbody_train.txt"
    val_upperbody_txt = "./data_mine/xiao_6_scenes/xiao_upperbody_val.txt"

    if not osp.exists(osp.dirname(train_face_txt)):
        os.makedirs(osp.dirname(train_face_txt))


    f_train_face = open(train_face_txt, 'w')
    f_val_face = open(val_face_txt, 'w')
    f_train_upperbody = open(train_upperbody_txt, 'w')
    f_val_upperbody = open(val_upperbody_txt, 'w')



    for i in tqdm(range(len(train_all_images))):

        coords_face = []
        coords_upperbody = []


        img_dir = osp.join(train_all_images[i]['folder'], train_all_images[i]['file_name'])

        ### 过滤掉不存在的图片
        if not osp.exists(img_dir):
            continue


        relative_path = osp.basename(train_all_images[i]['folder']) + '/' + train_all_images[i]['file_name']

        # pdb.set_trace()

        anno_list = train_all_annos[i]

        for anno in anno_list:

            ### 检查坐标顺序，确保为x1y1x2y2
            bbox = check_coord_order(anno['bbox'])

            if anno['category_id'] == 1:
                coords_face.extend(bbox)
            elif anno['category_id'] == 2:
                coords_upperbody.extend(bbox)

        if len(coords_face) > 0:
            line = ' '.join([relative_path] + [str(coord) for coord in coords_face]) + '\n'

            f_train_face.write(line)

        if len(coords_upperbody) > 0:
            line = ' '.join([relative_path] + [str(coord) for coord in coords_upperbody]) + '\n'

            f_train_upperbody.write(line)

            # pdb.set_trace()


    for i in tqdm(range(len(test_all_images))):

        coords_face = []
        coords_upperbody = []

        img_dir = osp.join(test_all_images[i]['folder'], test_all_images[i]['file_name'])

        if not osp.exists(img_dir):
            continue

        relative_path = osp.basename(test_all_images[i]['folder']) + '/' + test_all_images[i]['file_name']

        # pdb.set_trace()

        anno_list = test_all_annos[i]

        for anno in anno_list:

            bbox = check_coord_order(anno['bbox'])

            if anno['category_id'] == 1:
                coords_face.extend(bbox)
            elif anno['category_id'] == 2:
                coords_upperbody.extend(bbox)

        if len(coords_face) > 0:
            line = ' '.join([relative_path] + [str(coord) for coord in coords_face]) + '\n'

            f_val_face.write(line)

        if len(coords_upperbody) > 0:
            line = ' '.join([relative_path] + [str(coord) for coord in coords_upperbody]) + '\n'

            f_val_upperbody.write(line)


def gen_origin_train_val_txt_one_scene():
    """
        将蜂联的标注文件转换成txt格式
    """

    upperBody_annotation_4_82_flag = True



    anno_file_list = ['/disk5/lichenyang/Datasets_backup/xiao/upperBody_annotation_4_82.json', ]
    image_folder_list = ['/disk5/lichenyang/Datasets_backup/xiao/upperBody_annotation_4_82', ]

    # anno_file_list = ['/home/LiChenyang/Datasets/xi_ao/upperBody_annotation_4_82.json', ]
    # image_folder_list = ['/home/LiChenyang/Datasets/xi_ao/upperBody_annotation_4_82', ]

    train_all_images = []
    train_all_annos = []
    
    test_all_images = []
    test_all_annos = []
    
    for i in range(len(anno_file_list)):
        train_images, val_images, train_annos, val_annos = aic_format_to_image_based_anno(anno_file_list[i], \
            image_folder_list[i], new_platform=False, val_ratio=0.05)
        # print(train_annos)
        train_all_images.extend(train_images)
        train_all_annos.extend(train_annos)

        test_all_images.extend(val_images)
        test_all_annos.extend(val_annos)

    print(len(train_all_images))
    print(len(test_all_images))

    # pdb.set_trace()

    # train_face_txt = "./data_mine/fenglian/fenglian-1_head_train.txt"
    # val_face_txt = "./data_mine/fenglian/fenglian-1_head_val.txt"
    # train_upperbody_txt = "./data_mine/fenglian/fenglian-1_upperbody_train.txt"
    # val_upperbody_txt = "./data_mine/fenglian/fenglian-1_upperbody_val.txt"
    train_face_txt = "./data_mine/upperBody_annotation_4_82/upperBody_annotation_4_82_head_train.txt"
    val_face_txt = "./data_mine/upperBody_annotation_4_82/upperBody_annotation_4_82_head_val.txt"
    train_upperbody_txt = "./data_mine/upperBody_annotation_4_82/upperBody_annotation_4_82_upperbody_train.txt"
    val_upperbody_txt = "./data_mine/upperBody_annotation_4_82/upperBody_annotation_4_82_upperbody_val.txt"

    f_train_face = open(train_face_txt, 'w')
    f_val_face = open(val_face_txt, 'w')
    f_train_upperbody = open(train_upperbody_txt, 'w')
    f_val_upperbody = open(val_upperbody_txt, 'w')



    for i in tqdm(range(len(train_all_images))):

        coords_face = []
        coords_upperbody = []


        img_dir = osp.join(train_all_images[i]['folder'], train_all_images[i]['file_name'])

        if upperBody_annotation_4_82_flag:
            img_dir = img_dir.replace('!', '%21')   ### for upperBody_annotation_4_82

        ### 过滤掉不存在的图片
        if not osp.exists(img_dir):
            continue


        if upperBody_annotation_4_82_flag:
            relative_path = osp.basename(train_all_images[i]['folder']) + '/' + train_all_images[i]['file_name'].replace('!', '%21')
        else:
            relative_path = osp.basename(train_all_images[i]['folder']) + '/' + train_all_images[i]['file_name']

        # pdb.set_trace()

        anno_list = train_all_annos[i]

        for anno in anno_list:

            ### 检查坐标顺序，确保为x1y1x2y2
            bbox = check_coord_order(anno['bbox'])

            if anno['category_id'] == 1:
                coords_face.extend(bbox)
            elif anno['category_id'] == 2:
                coords_upperbody.extend(bbox)

        if len(coords_face) > 0:
            line = ' '.join([relative_path] + [str(coord) for coord in coords_face]) + '\n'

            f_train_face.write(line)

        if len(coords_upperbody) > 0:
            line = ' '.join([relative_path] + [str(coord) for coord in coords_upperbody]) + '\n'

            f_train_upperbody.write(line)

            # pdb.set_trace()


    for i in tqdm(range(len(test_all_images))):

        coords_face = []
        coords_upperbody = []

        img_dir = osp.join(test_all_images[i]['folder'], test_all_images[i]['file_name'])

        if not osp.exists(img_dir):
            continue

        relative_path = osp.basename(test_all_images[i]['folder']) + '/' + test_all_images[i]['file_name']

        # pdb.set_trace()

        anno_list = test_all_annos[i]

        for anno in anno_list:

            bbox = check_coord_order(anno['bbox'])

            if anno['category_id'] == 1:
                coords_face.extend(bbox)
            elif anno['category_id'] == 2:
                coords_upperbody.extend(bbox)

        if len(coords_face) > 0:
            line = ' '.join([relative_path] + [str(coord) for coord in coords_face]) + '\n'

            f_val_face.write(line)

        if len(coords_upperbody) > 0:
            line = ' '.join([relative_path] + [str(coord) for coord in coords_upperbody]) + '\n'

            f_val_upperbody.write(line)



def gen_aux_train_txt():
    """
        generate original train/val txt for xi_ao auxiliary scenes.
    """

    anno_file_list = [
                    # '/disk3/hjy/work/data/aic_upper_head/gym_8.json',
                    # '/disk3/hjy/work/data/aic_upper_head/gym_12.json',
                    # '/disk3/hjy/work/data/aic_upper_head/subway.json',
                    # '/disk5/lichenyang/Datasets_backup/fenglian/IUG3Ln.json',
                    ]

    image_folder_list = [
                    # '/disk3/hjy/work/data/aic_upper_head/gym_8',
                    # '/disk3/hjy/work/data/aic_upper_head/gym_12',
                    # '/disk3/hjy/work/data/aic_upper_head/subway',
                    # '/disk5/lichenyang/Datasets_backup/fenglian/IUG3Ln', 
                    ]



    train_all_images = []
    train_all_annos = []

    test_all_images = []
    test_all_annos = []
    
    for i in range(len(anno_file_list)):
        train_images, val_images, train_annos, val_annos = \
            aic_format_to_image_based_anno(anno_file_list[i], image_folder_list[i], \
                new_platform=False, val_ratio=0.)
        # print(train_annos)
        train_all_images.extend(train_images)
        train_all_annos.extend(train_annos)

        test_all_images.extend(val_images)
        test_all_annos.extend(val_annos)

    print(len(train_all_images))
    print(len(test_all_images))

    # pdb.set_trace()

    # train_face_txt = "./data_mine/auxiliary/xiao_gym_subway_head.txt"
    # train_face_txt = "./data_mine/fenglian/fenglian-2_head.txt"

    # train_upperbody_txt = "./data_mine/auxiliary/xiao_gym_subway_upperbody.txt"
    # train_upperbody_txt = "./data_mine/fenglian/fenglian-2_upperbody.txt"

    f_train_face = open(train_face_txt, 'w')
    # f_val_face = open(val_face_txt, 'w')
    f_train_upperbody = open(train_upperbody_txt, 'w')
    # f_val_upperbody = open(val_upperbody_txt, 'w')

    for i in tqdm(range(len(train_all_images))):

        coords_face = []
        coords_upperbody = []


        img_dir = osp.join(train_all_images[i]['folder'], train_all_images[i]['file_name'])

        ### 过滤掉不存在的图片
        if not osp.exists(img_dir):
            continue

        ### 过滤掉存在但是大小为0的图片
        if is_empty_img(img_dir):
            continue


        ### 过滤掉灰度图
        if is_grey_img(img_dir):
            continue


        relative_path = osp.basename(train_all_images[i]['folder']) + '/' + train_all_images[i]['file_name']

        # pdb.set_trace()

        anno_list = train_all_annos[i]

        for anno in anno_list:

            ### 检查坐标顺序，确保为x1y1x2y2
            bbox = check_coord_order(anno['bbox'])

            if anno['category_id'] == 1:
                coords_face.extend(bbox)
            elif anno['category_id'] == 2:
                coords_upperbody.extend(bbox)

        if len(coords_face) > 0:
            line = ' '.join([relative_path] + [str(coord) for coord in coords_face]) + '\n'

            f_train_face.write(line)

        if len(coords_upperbody) > 0:
            line = ' '.join([relative_path] + [str(coord) for coord in coords_upperbody]) + '\n'

            f_train_upperbody.write(line)

            # pdb.set_trace()


    for i in tqdm(range(len(test_all_images))):

        coords_face = []
        coords_upperbody = []

        img_dir = osp.join(test_all_images[i]['folder'], test_all_images[i]['file_name'])

        if not osp.exists(img_dir):
            continue

        relative_path = osp.basename(test_all_images[i]['folder']) + '/' + test_all_images[i]['file_name']

        # pdb.set_trace()

        anno_list = test_all_annos[i]

        for anno in anno_list:

            bbox = check_coord_order(anno['bbox'])

            if anno['category_id'] == 1:
                coords_face.extend(bbox)
            elif anno['category_id'] == 2:
                coords_upperbody.extend(bbox)

        if len(coords_face) > 0:
            line = ' '.join([relative_path] + [str(coord) for coord in coords_face]) + '\n'

            f_val_face.write(line)

        if len(coords_upperbody) > 0:
            line = ' '.join([relative_path] + [str(coord) for coord in coords_upperbody]) + '\n'

            f_val_upperbody.write(line)



def statistic_label_file():
    """
        统计从标注平台下载的文件的相关信息。
    """

    anno_file_list = ['D:/Dataset/xi_ao/upperBody_annotation_4_82.json', ]
    image_folder_list = ['E:/Datasets/xiao_head_upperbody/upperBody_annotation_4_82', ]

    # anno_file_list = ['D:/Dataset/xi_ao/upperBody_annotation_4.json', ]
    # image_folder_list = ['E:/Datasets/xiao_head_upperbody/upperBody_annotation_4', ]

    # anno_file_list = ['E:/Datasets/fenglian_head_upperbody/W9Frap.json', ]
    # image_folder_list = ['E:/Datasets/fenglian_head_upperbody/W9Frap', ]

    train_all_images = []
    train_all_annos = []
    
    test_all_images = []
    test_all_annos = []
    
    for i in range(len(anno_file_list)):
        train_images, val_images, train_annos, val_annos = aic_format_to_image_based_anno(anno_file_list[i], \
            image_folder_list[i], new_platform=False, val_ratio=0.)
        # print(train_annos)
        train_all_images.extend(train_images)
        train_all_annos.extend(train_annos)

        test_all_images.extend(val_images)
        test_all_annos.extend(val_annos)

    print(len(train_all_images))
    print(len(test_all_images))



    # pdb.set_trace()

    # train_face_txt = "./data_mine/fenglian/fenglian-1_head_train.txt"
    # val_face_txt = "./data_mine/fenglian/fenglian-1_head_val.txt"
    # train_upperbody_txt = "./data_mine/fenglian/fenglian-1_upperbody_train.txt"
    # val_upperbody_txt = "./data_mine/fenglian/fenglian-1_upperbody_val.txt"


    # f_train_face = open(train_face_txt, 'w')
    # f_val_face = open(val_face_txt, 'w')
    # f_train_upperbody = open(train_upperbody_txt, 'w')
    # f_val_upperbody = open(val_upperbody_txt, 'w')


    valid_img_num = 0
    head_bbox_num = 0
    upperbody_bbox_num = 0
    img_not_exist_num = 0
    imgs_not_exist = []



    for i in tqdm(range(len(train_all_images))):

        coords_face = []
        coords_upperbody = []


        img_dir = osp.join(train_all_images[i]['folder'], train_all_images[i]['file_name'])

        img_dir = img_dir.replace('!', '%21')

        ### 过滤掉不存在的图片
        if not osp.exists(img_dir):
            img_not_exist_num += 1
            imgs_not_exist.append(train_all_images[i]['file_name'])
            continue


        relative_path = osp.basename(train_all_images[i]['folder']) + '/' + train_all_images[i]['file_name']

        # pdb.set_trace()

        anno_list = train_all_annos[i]

        for anno in anno_list:

            ### 检查坐标顺序，确保为x1y1x2y2
            bbox = check_coord_order(anno['bbox'])

            if anno['category_id'] == 1:
                coords_face.extend(bbox)
                head_bbox_num += 1
            elif anno['category_id'] == 2:
                coords_upperbody.extend(bbox)
                upperbody_bbox_num += 1

        valid_img_num += 1


    print("img_num: {}. img_not_exist_num: {}. head bbox num: {}. upperbody bbox num {}.".format(
        valid_img_num, img_not_exist_num, head_bbox_num, upperbody_bbox_num))

    # print(imgs_not_exist[:10])




def is_grey_img(img_dir):

    img = cv2.imread(img_dir)

    # print(img_dir, img is None)

    if (img[:,:,0]==img[:,:,1]).all() and (img[:,:,0]==img[:,:,2]).all():
        return True
    else:
        return False

def is_empty_img(img_dir):
    
    img = cv2.imread(img_dir)

    if img is None:
        return True
    else:
        return False


def merge_txt():

    src_txt_1_dir = "/disk5/lichenyang/Projects/tensorflow-MTCNN_quanzhi_8P100_v1/data_mine/fenglian/fenglian-1_head_train.txt"
    src_txt_2_dir = "/disk5/lichenyang/Projects/tensorflow-MTCNN_quanzhi_8P100_v1/data_mine/fenglian/fenglian-2_head.txt"

    dst_txt_dir = "/disk5/lichenyang/Projects/tensorflow-MTCNN_quanzhi_8P100_v1/data_mine/fenglian/fenglian-1-2_head_train.txt"

    cmd_str_1 = "cat {} >> {}".format(src_txt_1_dir, dst_txt_dir)
    cmd_str_2 = "cat {} >> {}".format(src_txt_2_dir, dst_txt_dir)

    os.system(cmd_str_1)
    os.system(cmd_str_2)


def aic_format_to_image_based_anno(customer_annos_path, customer_image_folder, \
    new_platform=False, val_ratio=0.1):
    """
        蜂联的数据是在新的标注平台上标注的，所以 new_platform 设为True，新旧标注平台bbox的标注格式有一些不太一样的地方。
    """

    images = []
    categories = []

    categories_based_images = {}
    image_based_annos = {}

    with open(customer_annos_path, 'r', encoding='UTF-8') as json_file:

        # lines = json_file.readlines()

        # print(lines[0])

        # configstr = json_file.read().replace('\\', '\\\\')

        # customer_org_annos =json.loads(configstr, strict=False)

        configstr = json_file.read()
        configstr = configstr[:52508429-194] + configstr[52508429+766:]
        customer_org_annos = json.loads(configstr)

        # customer_org_annos =json.load(json_file)



    print('processing annotation')
    idx = 0
    for aid in tqdm(range(len(customer_org_annos))):
        customer_org_anno = customer_org_annos[aid]

        mark_type = customer_org_anno['markType']

        if 'imageId' in customer_org_anno:
            img_name = customer_org_anno['imageId']
        elif 'imageName' in customer_org_anno:
            img_name = customer_org_anno['imageName']
        else:
            print('No imageId or imageName in anno!')
            continue

        if 'tag' in customer_org_anno:
            tag = customer_org_anno['tag']
        elif 'labelName' in customer_org_anno:
            tag = customer_org_anno['labelName']
        else:
            print('No tag or labelName in anno!')
            continue


        # tag = customer_org_anno['tag']
        category_id = customer_org_anno['labelId']
        anno = dict()

        # print(category_id)
        # pdb.set_trace()

        # assert category_id in [56, 57, 2081, 2082]

        md = json.loads(customer_org_anno['markDetail'])

        # 目前只支持标注框类型
        if mark_type.lower() == 'rect':
            if new_platform:
                assert len(md['children']) == 1
                left= md['children'][0]['left']
                top =md['children'][0]['top']
                width =md['children'][0]['width']
                height=md['children'][0]['height']
                x1 = left
                y1 = top
                x2 = left + width
                y2 = top + height

            else:       
                x1 = float(md['xfrom'])
                y1 = float(md['yfrom'])
                x2 = float(md['xto'])
                y2 = float(md['yto'])

        else:
            continue

        anno['id'] = idx
        anno['bbox'] = [x1, y1, x2, y2]
        anno['iscrowd'] = 0
        if new_platform:    # 82 platform
            # anno['category_id'] = category_id - 2080
            # pdb.set_trace()
            if tag == 'head':
                anno['category_id'] = 1
            elif tag == 'upperBody':
                anno['category_id'] = 2
            else:
                anno['category_id'] = -1
        else:   # 80 platform
            # anno['category_id'] = category_id - 55
            if tag == '人头':
                anno['category_id'] = 1
            elif tag == '上半身':
                # print(tag)
                anno['category_id'] = 2
            else:
                # print(tag)
                anno['category_id'] = -1

            # pdb.set_trace()

        if anno['category_id'] == 1:
            anno['category_name'] = 'head'
        elif anno['category_id'] == 2 :
            anno['category_name'] = 'upperBody'
        else:
            continue   

        if img_name not in image_based_annos.keys():
            image  = {}
            # img = cv2.imread(os.path.join(customer_image_folder, img_name).rstrip())
            # if img is None:
            #     print(os.path.join(customer_image_folder, img_name).rstrip())
            #     continue

            # cv2.rectangle(img, (240, 0), (480, 375), (0, 255, 0), 2)

            # height, width = img.shape[0], img.shape[1]
            image['license'] = 3
            image['file_name'] = img_name
            image['folder'] = customer_image_folder
            image['coco_url'] = ''
            # image['width'], image['height'] = width, height
            image['flickr_url'] = ''
            image['id'] = len(image_based_annos.keys())
            anno['image_id'] = image['id']

            images.append(image)
            image_based_annos[img_name] = []
            image_based_annos[img_name].append(anno)
        else:
            anno['image_id'] = image_based_annos[img_name][0]['image_id']
            image_based_annos[img_name].append(anno)
        idx += 1

    # pdb.set_trace()

    # split train test
    # if split_trainval:
    #     return sample_obj(images, image_based_annos, val_ratio)
    # else:
    #     return sample_obj(images, image_based_annos, 0.)

    return sample_obj(images, image_based_annos, val_ratio)

# 划分train val数据集
# 目前是根据images 随机抽取一定比例 后面改进
def sample_obj(images, image_based_annos, ratio=0.1):
    train_annos = []
    val_annos = []
    train_images = []
    val_images = []

    train_images = []
    val_images = []

    random.shuffle(images)
    train_count = int(len(images) * (1-ratio))
    train_images = images[0: train_count]
    val_images = images[train_count:]


    for image in train_images:
        train_annos.append(image_based_annos[image['file_name']])
    
    for image in val_images:
        val_annos.append(image_based_annos[image['file_name']])


    return train_images, val_images, train_annos, val_annos


def check_coord_order(bbox):

    res = bbox.copy()

    if bbox[0] > bbox[2]:
        res[0] = bbox[2]
        res[2] = bbox[0]

    if bbox[1] > bbox[3]:
        res[1] = bbox[3]
        res[3] = bbox[1]

    return res




def convert_txt_to_eval_form():
    """
        Convert the txt into evaluation form.
    """

    src_txt_dir = "./data_mine/fenglian/fenglian-1_upperbody_val.txt"
    dst_txt_root = "./data_mine/fenglian/eval_form/fenglian-1_upperbody_val/"
    obj_name = "upperbody"

    if not osp.isdir(dst_txt_root):
        os.makedirs(dst_txt_root)

    with open(src_txt_dir, 'r') as f:
        lines = f.readlines()

    for line in lines:
        conts = line.strip().split()
        img_name = osp.basename(conts[0])
        coords = [float(x) for x in conts[1:]]

        txt_name = img_name[:-4] + '.txt'
        dst_txt_dir = osp.join(dst_txt_root, txt_name)
        
        with open(dst_txt_dir, 'w') as f:

            for i in range(len(coords) // 4):
                dst_txt_line = ' '.join([obj_name] + [str(int(x)) for x in coords[i*4:(i+1)*4]]) + '\n'

                f.write(dst_txt_line)


def read_json():

    # json_dir = "/home/LiChenyang/Datasets/xi_ao/upperBody_annotation_4_82.json"
    # json_dir = "/home/LiChenyang/Datasets/xi_ao/WFP.json"

    json_dir = "D:/Dataset/xi_ao/gym_12.json"




    with open(json_dir, 'r', encoding='UTF-8') as f:

        # pdb.set_trace()
        # configstr = f.read()
        # configstr = configstr[:52508429] + "6:08\\" + configstr[52508429+5:]
        # configstr = configstr[:52508429] + "6:08\\" + configstr[52508429+5:]
        # configstr = configstr[:52508429-194] + configstr[52508429+766:]
        # configstr = configstr[:52508429-194] + configstr[52508429+766:]

        # pdb.set_trace()
        # annos = json.loads(configstr)

        annos = json.load(f)

    # pdb.set_trace()

    label_id_set = set()
    cls_name_set = set()

    count = 0

    for anno in annos:

        label_id_set.add(anno['labelId'])
        cls_name_set.add(anno['tag'])

        if anno['tag'] == "上半身":
            count += 1



    print(label_id_set)
    print(cls_name_set)
    print(count)



def statistic_txt_file():

    # txt_dir = "./data/wider_face_train.txt"
    txt_dir = "./data_mine/upperBody_annotation_4_82/upperBody_annotation_4_82_upperbody_train.txt"

    img_num = 0
    bbox_num = 0

    with open(txt_dir, 'r') as f:

        lines = f.readlines()

        img_num = len(lines)

        for line in lines:
            line = line.strip().split()
            bbox_num_cur = (len(line) - 1) // 4
            bbox_num += bbox_num_cur


    print("Img num: {}. Bbox num: {}.".format(img_num, bbox_num))



def filterCoordinate(c,m):
    if c < 0:
        return 0
    elif c > m:
        return m
    else:
        return c


def convert_ellipse_to_rect():
    """
        将FDDB数据集的椭圆标签转换为矩形标签
        参考repo: https://github.com/ankanbansal/fddb-for-yolo/blob/master/convertEllipseToRect.py
    """

    ellipse_filename_format = '/home/LiChenyang/Datasets/FDDB/FDDB-folds/FDDB-fold-{:0>2}-ellipseList.txt'
    rect_filename_format = '/home/LiChenyang/Datasets/FDDB/FDDB-folds/FDDB-fold-{:0>2}-rectList.txt'

    for fold_id in range(1, 11):

        ellipse_filename = ellipse_filename_format.format(fold_id)
        rect_filename = rect_filename_format.format(fold_id)

        with open(ellipse_filename) as f:
            lines = [line.rstrip('\n') for line in f]

        f = open(rect_filename,'w')
        i = 0
        while i < len(lines):
            img_file = '/home/LiChenyang/Datasets/FDDB/images/' + lines[i] + '.jpg'
            img = Image.open(img_file)
            w = img.size[0]
            h = img.size[1]
            num_faces = int(lines[i+1])
            for j in range(num_faces):
                ellipse = lines[i+2+j].split()[0:5]
                a = float(ellipse[0])
                b = float(ellipse[1])
                angle = float(ellipse[2])
                centre_x = float(ellipse[3])
                centre_y = float(ellipse[4])
                
                tan_t = -(b/a)*tan(angle)
                t = atan(tan_t)
                x1 = centre_x + (a*cos(t)*cos(angle) - b*sin(t)*sin(angle))
                x2 = centre_x + (a*cos(t+pi)*cos(angle) - b*sin(t+pi)*sin(angle))
                x_max = filterCoordinate(max(x1,x2),w)
                x_min = filterCoordinate(min(x1,x2),w)
                
                if tan(angle) != 0:
                    tan_t = (b/a)*(1/tan(angle))
                else:
                    tan_t = (b/a)*(1/(tan(angle)+0.0001))
                t = atan(tan_t)
                y1 = centre_y + (b*sin(t)*cos(angle) + a*cos(t)*sin(angle))
                y2 = centre_y + (b*sin(t+pi)*cos(angle) + a*cos(t+pi)*sin(angle))
                y_max = filterCoordinate(max(y1,y2),h)
                y_min = filterCoordinate(min(y1,y2),h)
            
                text = img_file + ',' + str(x_min) + ',' + str(y_min) + ',' + str(x_max) + ',' + str(y_max) + '\n'
                f.write(text)

            i = i + num_faces + 2

        f.close()


def fddb_label_to_mtcnn_txt_form():

    src_txt_root = "/home/LiChenyang/Datasets/FDDB/FDDB-folds/"
    dst_txt_dir = "/home/LiChenyang/Projects/tensorflow-MTCNN/data_mine/FDDB/FDDB.txt"
    src_txt_dirs = glob(osp.join(src_txt_root, "FDDB-fold-*-rectList.txt"))

    dst_dict = defaultdict(list)

    for src_txt_dir in src_txt_dirs:

        with open(src_txt_dir, 'r') as f_r:
            src_lines = f_r.readlines()

            for src_line in src_lines:

                src_line_conts = src_line.strip().split(',')
                relative_path = '/'.join(src_line_conts[0].split('/')[-5:])
                # dst_line = ' '.join([relative_path] +  src_line_conts[1:]) + '\n'
                # pdb.set_trace()
                # print(dst_line)

                dst_dict[relative_path].extend(src_line_conts[1:])

    # pdb.set_trace()

    with open(dst_txt_dir, 'w') as f_w:

        for k in dst_dict.keys():

            dst_line = ' '.join([k] + dst_dict[k]) + '\n'

            f_w.write(dst_line)



def vis_imgs_txt():

    src_img_root = "/home/LiChenyang/Datasets/FDDB/images/"
    dst_img_root = "./output_mine/fddb_gt_bbox"
    txt_dir = "./data_mine/FDDB/FDDB.txt"
    vis_num = 10

    if not osp.isdir(dst_img_root):
        os.makedirs(dst_img_root)

    with open(txt_dir, 'r') as f:

        lines = f.readlines()
        random.shuffle(lines)

    assert vis_num <= len(lines)

    for line in lines[:vis_num]:

        line_conts = line.strip().split()

        src_img_dir = osp.join(src_img_root, line_conts[0])
        dst_img_dir = osp.join(dst_img_root, osp.basename(line_conts[0]))
        coords = [int(float(x)) for x in line_conts[1:]]

        img = cv2.imread(src_img_dir)

        for i in range(len(coords)//4):
            cv2.rectangle(img, (coords[i*4], coords[i*4+1]), (coords[i*4+2], coords[i*4+3]), (0, 255, 0), 2)


        cv2.imwrite(dst_img_dir, img)




def read_imgs_txt():

    src_img_root = "/home/LiChenyang/Datasets/xi_ao/"
    txt_dir = "./data_mine/upperBody_annotation_4_82/upperBody_annotation_4_82_upperbody_train.txt"

    jpg_num = 0
    png_num = 0

    with open(txt_dir, 'r') as f:

        lines = f.readlines()

    for line in lines:
        relative_path = line.strip().split()[0]

        img_dir = osp.join(src_img_root, relative_path)

        if img_dir[-3:] == 'jpg':
            jpg_num += 1
        elif img_dir[-3:] == 'png':
            png_num += 1
        else:
            raise

        print(img_dir)
        im = cv2.imread(img_dir)

        # try:
        #     im = cv2.imread(img_dir)
        # except IOError:
        #     print(img_dir)

        # try:
        #     img = Image.open(img_dir)
        # except IOError:
        #     print(img_dir)

        # try:
        #     img= np.array(img, dtype=np.float32)
        # except :
        #     print('corrupt img',absolute_path)

    print(jpg_num, png_num)


def read_imgs_folder():

    # imgs_root = "D:/Code/python/tensorflow-MTCNN_quanzhi_8P100_v1/output_mine/upperbody_4_corrupt/"
    imgs_root = "/home/LiChenyang/Datasets/xi_ao/corrupt_img_1/"


    img_dirs = glob(osp.join(imgs_root, '*'))

    for img_dir in img_dirs:

        im = cv2.imread(img_dir)

        print(img_dir, im.shape)



def load_odgt(file_dir):

    # file_dir = "D:/BaiduNetdiskDownload/annotation_train.odgt"
    annos_dict = {}

    with open(file_dir, 'r') as f:
        lines = f.readlines()

    for line in lines:

        line_json = json.loads(line.strip())

        annos_dict[line_json['ID']] = line_json

    # pdb.set_trace()

    # print(annos_dict["273271,c9db000d5146c15"], len(annos_dict))

    return annos_dict


def crowdhuman_draw(img, anno_dict):

    gt_boxes = anno_dict['gtboxes']

    for instance in gt_boxes:

        hbox = instance['hbox']
        vbox = instance['vbox']
        fbox = instance['fbox']


        if instance['tag'] == 'mask':
            cv2.rectangle(img, (hbox[0], hbox[1]), (hbox[0]+hbox[2], hbox[1]+hbox[3]), (0, 0, 0), 1)
        elif instance['tag'] == 'person':
            if instance['head_attr']['ignore'] == 1 or instance['head_attr']['occ'] == 1:
                cv2.rectangle(img, (hbox[0], hbox[1]), (hbox[0]+hbox[2], hbox[1]+hbox[3]), (0, 255, 255), 1)
            else:
                cv2.rectangle(img, (hbox[0], hbox[1]), (hbox[0]+hbox[2], hbox[1]+hbox[3]), (255, 0, 0), 1)
            cv2.rectangle(img, (vbox[0], vbox[1]), (vbox[0]+vbox[2], vbox[1]+vbox[3]), (0, 255, 0), 1)
            cv2.rectangle(img, (fbox[0], fbox[1]), (fbox[0]+fbox[2], fbox[1]+fbox[3]), (0, 0, 255), 1)
            cv2.putText(img, '{}'.format(instance['extra']['box_id']), (hbox[0], hbox[1]-2), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (125, 0, 0), 1)
        else:
            raise

    return img


def vis_crowdhuman():

    img_root = "D:/BaiduNetdiskDownload/"
    odgt_dir = "D:/BaiduNetdiskDownload/annotation_train.odgt"
    # img_id = "273275,cd061000af95f691"
    # img_id = "273275,59b50000f22dfcf2"
    # img_id = "273278,cac480005a9d945f"

    # ig 1, occ 1   为比较严重的遮挡，可以忽略的
    # img_id = "273275,13ad450003c347e49"
    # img_id = "273275,ab13900012ddf050"

    # ig 0, occ 1   不是很严重的遮挡 (但是有一些感觉也挺严重的)，身体部分可见，可有一些可能是侧脸或者背面 
    # img_id = "273275,13ad450003c347e49"
    # img_id = "273275,ab13900012ddf050"

    # ig 1, occ 0 
    # img_id = "273278,cfab500071258472"
    # img_id = "282555,ebde6000591f3995"




    img_dirs = glob(osp.join(img_root, '*', '*', '{}.*'.format(img_id)))

    if len(img_dirs) != 1:
        return
    else:
        img_dir = img_dirs[0]

    # print(img_dir)

    annos = load_odgt(odgt_dir)


    img = cv2.imread(img_dir)
    img_src = img.copy()


    img = crowdhuman_draw(img, annos[img_id])

    cv2.imshow("src", img_src)
    cv2.imshow("vis", img)

    cv2.waitKey()






if __name__ == '__main__':
    # gen_origin_train_val_txt_multi_scenes()

    # gen_origin_train_val_txt_one_scene()

    # gen_aux_train_txt()

    # merge_txt()

    # statistic_label_file()

    # convert_txt_to_eval_form()

    # read_json()

    # statistic_txt_file()

    # convert_ellipse_to_rect()

    # fddb_label_to_mtcnn_txt_form()

    # vis_imgs_txt()

    # read_imgs_txt()

    # read_imgs_folder()

    # load_odgt()

    vis_crowdhuman()

