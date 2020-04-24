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



def gen_origin_train_val_txt_xiao_6_scenes():
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


def gen_origin_txt_new_platform():
    """
        将蜂联的标注文件转换成txt格式
    """


    # anno_file_list = ['/disk5/lichenyang/Datasets_backup/fenglian/W9Frap.json', ]
    # image_folder_list = ['/disk5/lichenyang/Datasets_backup/fenglian/W9Frap', ]


    train_all_images = []
    train_all_annos = []
    
    test_all_images = []
    test_all_annos = []
    
    for i in range(len(anno_file_list)):
        train_images, val_images, train_annos, val_annos = aic_format_to_image_based_anno(anno_file_list[i], \
            image_folder_list[i], new_platform=True, val_ratio=0.2)
        # print(train_annos)
        train_all_images.extend(train_images)
        train_all_annos.extend(train_annos)

        test_all_images.extend(val_images)
        test_all_annos.extend(val_annos)

    print(len(train_all_images))
    print(len(test_all_images))

    # pdb.set_trace()

    train_face_txt = "./data_mine/fenglian/fenglian-1_head_train.txt"
    val_face_txt = "./data_mine/fenglian/fenglian-1_head_val.txt"
    train_upperbody_txt = "./data_mine/fenglian/fenglian-1_upperbody_train.txt"
    val_upperbody_txt = "./data_mine/fenglian/fenglian-1_upperbody_val.txt"


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



def gen_aux_train_txt():
    """
        generate original train/val txt for xi_ao auxiliary scenes.
    """

    anno_file_list = [
                    '/disk3/hjy/work/data/aic_upper_head/gym_8.json',
                    '/disk3/hjy/work/data/aic_upper_head/gym_12.json',
                    '/disk3/hjy/work/data/aic_upper_head/subway.json',
                    # '/disk5/lichenyang/Datasets_backup/fenglian/IUG3Ln.json',
                    ]

    image_folder_list = [
                    '/disk3/hjy/work/data/aic_upper_head/gym_8',
                    '/disk3/hjy/work/data/aic_upper_head/gym_12',
                    '/disk3/hjy/work/data/aic_upper_head/subway',
                    # '/disk5/lichenyang/Datasets_backup/fenglian/IUG3Ln'
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

    train_face_txt = "./data_mine/auxiliary/xiao_gym_subway_head.txt"
    # train_face_txt = "./data_mine/fenglian/fenglian-2_head.txt"
    train_upperbody_txt = "./data_mine/auxiliary/xiao_gym_subway_upperbody.txt"
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


    anno_file_list = ['/disk3/hjy/work/data/aic_upper_head/upperBody_annotation_4.json', ]
    image_folder_list = ['/disk3/hjy/work/data/aic_upper_head/upperBody_annotation_4', ]


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



    for i in tqdm(range(len(train_all_images))):

        coords_face = []
        coords_upperbody = []


        img_dir = osp.join(train_all_images[i]['folder'], train_all_images[i]['file_name'])

        ### 过滤掉不存在的图片
        if not osp.exists(img_dir):
            img_not_exist_num += 1
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

    with open(customer_annos_path, 'r') as json_file:
        customer_org_annos =json.load(json_file)


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
        if new_platform:
            # anno['category_id'] = category_id - 2080
            # pdb.set_trace()
            if tag == 'head':
                anno['category_id'] = 1
            elif tag == 'upperBody':
                anno['category_id'] = 2
            else:
                anno['category_id'] = -1
        else:
            # anno['category_id'] = category_id - 55
            if tag == '人头':
                anno['category_id'] = 1
            elif tag == '上半身':
                anno['category_id'] = 2
            else:
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





if __name__ == '__main__':
    # gen_origin_train_val_txt_xiao_6_scenes()

    # gen_origin_txt_new_platform()

    # gen_aux_train_txt()

    # merge_txt()

    # statistic_label_file()

    convert_txt_to_eval_form()

