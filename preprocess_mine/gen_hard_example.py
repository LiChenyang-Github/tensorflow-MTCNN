
# coding: utf-8

# In[1]:


import sys
sys.path.append('../preprocess/')
from utils import *
import numpy as np
import argparse
import os
import pickle
import cv2
from tqdm import tqdm
from loader import TestLoader
sys.path.append('../')
from train.model import P_Net,R_Net,O_Net,P_Net_v1,R_Net_v1,R_Net_fcn,R_Net_fcn_v1, \
    P_Net_aspect_24_12, P_Net_aspect_18_12

R_Net_aspect_24_12 = R_Net

import train.config as config
from detection.detector import Detector
from detection.fcn_detector import FcnDetector
from detection.MtcnnDetector import MtcnnDetector

import pdb

# In[3]:


def main(args):
    '''通过PNet或RNet生成下一个网络的输入'''
    size=args.input_size
    batch_size=config.batches
    min_face_size=config.min_face
    stride=config.stride
    thresh=config.thresh_hard_examples
    filename=args.filename
    base_dir=args.base_dir
    net_name=args.net_name
    add_img_suffix=args.add_img_suffix
    aspect=args.aspect

    # pdb.set_trace()


    #模型地址
    if args.suffix is None:
        model_path=['../model_mine/PNet/','../model_mine/RNet/','../model_mine/ONet']
    else:
        model_path=['../model_mine/{}/PNet/'.format(args.suffix),'../model_mine/{}/RNet/'.format(args.suffix),'../model_mine/{}/ONet'.format(args.suffix)]

    if size==12:
        net='PNet'
        save_size=24
    elif size==24:
        net='RNet'
        save_size=48

    if args.suffix is None:
        data_dir='../data_mine/%d'%(save_size)
    else:
        data_dir='../data_mine/%d_%s'%(save_size, args.suffix)


    if net_name is not None:
        # assert args.suffix is not None
        assert isinstance(net_name, list)
        assert len(net_name) == 1 or len(net_name) == 2


        # if 'P_Net' in net_name:
        #     model_path[0] = '../model_mine/{}/{}/'.format(args.suffix, net_name)
        # elif 'R_Net' in net_name:
        #     model_path[1] = '../model_mine/{}/{}/'.format(args.suffix, net_name)
        #     model_path[0] = '../model_mine/{}/{}/'.format(args.suffix, net_name.replace('R_Net', 'P_Net'))
        # elif 'O_Net' in net_name:
        #     model_path[2] = '../model_mine/{}/{}/'.format(args.suffix, net_name)
        # else:
        #     raise
        # net = net_name
        # data_dir='../data_mine/{}_{}_{}'.format(save_size, args.suffix, net_name)

        if len(net_name) == 1:
            model_path[0] = '../model_mine/{}/{}/'.format(args.suffix, net_name[0])
            data_dir='../data_mine/{}_{}_{}'.format(save_size, args.suffix, net_name[0])
        elif len(net_name) == 2:
            model_path[0] = '../model_mine/{}/{}/'.format(args.suffix, net_name[0])
            model_path[1] = '../model_mine/{}/{}/'.format(args.suffix, net_name[1])
            data_dir='../data_mine/{}_{}_{}'.format(save_size, args.suffix, net_name[1])


    neg_dir=os.path.join(data_dir,'negative')
    pos_dir=os.path.join(data_dir,'positive')
    part_dir=os.path.join(data_dir,'part')
    for dir_path in [neg_dir,pos_dir,part_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    detectors=[None,None,None]


    if net_name is None:
        PNet=FcnDetector(P_Net,model_path[0])
        if net=='RNet':
            RNet=Detector(R_Net,24,batch_size[1],model_path[1])
            detectors[1]=RNet
    else:
        if len(net_name) == 1:
            PNet=FcnDetector(eval(net_name[0]),model_path[0])
        elif len(net_name) == 2:
            PNet=FcnDetector(eval(net_name[0]),model_path[0])
            RNet=Detector(eval(net_name[1]),24,batch_size[1],model_path[1])
            detectors[1]=RNet
    detectors[0]=PNet

    # pdb.set_trace()


    data=read_annotation_xiao(base_dir, filename, img_num=None, add_img_suffix=add_img_suffix)


    mtcnn_detector=MtcnnDetector(detectors,min_face_size=min_face_size,
                                stride=stride,threshold=thresh,aspect=aspect)
    save_path=data_dir
    save_file=os.path.join(save_path,'detections.pkl')
    if not os.path.exists(save_file):
        #将data制作成迭代器
        print('载入数据')
        test_data=TestLoader(data['images'])
        detectors,_=mtcnn_detector.detect_face(test_data)
        print('完成识别')

        with open(save_file,'wb') as f:
            pickle.dump(detectors,f,1)
    print('开始生成图像')
    # save_hard_example(save_size,data,neg_dir,pos_dir,part_dir,save_path)
    save_hard_example(save_size,data,neg_dir,pos_dir,part_dir,save_path,args.suffix)



# In[2]:


def save_hard_example(save_size, data,neg_dir,pos_dir,part_dir,save_path,suffix=None):
    '''将网络识别的box用来裁剪原图像作为下一个网络的输入'''

    im_idx_list = data['images']
    
    gt_boxes_list = data['bboxes']
    num_of_images = len(im_idx_list)

    # save files
    neg_label_file = osp.join(save_path, "neg_{}.txt".format(save_size))
    neg_file = open(neg_label_file, 'w')

    pos_label_file = osp.join(save_path, "pos_{}.txt".format(save_size))
    pos_file = open(pos_label_file, 'w')

    part_label_file = osp.join(save_path, "part_{}.txt".format(save_size))
    part_file = open(part_label_file, 'w')

    #read detect result
    det_boxes = pickle.load(open(os.path.join(save_path, 'detections.pkl'), 'rb'))
    # print(len(det_boxes), num_of_images)
   
    assert len(det_boxes) == num_of_images, "弄错了"

    
    n_idx = 0
    p_idx = 0
    d_idx = 0
    image_done = 0

    for im_idx, dets, gts in tqdm(zip(im_idx_list, det_boxes, gt_boxes_list)):
        gts = np.array(gts, dtype=np.float32).reshape(-1, 4)
        image_done += 1

        if dets.shape[0] == 0:
            continue
        img = cv2.imread(im_idx)
        #转换成正方形
        dets = convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])
        neg_num = 0
        for box in dets:
            x_left, y_top, x_right, y_bottom, _ = box.astype(int)
            width = x_right - x_left + 1
            height = y_bottom - y_top + 1

            # 除去过小的
            if width < 20 or x_left < 0 or y_top < 0 or x_right > img.shape[1] - 1 or y_bottom > img.shape[0] - 1:
                continue

           
            Iou = IOU(box, gts)
            cropped_im = img[y_top:y_bottom + 1, x_left:x_right + 1, :]
            resized_im = cv2.resize(cropped_im, (save_size, save_size),
                                    interpolation=cv2.INTER_LINEAR)

            #划分种类           
            if np.max(Iou) < 0.3 and neg_num < 60:
                
                save_file = os.path.join(neg_dir, "%s.jpg" % n_idx)
                
                neg_file.write(save_file + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1
                neg_num += 1
            else:
               
                idx = np.argmax(Iou)
                assigned_gt = gts[idx]
                x1, y1, x2, y2 = assigned_gt

                #偏移量
                offset_x1 = (x1 - x_left) / float(width)
                offset_y1 = (y1 - y_top) / float(height)
                offset_x2 = (x2 - x_right) / float(width)
                offset_y2 = (y2 - y_bottom) / float(height)

                # pos和part
                if np.max(Iou) >= 0.65:
                    save_file = os.path.join(pos_dir, "%s.jpg" % p_idx)
                    pos_file.write(save_file + ' 1 %.2f %.2f %.2f %.2f\n' % (
                        offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    p_idx += 1

                elif np.max(Iou) >= 0.4:
                    save_file = os.path.join(part_dir, "%s.jpg" % d_idx)
                    part_file.write(save_file + ' -1 %.2f %.2f %.2f %.2f\n' % (
                        offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    d_idx += 1
    neg_file.close()
    part_file.close()
    pos_file.close()


# In[4]:


def parse_arguments(argv):

    parser = argparse.ArgumentParser()

    parser.add_argument('input_size', type=int,
                        help='The input size for specific net')

    parser.add_argument('--suffix', type=str, default=None,
                        help='The suffix for the folder')

    parser.add_argument('--filename', type=str, default=None,
                        help='The path to the gt txt.')

    parser.add_argument('--base_dir', type=str, default="/disk2/lichenyang/lichenyang/Datasets/fenglian/",
                        help='The path to the gt txt.')

    parser.add_argument('--net_name', nargs='+', default=None,
                        help='The name for the net.')

    parser.add_argument('--add_img_suffix', action='store_true', default=False, 
                        help='The input size for specific net')

    parser.add_argument('--aspect', nargs='+', type=int, default=None,
                        help='Specify the (height, width) when the input is not square.')


    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

