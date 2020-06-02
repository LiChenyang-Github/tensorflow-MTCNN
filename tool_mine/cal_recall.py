"""
    2020.04.21 @lichenyang
    
    A script to calculate the recall of PNet and RNet.
"""


import sys
sys.path.insert(0, '../preprocess/')
from utils import *
import numpy as np
import argparse
import os
import pickle
import cv2
from tqdm import tqdm
from loader import TestLoader
sys.path.insert(0, '../')
from train.model import P_Net,R_Net,O_Net,P_Net_v1,R_Net_v1,R_Net_fcn,R_Net_fcn_v1
from train.model import P_Net_org,R_Net_org,P_Net_aspect_24_12
import train.config_cal_recall as config
from detection.detector import Detector
from detection.fcn_detector import FcnDetector
from detection.MtcnnDetector import MtcnnDetector

import pdb



def main(args):

    size=args.input_size
    batch_size=config.batches
    min_face_size=config.min_face
    stride=config.stride
    thresh=config.thresh_recall
    filename=args.filename
    base_dir=args.base_dir
    net_name=args.net_name
    iou_threds=args.iou_threds
    thred_stride=args.thred_stride
    img_num=args.img_num
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
        assert isinstance(net_name, list)
        assert len(net_name) == 1 or len(net_name) == 2

        if len(net_name) == 1:
            model_path[0] = '../model_mine/{}/{}/'.format(args.suffix, net_name[0])
            data_dir='../data_mine/{}_{}_{}'.format(save_size, args.suffix, net_name[0])
        elif len(net_name) == 2:
            model_path[0] = '../model_mine/{}/{}/'.format(args.suffix, net_name[0])
            model_path[1] = '../model_mine/{}/{}/'.format(args.suffix, net_name[1])
            data_dir='../data_mine/{}_{}_{}'.format(save_size, args.suffix, net_name[1])

    detectors=[None,None,None]

    if net_name is None:
        PNet=FcnDetector(P_Net,model_path[0])
        if net=='RNet':
            RNet=Detector(R_Net,24,batch_size[1],model_path[1])
            detectors[1]=RNet
    else:
        if len(net_name) == 1:
            PNet=FcnDetector(eval(net_name[0]),model_path[0])
            net = net_name[0]
        elif len(net_name) == 2:
            PNet=FcnDetector(eval(net_name[0]),model_path[0])
            RNet=Detector(eval(net_name[1]),24,batch_size[1],model_path[1])
            detectors[1]=RNet
            net = net_name[1]
    detectors[0]=PNet



    data=read_annotation_xiao(base_dir, filename, img_num=img_num, add_img_suffix=add_img_suffix)


    # pdb.set_trace()

    mtcnn_detector=MtcnnDetector(detectors,min_face_size=min_face_size,
                                stride=stride,threshold=thresh,aspect=aspect)
    # save_path=data_dir
    # save_file=os.path.join(save_path,'detections.pkl')
    save_path = osp.join(
        data_dir, \
        'recall', \
        osp.basename(filename).split('.')[0], \
        net \
        )

    # pdb.set_trace()

    if not osp.isdir(save_path):
        os.makedirs(save_path)

    if size == 12:
        save_det_file = osp.join(save_path, \
            'detections_scores_thred_{}.pkl'.format(thresh[0]))
    elif size == 24:
        save_det_file = osp.join(save_path, \
            'detections_scores_thred_{}-{}.pkl'.format(thresh[0], thresh[1]))
    else:
        raise

    if img_num is not None:
        save_det_file = osp.join(save_path, \
            "{}-imgs_{}".format(img_num, osp.basename(save_det_file)))

    # pdb.set_trace()

    if not os.path.exists(save_det_file):
        #将data制作成迭代器
        print('载入数据')
        test_data=TestLoader(data['images'])
        detectors,_=mtcnn_detector.detect_face(test_data)
        print('完成识别')

        # pdb.set_trace()

        with open(save_det_file,'wb') as f:
            pickle.dump(detectors,f,1)
    print('开始计算recall')


    cal_recall(
        data, \
        save_path, \
        save_det_file, \
        thresh[0] if size in [12] else thresh[1], \
        thred_stride, \
        iou_threds, \
        img_num
        )


def cal_recall(data, save_path, save_det_file, thresh_min, thresh_stride, iou_threds, img_num=None):

    im_idx_list = data['images']
    gt_boxes_list = data['bboxes']
    num_of_images = len(im_idx_list)
    #read detect result
    det_boxes = pickle.load(open(save_det_file, 'rb'))
    assert len(det_boxes) == num_of_images, "图片数量不相等！"

    thresh_num = int((1 - thresh_min) / thresh_stride) + 1

    # pdb.set_trace()

    if img_num is None:
        res_txt = osp.join(save_path, 'res.txt')
    else:
        res_txt = osp.join(save_path, '{}-imgs_res.txt'.format(img_num))

    with open(res_txt, 'w') as f:

        for iou_thred in iou_threds:    # Loop for iou threds
            for i in range(thresh_num): # Loop for score thresh
                thresh_cur = thresh_min+i*thresh_stride
                recall = cal_recall_core(
                    det_boxes, \
                    gt_boxes_list, \
                    thresh_cur, \
                    iou_thred \
                    )
                line = "iou_thred: {}\t, score_thred: {:.3}\t, recall: {}\n".format(iou_thred, \
                    thresh_cur, recall)
                # print(line)
                f.write(line)




def cal_recall_core(det_list, gt_list, score_thred, iou_thred):
    
    assert len(det_list) == len(gt_list)

    gt_num = 0
    correct_num = 0

    for i in range(len(det_list)):
        det_cur = det_list[i]
        gt_cur = gt_list[i]

        if len(det_cur) == 0 or len(gt_cur) == 0: continue  ### 没有检测结果或者没有gt，则continue

        # print(i, det_cur.shape, np.array(gt_cur).shape)
        det_cur = det_cur[np.where(det_cur[:, 4]>score_thred)[0], :4]
        det_num_cur = len(det_cur)
        gt_num_cur = len(gt_cur)
        gt_num += gt_num_cur
        

        # pdb.set_trace()

        iou_mat = calculate_iou(np.array(det_cur), np.array(gt_cur))

        # print(det_num_cur, np.array(det_cur).shape, np.array(gt_cur).shape, iou_mat.shape)

        correct_flag = np.zeros((gt_num_cur, ), dtype=np.float)

        # pdb.set_trace()

        for j in range(det_num_cur):
            max_iou = np.max(iou_mat[j, :])
            # print(max_iou, iou_thred)
            if max_iou > iou_thred:
                correct_flag += (iou_mat[j, :] == max_iou).astype(np.float)

        # pdb.set_trace()

        correct_num_cur = np.sum(correct_flag.astype(np.bool))
        correct_num += correct_num_cur

    recall = correct_num / gt_num

    # pdb.set_trace()

    return recall








def parse_arguments(argv):

    parser = argparse.ArgumentParser()

    parser.add_argument('input_size', type=int,
                        help='The input size for specific net')

    parser.add_argument('--suffix', type=str, default=None,
                        help='The suffix for the folder')

    parser.add_argument('--filename', type=str, default=None,
                        help='The path to the gt txt.')

    parser.add_argument('--base_dir', type=str, default=None,
                        help='The path to the gt txt.')

    parser.add_argument('--net_name', nargs='+', default=None,
                        help='The name for the net.')

    parser.add_argument('--iou_threds', nargs='+', default=[0.65, 0.5, 0.4],
                        help='The iou threshold for recall calculation.')

    parser.add_argument('--thred_stride', type=float, default=0.1, 
                        help='The stride of scores thresh.')

    parser.add_argument('--img_num', type=int, default=None, 
                        help='The input size for specific net')

    parser.add_argument('--add_img_suffix', action='store_true', default=False, 
                        help='The input size for specific net')

    parser.add_argument('--aspect', nargs='+', type=int, default=None,
                        help='Specify the (height, width) when the input is not square.')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))


