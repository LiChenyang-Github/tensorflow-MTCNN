
# coding: utf-8

# In[1]:


'''
截取pos，neg,part三种类型图片并resize成12x12大小作为PNet的输入
'''
import os
import cv2
import sys
import numpy as np
npr=np.random
sys.path.append('../preprocess/')
from tqdm import  tqdm
from utils import IOU 

import pdb
import argparse

from glob import glob



def main(args):

    suffix = args.suffix
    anno_file = args.anno_file
    im_root = args.im_root
    add_img_suffix = args.add_img_suffix
    crop_width = args.crop_width
    crop_height = args.crop_height
    aspect_ratio = crop_height / crop_width

    # pdb.set_trace()


    if suffix is None:
        pos_save_dir='../data_mine/12/positive'
        part_save_dir='../data_mine/12/part'
        neg_save_dir='../data_mine/12/negative'
        save_dir='../data_mine/12'
    else:
        pos_save_dir='../data_mine/12_{}/positive'.format(suffix)
        part_save_dir='../data_mine/12_{}/part'.format(suffix)
        neg_save_dir='../data_mine/12_{}/negative'.format(suffix)
        save_dir='../data_mine/12_{}'.format(suffix)


    ###
    SEED = 333
    npr.seed(SEED)

    # In[2]:


    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(pos_save_dir):
        os.mkdir(pos_save_dir)
    if not os.path.exists(part_save_dir):
        os.mkdir(part_save_dir)
    if not os.path.exists(neg_save_dir):
        os.mkdir(neg_save_dir)
        
    f1=open(os.path.join(save_dir,'pos_12.txt'),'w')
    f2=open(os.path.join(save_dir,'neg_12.txt'),'w')
    f3=open(os.path.join(save_dir,'part_12.txt'),'w')

    with open(anno_file,'r') as f:
        annotations=f.readlines()
    num=len(annotations)
    print('总共的图片数： %d' % num)
    #记录pos,neg,part三类生成数
    p_idx=0
    n_idx=0
    d_idx=0
    #记录读取图片数
    idx=0
    for annotation in tqdm(annotations):
        annotation=annotation.strip().split(' ')
        im_path=annotation[0]
        box=list(map(float,annotation[1:]))
        
        boxes=np.array(box,dtype=np.float32).reshape(-1,4)
        
        if not add_img_suffix:
            img_dir = os.path.join(im_root,im_path)
        else:
            img_dir = glob(os.path.join(im_root,im_path+".*"))[0]
        img=cv2.imread(img_dir)
        idx+=1
        height,width,channel=img.shape
        
        neg_num=0
        #先采样一定数量neg图片
        # while neg_num<50:
        while neg_num<200:


            #随机选取截取图像大小
            # size=npr.randint(12,min(width,height)/2)
            size_w = npr.randint(12,min(width,height)/2)
            size_h = int(size_w * aspect_ratio)


            #随机选取左上坐标
            # nx=npr.randint(0,width-size)
            # ny=npr.randint(0,height-size)
            nx=npr.randint(0,width-size_w)
            ny=npr.randint(0,height-size_h)

            #截取box
            # crop_box=np.array([nx,ny,nx+size,ny+size])
            crop_box=np.array([nx,ny,nx+size_w,ny+size_h])

            #计算iou值
            Iou=IOU(crop_box,boxes)
            #截取图片并resize成crop_widthxcrop_height大小
            # cropped_im=img[ny:ny+size,nx:nx+size,:]
            # resized_im=cv2.resize(cropped_im,(12,12),interpolation=cv2.INTER_LINEAR)

            
            #iou值小于0.3判定为neg图像
            if np.max(Iou)<0.3:
                cropped_im=img[ny:ny+size_h,nx:nx+size_w,:]
                resized_im=cv2.resize(cropped_im,(crop_width,crop_height),interpolation=cv2.INTER_LINEAR)

                # pdb.set_trace()

                save_file=os.path.join(neg_save_dir,'%s.jpg'%n_idx)
                f2.write(neg_save_dir+'/%s.jpg'%n_idx+' 0\n')
                cv2.imwrite(save_file,resized_im)
                n_idx+=1
                neg_num+=1

        for box in boxes:
            ### 截取更加靠近gt bboxes的neg框
            #左上右下坐标
            x1,y1,x2,y2=box
            w=x2-x1+1
            h=y2-y1+1
            #舍去图像过小和box在图片外的图像
            if max(w,h)<20 or x1<0 or y1<0:
                continue
            # for i in range(5):
            for i in range(20):
                # size=npr.randint(12,min(width,height)/2)
                size_w = npr.randint(12,min(width,height)/2)
                size_h = int(size_w * aspect_ratio)

                #随机生成的关于x1,y1的偏移量，并且保证x1+delta_x>0,y1+delta_y>0
                # print(x1,y1,x2,y2,size)
                # delta_x=npr.randint(max(-size,-x1),w)
                # delta_y=npr.randint(max(-size,-y1),h)
                delta_x=npr.randint(max(-size_w,-x1),w)
                delta_y=npr.randint(max(-size_h,-y1),h)

                #截取后的左上角坐标
                nx1=int(max(0,x1+delta_x))
                ny1=int(max(0,y1+delta_y))
                #排除大于图片尺度的
                # if nx1+size>width or ny1+size>height:
                #     continue
                if nx1+size_w>width or ny1+size_h>height:
                    continue
                # crop_box=np.array([nx1,ny1,nx1+size,ny1+size])
                crop_box=np.array([nx1,ny1,nx1+size_w,ny1+size_h])

                Iou=IOU(crop_box,boxes)
                # cropped_im=img[ny1:ny1+size,nx1:nx1+size,:]
                # resized_im=cv2.resize(cropped_im,(12,12),interpolation=cv2.INTER_LINEAR)
                
                if np.max(Iou)<0.3:
                    cropped_im=img[ny1:ny1+size_h,nx1:nx1+size_w,:]
                    resized_im=cv2.resize(cropped_im,(crop_width,crop_height),interpolation=cv2.INTER_LINEAR)

                    save_file=os.path.join(neg_save_dir,'%s.jpg'%n_idx)
                    f2.write(neg_save_dir+'/%s.jpg'%n_idx+' 0\n')
                    cv2.imwrite(save_file,resized_im)
                    n_idx+=1
            for i in range(20):
                #缩小随机选取size范围，更多截取pos和part图像
                # size=npr.randint(int(min(w,h)*0.8),np.ceil(1.25*max(w,h)))
                size_w=npr.randint(int(min(w,h)*0.8),np.ceil(1.25*min(w,h)))
                size_h = int(size_w * aspect_ratio)

                #除去尺度小的
                if w<5:
                    continue
                #偏移量，范围缩小了
                delta_x=npr.randint(-w*0.2,w*0.2)
                delta_y=npr.randint(-h*0.2,h*0.2)
                #截取图像左上坐标计算是先计算x1+w/2表示的中心坐标，再+delta_x偏移量，再-size/2，
                #变成新的左上坐标
                # nx1=int(max(x1+w/2+delta_x-size/2,0))
                # ny1=int(max(y1+h/2+delta_y-size/2,0))
                nx1=int(max(x1+w/2+delta_x-size_w/2,0))
                ny1=int(max(y1+h/2+delta_y-size_h/2,0))

                # nx2=nx1+size
                # ny2=ny1+size
                nx2=nx1+size_w
                ny2=ny1+size_h
                
                #排除超出的图像
                if nx2>width or ny2>height:
                    continue
                crop_box=np.array([nx1,ny1,nx2,ny2])
                #人脸框相对于截取图片的偏移量并做归一化处理
                # offset_x1=(x1-nx1)/float(size)
                # offset_y1=(y1-ny1)/float(size)
                # offset_x2=(x2-nx2)/float(size)
                # offset_y2=(y2-ny2)/float(size)
                offset_x1=(x1-nx1)/float(size_w)
                offset_y1=(y1-ny1)/float(size_h)
                offset_x2=(x2-nx2)/float(size_w)
                offset_y2=(y2-ny2)/float(size_h)

                cropped_im=img[ny1:ny2,nx1:nx2,:]
                # resized_im=cv2.resize(cropped_im,(12,12),interpolation=cv2.INTER_LINEAR)
                #box扩充一个维度作为iou输入
                box_=box.reshape(1,-1)
                iou=IOU(crop_box,box_)
                if iou>=0.65:
                    resized_im=cv2.resize(cropped_im,(crop_width,crop_height),interpolation=cv2.INTER_LINEAR)

                    save_file=os.path.join(pos_save_dir,'%s.jpg'%p_idx)
                    f1.write(pos_save_dir+'/%s.jpg'%p_idx+' 1 %.2f %.2f %.2f %.2f\n'%(offset_x1,
                            offset_y1,offset_x2,offset_y2))
                    cv2.imwrite(save_file,resized_im)
                    p_idx+=1
                elif iou>=0.4:
                    resized_im=cv2.resize(cropped_im,(crop_width,crop_height),interpolation=cv2.INTER_LINEAR)

                    save_file=os.path.join(part_save_dir,'%s.jpg'%d_idx)
                    f3.write(part_save_dir+'/%s.jpg'%d_idx+' -1 %.2f %.2f %.2f %.2f\n'%(offset_x1,
                            offset_y1,offset_x2,offset_y2))
                    cv2.imwrite(save_file,resized_im)
                    d_idx+=1

       
    print('%s 个图片已处理，pos：%s  part: %s neg:%s'%(idx,p_idx,d_idx,n_idx))
    f1.close()
    f2.close()
    f3.close()



def parse_arguments(argv):

    parser = argparse.ArgumentParser()

    parser.add_argument('--suffix', type=str, default=None,
                        help='The suffix for the folder.')

    parser.add_argument('--anno_file', type=str, default=None,
                        help='The path of annotation txt.')

    parser.add_argument('--im_root', type=str, default=None,
                        help='The root of img.')

    parser.add_argument('--add_img_suffix', action='store_true', default=False, 
                        help='Whether to add the suffix for the img.')

    parser.add_argument('--crop_width', type=int, default=12, 
                        help='The width for the crop area.')

    parser.add_argument('--crop_height', type=int, default=12, 
                        help='The height for the crop area.')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))



