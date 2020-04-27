
# coding: utf-8

# In[3]:


import cv2
import numpy as np
import sys

sys.path.append('../')
from preprocess.utils import *
from tqdm import tqdm


# In[4]:


def py_nms(dets,thresh):
    '''剔除太相似的box'''
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    #将概率值从大到小排列
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        
        ovr = inter / (areas[i] + areas[order[1:]] - inter+1e-10)
       
        #保留小于阈值的下标，因为order[0]拿出来做比较了，所以inds+1是原来对应的下标
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


# In[ ]:


class MtcnnDetector:
    '''来生成人脸的图像'''
    def __init__(self,detectors,
                min_face_size=20,
                stride=2,
                threshold=[0.6,0.7,0.7],
                scale_factor=0.79,#图像金字塔的缩小率
                pnet_size=12,
                sliding_win=False,
                win_size=(300,300)):
        self.pnet_detector=detectors[0]
        self.rnet_detector=detectors[1]
        self.onet_detector=detectors[2]
        self.min_face_size=min_face_size
        self.stride=stride
        self.thresh=threshold
        self.scale_factor=scale_factor

        ###
        self.pnet_size = pnet_size
        self.sliding_win = sliding_win
        self.win_size = win_size

        # pdb.set_trace()

    def detect_face(self,test_data):
        all_boxes=[]
        landmarks=[]
        batch_idx=0
        num_of_img=test_data.size
        empty_array=np.array([])
        for databatch in tqdm(test_data):
            batch_idx+=1
            im=databatch
            if self.pnet_detector:
                boxes,boxes_c,landmark=self.detect_pnet(im)
                if boxes_c is None:
                    all_boxes.append(empty_array)
                    landmarks.append(empty_array)
                    continue
            if self.rnet_detector:
                boxes, boxes_c, landmark = self.detect_rnet(im, boxes_c)

                if boxes_c is None:
                    all_boxes.append(empty_array)
                    landmarks.append(empty_array)

                    continue
            if self.onet_detector:
                boxes, boxes_c, landmark = self.detect_onet(im, boxes_c)
               
                if boxes_c is None:
                    all_boxes.append(empty_array)
                    landmarks.append(empty_array)

                    continue

            all_boxes.append(boxes_c)
            landmark = [1]
            landmarks.append(landmark)
        return all_boxes, landmarks

    def check_bandwidth(self, im_size):

        BANDWIDTH = 1024 * 1024
        channel = 10

        h, w = im_size

        if h*w*channel > BANDWIDTH:
            return True
        else:
            return False

    def gen_sliding_win(self, im):
        """
            产生im上的滑窗
        """

        res = []

        im_tmp = im.copy()

        im_h, im_w, _ = im.shape
        win_h, win_w = self.win_size

        assert im_h > self.pnet_size and im_w > self.pnet_size

        h_num = np.ceil(im_h / win_h).astype(np.int)
        w_num = np.ceil(im_w / win_w).astype(np.int)


        for i in range(h_num):
            if i == h_num-1:
                lty = max(0, im_h - win_h)
            else:
                lty = i * win_h
            for j in range(w_num):
                if j == w_num-1:
                    ltx = max(0, im_w - win_w)
                else:
                    ltx = j * win_w

                win = im_tmp[lty:lty+win_h, ltx:ltx+win_w, :].copy()
                coords = (ltx, lty)

                res.append((win, coords))


        return res


    def detect_pnet(self,im):
        '''通过pnet筛选box和landmark
        参数：
          im:输入图像[h,2,3]
        '''
        h,w,c=im.shape

        # net_size=12
        net_size = self.pnet_size

        # pdb.set_trace()

        #人脸和输入图像的比率
        current_scale=float(net_size)/self.min_face_size
        im_resized=self.processed_image(im,current_scale)
        current_height,current_width,_=im_resized.shape
        all_boxes=list()
        #图像金字塔
        while min(current_height,current_width)>net_size:

            # pdb.set_trace()


            if self.sliding_win and self.check_bandwidth((current_height, current_width)):

                # 获取滑窗
                sliding_win_list = self.gen_sliding_win(im_resized)

                for sliding_win in sliding_win_list:
                    # 遍历所有的滑窗

                    win_im, left_top = sliding_win

                    ltx, lty = left_top

                    cls_cls_map,reg = self.pnet_detector.predict(win_im)

                    boxes = self.generate_bbox(cls_cls_map[:,:,1],reg,current_scale,self.thresh[0])
                    # boxes = self.generate_bbox(cls_cls_map[:,:,idx],reg[:,:,(idx-1)*4:idx*4],current_scale,self.thresh[0])

                    if boxes.size==0:
                        continue

                    boxes[:, 0] += ltx
                    boxes[:, 1] += lty
                    boxes[:, 2] += ltx
                    boxes[:, 3] += lty

                    keep=py_nms(boxes[:,:5],0.5)
                    boxes=boxes[keep]
                    all_boxes.append(boxes)

                    # pdb.set_trace()


                current_scale*=self.scale_factor
                im_resized=self.processed_image(im,current_scale)
                current_height,current_width,_=im_resized.shape
                continue


            #类别和box
            cls_cls_map,reg=self.pnet_detector.predict(im_resized)
            boxes=self.generate_bbox(cls_cls_map[:,:,1],reg,current_scale,self.thresh[0])
            # boxes=self.generate_bbox(cls_cls_map[:,:,idx],reg[:,:,(idx-1)*4:idx*4],current_scale,self.thresh[0])

            current_scale*=self.scale_factor#继续缩小图像做金字塔
            im_resized=self.processed_image(im,current_scale)
            current_height,current_width,_=im_resized.shape
            
            if boxes.size==0:
                continue
            #非极大值抑制留下重复低的box
            ### 这里抑制的其实是偏移之前的bbox，最下面的boxes_c才是偏移之后的bbox，此函数并没有对偏移之后的bbox做nms，因为本函数主要是用于生成proposal
            keep=py_nms(boxes[:,:5],0.5)
            boxes=boxes[keep]
            all_boxes.append(boxes)
        if len(all_boxes)==0:
            return None,None,None
        all_boxes=np.vstack(all_boxes)
        #将金字塔之后的box也进行非极大值抑制
        keep = py_nms(all_boxes[:, 0:5], 0.7)
        all_boxes = all_boxes[keep]
        boxes = all_boxes[:, :5]
        #box的长宽
        bbw = all_boxes[:, 2] - all_boxes[:, 0] + 1
        bbh = all_boxes[:, 3] - all_boxes[:, 1] + 1
        #对应原图的box坐标和分数
        boxes_c = np.vstack([all_boxes[:, 0] + all_boxes[:, 5] * bbw,
                             all_boxes[:, 1] + all_boxes[:, 6] * bbh,
                             all_boxes[:, 2] + all_boxes[:, 7] * bbw,
                             all_boxes[:, 3] + all_boxes[:, 8] * bbh,
                             all_boxes[:, 4]])
        boxes_c = boxes_c.T

        return boxes, boxes_c, None
    def detect_rnet(self,im,dets):
        '''通过rent选择box
        参数：
          im：输入图像
          dets:pnet选择的box，是相对原图的绝对坐标
        返回值：
          box绝对坐标
        '''
        h,w,c=im.shape
        #将pnet的box变成包含它的正方形，可以避免信息损失
        dets=convert_to_square(dets)
        dets[:,0:4]=np.round(dets[:,0:4])
        #调整超出图像的box
        [dy,edy,dx,edx,y,ey,x,ex,tmpw,tmph]=self.pad(dets,w,h)
        delete_size=np.ones_like(tmpw)*20
        ones=np.ones_like(tmpw)
        zeros=np.zeros_like(tmpw)

        ### 使用delete_size获取num_boxes会有bug，造成下面的for循环不能遍历PNet提供的所有box
        # num_boxes=np.sum(np.where((np.minimum(tmpw,tmph)>=delete_size),ones,zeros))
        num_boxes = dets.shape[0]

        # import pdb
        # pdb.set_trace()

        cropped_ims=np.zeros((num_boxes,24,24,3),dtype=np.float32)
        for i in range(num_boxes):
            ###
            #将pnet生成的box相对与原图进行裁剪，超出部分用0补
            # if tmph[i]<20 or tmpw[i]<20:
            #     continue
            if tmph[i]<5 or tmpw[i]<5:
                continue

            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]  ### 这里就是将超出图片边界的部分做0-padding
            cropped_ims[i, :, :, :] = (cv2.resize(tmp, (24, 24)) - 127.5) / 128
        cls_scores, reg, _ = self.rnet_detector.predict(cropped_ims)
        cls_scores = cls_scores[:, 1]
        # cls_scores = cls_scores[:, idx]
        # reg = reg[:, (idx-1)*4:idx*4]

        keep_inds = np.where(cls_scores > self.thresh[1])[0]
        if len(keep_inds) > 0:
            boxes = dets[keep_inds]
            boxes[:, 4] = cls_scores[keep_inds]
            reg = reg[keep_inds]
        else:
            return None, None, None

        keep = py_nms(boxes, 0.6)
        boxes = boxes[keep]
        #对pnet截取的图像的坐标进行校准，生成rnet的人脸框对于原图的绝对坐标
        boxes_c = self.calibrate_box(boxes, reg[keep])
        return boxes, boxes_c, None
    
    def detect_onet(self,im,dets):
        '''将onet的选框继续筛选基本和rnet差不多但多返回了landmark'''
        h,w,c=im.shape
        dets=convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(dets, w, h)
        num_boxes = dets.shape[0]
        cropped_ims = np.zeros((num_boxes, 48, 48, 3), dtype=np.float32)
        for i in range(num_boxes):
            ### 
            if tmph[i] <=0 or tmpw[i] <= 0:
                continue
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            # print(tmph[i], tmpw[i])
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            cropped_ims[i, :, :, :] = (cv2.resize(tmp, (48, 48)) - 127.5) / 128

        # cls_scores, reg, landmark = self.onet_detector.predict(cropped_ims)
        cls_scores, reg, _ = self.onet_detector.predict(cropped_ims)

        
        cls_scores = cls_scores[:, 1]
        keep_inds = np.where(cls_scores > self.thresh[2])[0]
        if len(keep_inds) > 0:
            
            boxes = dets[keep_inds]
            boxes[:, 4] = cls_scores[keep_inds]
            reg = reg[keep_inds]
            # landmark = landmark[keep_inds]
        else:
            return None, None, None

    
        w = boxes[:, 2] - boxes[:, 0] + 1
        
        h = boxes[:, 3] - boxes[:, 1] + 1
        # landmark[:, 0::2] = (np.tile(w, (5, 1)) * landmark[:, 0::2].T + np.tile(boxes[:, 0], (5, 1)) - 1).T
        # landmark[:, 1::2] = (np.tile(h, (5, 1)) * landmark[:, 1::2].T + np.tile(boxes[:, 1], (5, 1)) - 1).T
        boxes_c = self.calibrate_box(boxes, reg)

        boxes = boxes[py_nms(boxes, 0.6)]
        # keep = py_nms(boxes_c, 0.6)
        keep = py_nms(boxes_c, 0.3)

        boxes_c = boxes_c[keep]
        # landmark = landmark[keep]
        # return boxes, boxes_c, landmark
        return boxes, boxes_c, None


    def processed_image(self, img, scale):
        '''预处理数据，转化图像尺度并对像素归一到[-1,1]
        '''
        height, width, channels = img.shape
        new_height = int(height * scale)  
        new_width = int(width * scale)  
        new_dim = (new_width, new_height)
        img_resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_LINEAR) 
        img_resized = (img_resized - 127.5) / 128
        return img_resized
        
    def generate_bbox(self, cls_map, reg, scale, threshold):
        """
         得到对应原图的box坐标，分类分数，box偏移量
        """
        """###
            - output: 
                * boundingbox.T: (N, 9), 9 means [x1, y1, x2, y2, score, dx1, dy1, dx2, dy2]
                * 这里的 x1, y1, x2, y2 是最后输出的feature map的每一个像素映射回原图的感受野坐标，因为这里使用的是全卷积，所以这一个区域等价于训练的时候crop的区域，其宽高可以用于后面偏移量的反归一化
                * 所以此函数的bbox指的是输出的feature map的每一个像素位置对应回原图的感受野区域 (类似于anchor的作用)，这里看成等价于训练时候的crop img
        """
        #pnet大致将图像size缩小2倍
        ### PNet中间做了一次stride为2的pooling，即网络的步长因子 (jump)为2
        # stride = 2
        stride = self.stride

        
        ### 这里的cellsize其实就是PNet的感受野
        # cellsize = 12
        cellsize = self.pnet_size


        #将置信度高的留下
        t_index = np.where(cls_map > threshold)

        # 没有人脸
        if t_index[0].size == 0:
            return np.array([])
        # 偏移量
        dx1, dy1, dx2, dy2 = [reg[t_index[0], t_index[1], i] for i in range(4)]

        reg = np.array([dx1, dy1, dx2, dy2])
        score = cls_map[t_index[0], t_index[1]]
        #对应原图的box坐标，分类分数，box偏移量
        boundingbox = np.vstack([np.round((stride * t_index[1]) / scale),
                                 np.round((stride * t_index[0]) / scale),
                                 np.round((stride * t_index[1] + cellsize) / scale),
                                 np.round((stride * t_index[0] + cellsize) / scale),
                                 score,
                                 reg])
        #shape[n,9]
        return boundingbox.T 
    def pad(self, bboxes, w, h):
        '''将超出图像的box进行处理
        参数：
          bboxes:人脸框
          w,h:图像长宽
        返回值：
          dy, dx : 为调整后的box的左上角坐标相对于原box左上角的坐标
          edy, edx : n为调整后的box右下角相对原box左上角的相对坐标
          y, x : 调整后的box在原图上左上角的坐标
          ex, ex : 调整后的box在原图上右下角的坐标
          tmph, tmpw: 原始box的长宽
        '''
        """###
            本函数实际上的作用就是将超出边界的框截断到边界的位置。
        """
        #box的长宽
        tmpw, tmph = bboxes[:, 2] - bboxes[:, 0] + 1, bboxes[:, 3] - bboxes[:, 1] + 1
        num_box = bboxes.shape[0]

        dx, dy = np.zeros((num_box,)), np.zeros((num_box,))
        edx, edy = tmpw.copy() - 1, tmph.copy() - 1
        #box左上右下的坐标
        x, y, ex, ey = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
        #找到超出右下边界的box并将ex,ey归为图像的w,h
        #edx,edy为调整后的box右下角相对原box左上角的相对坐标
        tmp_index = np.where(ex > w - 1)
        edx[tmp_index] = tmpw[tmp_index] + w - 2 - ex[tmp_index]    ### (w-1)-(ex[tmp_index]-tmpw[tmp_index]+1)
        ex[tmp_index] = w - 1

        tmp_index = np.where(ey > h - 1)
        edy[tmp_index] = tmph[tmp_index] + h - 2 - ey[tmp_index]
        ey[tmp_index] = h - 1
        #找到超出左上角的box并将x,y归为0
        #dx,dy为调整后的box的左上角坐标相对于原box左上角的坐标
        tmp_index = np.where(x < 0)
        dx[tmp_index] = 0 - x[tmp_index]
        x[tmp_index] = 0

        tmp_index = np.where(y < 0)
        dy[tmp_index] = 0 - y[tmp_index]
        y[tmp_index] = 0

        return_list = [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]
        return_list = [item.astype(np.int32) for item in return_list]

        return return_list
    def calibrate_box(self, bbox, reg):
        '''校准box
        参数：
          bbox:pnet生成的box

          reg:rnet生成的box偏移值
        返回值：
          调整后的box是针对原图的绝对坐标
        '''

        bbox_c = bbox.copy()
        w = bbox[:, 2] - bbox[:, 0] + 1
        w = np.expand_dims(w, 1)
        h = bbox[:, 3] - bbox[:, 1] + 1
        h = np.expand_dims(h, 1)
        reg_m = np.hstack([w, h, w, h])
        aug = reg_m * reg
        bbox_c[:, 0:4] = bbox_c[:, 0:4] + aug
        return bbox_c
    def detect(self, img):
        '''用于测试当个图像的'''
        boxes = None

        ###
        landmark = None

        # pnet
        if self.pnet_detector:
            boxes, boxes_c, _ = self.detect_pnet(img)
            if boxes_c is None:
                return np.array([]), np.array([])


        # rnet
        if self.rnet_detector:
            boxes, boxes_c, _ = self.detect_rnet(img, boxes_c)
            if boxes_c is None:
                return np.array([]), np.array([])


        # onet
        if self.onet_detector:
            boxes, boxes_c, landmark = self.detect_onet(img, boxes_c)
            if boxes_c is None:
                return np.array([]), np.array([])


        return boxes_c, landmark

    def detect_2cls(self, img):
        '''用于测试当个图像的'''
        boxes = None

        ###
        landmark = None

        # pnet
        if self.pnet_detector:
            boxes, boxes_c, _ = self.detect_pnet_2cls(img)
            if boxes_c is None:
                return [np.array([]), np.array([])], None


        # rnet
        if self.rnet_detector:
            boxes, boxes_c, _ = self.detect_rnet_2cls(img, boxes_c)
            if boxes_c is None:
                return [np.array([]), np.array([])], None


        # onet
        if self.onet_detector:
            boxes, boxes_c, landmark = self.detect_onet_2cls(img, boxes_c)

        return boxes_c, landmark


    def _concat(self, list_1, list_2):

        assert len(list_1) == len(list_2)
 
        res = []

        for i in range(len(list_1)):

            if list_1[i] is None and list_2[i] is None:
                res.append(None)
                continue

            if list_1[i] is None:
                res.append(list_2[i])
                continue

            if list_2[i] is None:
                res.append(list_1[i])
                continue

            res.append(np.vstack([list_1[i], list_2[i]]))

        return res



    def detect_head_upperbody(self, test_data):
        all_boxes=[]
        landmarks=[]
        batch_idx=0
        num_of_img=test_data.size
        empty_array=np.array([])
        for databatch in tqdm(test_data):
            batch_idx+=1
            im=databatch
            if self.pnet_detector:

                boxes, boxes_c, landmark = self.detect_pnet_2cls(im)

                # pdb.set_trace()

                if boxes_c is None:
                    all_boxes.append(empty_array)
                    landmarks.append(empty_array)
                    continue
            if self.rnet_detector:

                boxes, boxes_c, landmark = self.detect_rnet_2cls(im, boxes_c)

                if boxes_c is None:
                    all_boxes.append(empty_array)
                    landmarks.append(empty_array)

                    continue
            if self.onet_detector:
                raise

            all_boxes.append(boxes_c)
            landmark = [1]
            landmarks.append(landmark)
        return all_boxes, landmarks


    def detect_pnet_2cls(self,im):
        '''通过pnet筛选box和landmark
        参数：
          im:输入图像[h,2,3]
        '''
        h,w,c=im.shape

        # net_size=12
        net_size = self.pnet_size

        # pdb.set_trace()

        #人脸和输入图像的比率
        current_scale=float(net_size)/self.min_face_size
        im_resized=self.processed_image(im,current_scale)
        current_height,current_width,_=im_resized.shape
        all_boxes=list()
        #图像金字塔
        while min(current_height,current_width)>net_size:

            # pdb.set_trace()


            if self.sliding_win and self.check_bandwidth((current_height, current_width)):

                # 获取滑窗
                sliding_win_list = self.gen_sliding_win(im_resized)

                for sliding_win in sliding_win_list:
                    # 遍历所有的滑窗

                    win_im, left_top = sliding_win

                    ltx, lty = left_top

                    cls_cls_map,reg = self.pnet_detector.predict(win_im)

                    for idx in [1,2]:
                        boxes = self.generate_bbox(cls_cls_map[:,:,idx],reg[:,:,(idx-1)*4:idx*4],current_scale,self.thresh[0])

                        if boxes.size==0:
                            continue

                        boxes[:, 0] += ltx
                        boxes[:, 1] += lty
                        boxes[:, 2] += ltx
                        boxes[:, 3] += lty

                        keep=py_nms(boxes[:,:5],0.5)
                        boxes=boxes[keep]
                        all_boxes.append(boxes)

                    # pdb.set_trace()


                current_scale*=self.scale_factor
                im_resized=self.processed_image(im,current_scale)
                current_height,current_width,_=im_resized.shape
                continue


            #类别和box
            cls_cls_map,reg=self.pnet_detector.predict(im_resized)

            for idx in [1,2]:
                boxes=self.generate_bbox(cls_cls_map[:,:,idx],reg[:,:,(idx-1)*4:idx*4],current_scale,self.thresh[0])

                # pdb.set_trace()

                if boxes.size==0:
                    continue
                #非极大值抑制留下重复低的box
                ### 这里抑制的其实是偏移之前的bbox，最下面的boxes_c才是偏移之后的bbox，此函数并没有对偏移之后的bbox做nms，因为本函数主要是用于生成proposal
                keep=py_nms(boxes[:,:5],0.5)
                boxes=boxes[keep]
                all_boxes.append(boxes)


            current_scale*=self.scale_factor#继续缩小图像做金字塔
            im_resized=self.processed_image(im,current_scale)
            current_height,current_width,_=im_resized.shape


        if len(all_boxes)==0:
            return None,None,None
        all_boxes=np.vstack(all_boxes)
        #将金字塔之后的box也进行非极大值抑制
        keep = py_nms(all_boxes[:, 0:5], 0.7)
        all_boxes = all_boxes[keep]
        boxes = all_boxes[:, :5]
        #box的长宽
        bbw = all_boxes[:, 2] - all_boxes[:, 0] + 1
        bbh = all_boxes[:, 3] - all_boxes[:, 1] + 1
        #对应原图的box坐标和分数
        boxes_c = np.vstack([all_boxes[:, 0] + all_boxes[:, 5] * bbw,
                             all_boxes[:, 1] + all_boxes[:, 6] * bbh,
                             all_boxes[:, 2] + all_boxes[:, 7] * bbw,
                             all_boxes[:, 3] + all_boxes[:, 8] * bbh,
                             all_boxes[:, 4]])
        boxes_c = boxes_c.T

        return boxes, boxes_c, None

    def detect_rnet_2cls(self,im,dets):
        '''通过rent选择box
        参数：
          im：输入图像
          dets:pnet选择的box，是相对原图的绝对坐标
        返回值：
          box绝对坐标
        '''
        all_boxes=list()
        all_boxes_c=list()

        h,w,c=im.shape
        #将pnet的box变成包含它的正方形，可以避免信息损失
        dets=convert_to_square(dets)
        dets[:,0:4]=np.round(dets[:,0:4])
        #调整超出图像的box
        [dy,edy,dx,edx,y,ey,x,ex,tmpw,tmph]=self.pad(dets,w,h)
        delete_size=np.ones_like(tmpw)*20
        ones=np.ones_like(tmpw)
        zeros=np.zeros_like(tmpw)

        ### 使用delete_size获取num_boxes会有bug，造成下面的for循环不能遍历PNet提供的所有box
        # num_boxes=np.sum(np.where((np.minimum(tmpw,tmph)>=delete_size),ones,zeros))
        num_boxes = dets.shape[0]

        # import pdb
        # pdb.set_trace()

        cropped_ims=np.zeros((num_boxes,24,24,3),dtype=np.float32)
        for i in range(num_boxes):
            ###
            #将pnet生成的box相对与原图进行裁剪，超出部分用0补
            # if tmph[i]<20 or tmpw[i]<20:
            #     continue

            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]  ### 这里就是将超出图片边界的部分做0-padding
            cropped_ims[i, :, :, :] = (cv2.resize(tmp, (24, 24)) - 127.5) / 128
        cls_scores, reg, _ = self.rnet_detector.predict(cropped_ims)

        for idx in [1,2]:

            cls_scores_cur = cls_scores[:, idx]
            reg_cur = reg[:, (idx-1)*4:idx*4]

            keep_inds = np.where(cls_scores_cur > self.thresh[1])[0]
            if len(keep_inds) > 0:
                boxes = dets[keep_inds]
                boxes[:, 4] = cls_scores_cur[keep_inds]
                reg_cur = reg_cur[keep_inds]
            else:
                continue

            keep = py_nms(boxes, 0.6)
            boxes = boxes[keep]
            #对pnet截取的图像的坐标进行校准，生成rnet的人脸框对于原图的绝对坐标
            boxes_c = self.calibrate_box(boxes, reg_cur[keep])

            all_boxes.append(boxes)
            all_boxes_c.append(boxes_c)

        if len(all_boxes)==0:
            return None, None, None

        boxes = np.vstack(all_boxes)
        boxes_c = np.vstack(all_boxes_c)

        return boxes, boxes_c, None


    def detect_onet_2cls(self,im,dets):
        '''
            将onet的选框继续筛选基本和rnet差不多但多返回了landmark
            - outputs:
                all_boxes_c: list. 分别代表head和upperbody的检测结果
        '''
        all_boxes = [np.array([]), np.array([])]
        all_boxes_c = [np.array([]), np.array([])]

        h,w,c=im.shape
        dets=convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(dets, w, h)
        num_boxes = dets.shape[0]
        cropped_ims = np.zeros((num_boxes, 48, 48, 3), dtype=np.float32)
        for i in range(num_boxes):
            ### 
            if tmph[i] <=0 or tmpw[i] <= 0:
                continue
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            # print(tmph[i], tmpw[i])
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            cropped_ims[i, :, :, :] = (cv2.resize(tmp, (48, 48)) - 127.5) / 128

        cls_scores, reg, _ = self.onet_detector.predict(cropped_ims)

        for idx in [1,2]:
        
            cls_scores_cur = cls_scores[:, idx]
            reg_cur = reg[:, (idx-1)*4:idx*4]

            keep_inds = np.where(cls_scores_cur > self.thresh[2])[0]

            if len(keep_inds) > 0:
                boxes = dets[keep_inds]
                boxes[:, 4] = cls_scores_cur[keep_inds]
                reg_cur = reg_cur[keep_inds]
                # landmark = landmark[keep_inds]
            else:
                continue

            boxes_c = self.calibrate_box(boxes, reg_cur)

            boxes = boxes[py_nms(boxes, 0.6)]
            keep = py_nms(boxes_c, 0.6)
            boxes_c = boxes_c[keep]

            all_boxes[idx-1] = boxes
            all_boxes_c[idx-1] = boxes_c

        return all_boxes, all_boxes_c, None


