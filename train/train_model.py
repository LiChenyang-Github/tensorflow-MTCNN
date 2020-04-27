
# coding: utf-8

# In[1]:


import os
import sys
from datetime import datetime
import numpy as np
import tensorflow as tf
import config as FLAGS
# import config_2cls as FLAGS_2CLS
import random
import cv2

import pdb
import os.path as osp
from glob import glob
from time import time

# In[ ]:


def train(net_factory,prefix,end_epoch,base_dir,display,base_lr, \
    suffix=None,pretrained=None,resume=None,size=12,net=None,exclude_vars=None,aspect=None):
    '''训练模型'''
    ###
    # if suffix is None:
    #     size=int(base_dir.split('/')[-1])
    # else:
    #     size=int(base_dir.split('/')[-1].split('_')[0])
    if size==12:
        # net='PNet'
        radio_cls_loss = 1.0;radio_bbox_loss = 0.5;radio_landmark_loss = 0.5;
    elif size==9:
        # net='PNet_9'
        # pdb.set_trace()
        radio_cls_loss = 1.0;radio_bbox_loss = 0.5;radio_landmark_loss = 0.5;
    elif size==8:
        # net='PNet_8'
        # pdb.set_trace()
        radio_cls_loss = 1.0;radio_bbox_loss = 0.5;radio_landmark_loss = 0.5;
    elif size==24:
        # net='RNet'
        radio_cls_loss = 1.0;radio_bbox_loss = 0.5;radio_landmark_loss = 0.5;
    elif size==48:
        # net='ONet'
        radio_cls_loss = 1.0;radio_bbox_loss = 0.5;radio_landmark_loss = 1;
        
    if net=='PNet' or 'P_Net' in net:
        #计算一共多少组数据
        label_file=os.path.join(base_dir,'train_pnet_landmark.txt')
        f = open(label_file, 'r')
   
        num = len(f.readlines())
        dataset_dir=os.path.join(base_dir,'tfrecord/train_PNet_landmark.tfrecord_shuffle')
        #从tfrecord读取数据
        image_batch,label_batch,bbox_batch,landmark_batch=read_single_tfrecord(dataset_dir, \
            FLAGS.batch_size,net,aspect=aspect)
    elif net=='PNet_9' or net=='PNet_8':
        #计算一共多少组数据
        label_file=os.path.join(base_dir, '../12/', 'train_pnet_landmark.txt')
        f = open(label_file, 'r')
   
        num = len(f.readlines())
        dataset_dir=os.path.join(base_dir,'tfrecord/train_{}_landmark.tfrecord_shuffle'.format(net))
        #从tfrecord读取数据
        image_batch,label_batch,bbox_batch,landmark_batch=read_single_tfrecord(dataset_dir, \
            FLAGS.batch_size,net,aspect=aspect)
    else:
        #计算一共多少组数据
        label_file1=os.path.join(base_dir,'pos_%d.txt'%(size))
        f1 = open(label_file1, 'r')
        label_file2=os.path.join(base_dir,'part_%d.txt'%(size))
        f2 = open(label_file2, 'r')
        label_file3=os.path.join(base_dir,'neg_%d.txt'%(size))
        f3 = open(label_file3, 'r')
        # label_file4=os.path.join(base_dir,'landmark_%d_aug.txt'%(size))
        # f4 = open(label_file4, 'r')
   
        # num = len(f1.readlines())+len(f2.readlines())+len(f3.readlines())+len(f4.readlines())
        num = len(f1.readlines())+len(f2.readlines())+len(f3.readlines())

    
        pos_dir = os.path.join(base_dir,'tfrecord/pos_landmark.tfrecord_shuffle')
        part_dir = os.path.join(base_dir,'tfrecord/part_landmark.tfrecord_shuffle')
        neg_dir = os.path.join(base_dir,'tfrecord/neg_landmark.tfrecord_shuffle')
        # landmark_dir = os.path.join(base_dir,'tfrecord/landmark_landmark.tfrecord_shuffle')
        # dataset_dirs=[pos_dir,part_dir,neg_dir,landmark_dir]
        dataset_dirs=[pos_dir,part_dir,neg_dir]

        #各数据占比
        #目的是使每一个batch的数据占比都相同
        # pos_radio,part_radio,landmark_radio,neg_radio=1.0/6,1.0/6,1.0/6,3.0/6
        pos_radio,part_radio,neg_radio=1.5/6,1.5/6,3.0/6

        pos_batch_size=int(np.ceil(FLAGS.batch_size*pos_radio))
        assert pos_batch_size != 0,"Batch Size 有误 "
        part_batch_size = int(np.ceil(FLAGS.batch_size*part_radio))
        assert part_batch_size != 0,"BBatch Size 有误 "
        neg_batch_size = int(np.ceil(FLAGS.batch_size*neg_radio))
        assert neg_batch_size != 0,"Batch Size 有误 "
        # landmark_batch_size = int(np.ceil(FLAGS.batch_size*landmark_radio))
        # assert landmark_batch_size != 0,"Batch Size 有误 "
        # batch_sizes = [pos_batch_size,part_batch_size,neg_batch_size,landmark_batch_size]
        batch_sizes = [pos_batch_size,part_batch_size,neg_batch_size]

        # image_batch, label_batch, bbox_batch,landmark_batch = read_multi_tfrecords(dataset_dirs,batch_sizes, net)  
        image_batch, label_batch, bbox_batch,landmark_batch = read_multi_tfrecords_xiao(dataset_dirs,batch_sizes, net)  

    # pdb.set_trace()

    if aspect is None:
        input_image=tf.placeholder(tf.float32,shape=[FLAGS.batch_size,size,size,3],name='input_image')
    else:
        assert len(aspect) == 2
        input_image=tf.placeholder(tf.float32,shape=[FLAGS.batch_size,aspect[0],aspect[1],3],name='input_image')

    label=tf.placeholder(tf.float32,shape=[FLAGS.batch_size],name='label')
    bbox_target=tf.placeholder(tf.float32,shape=[FLAGS.batch_size,4],name='bbox_target')
    # landmark_target=tf.placeholder(tf.float32,shape=[FLAGS.batch_size,10],name='landmark_target')

    # pdb.set_trace()

    #图像色相变换
    input_image=image_color_distort(input_image)
    # cls_loss_op,bbox_loss_op,landmark_loss_op,L2_loss_op,accuracy_op=net_factory(input_image,
    #                     label,bbox_target,landmark_target,training=True)
    cls_loss_op,bbox_loss_op,L2_loss_op,accuracy_op=net_factory(input_image,
                        label,bbox_target,training=True)

    # total_loss_op=radio_cls_loss*cls_loss_op+radio_bbox_loss*bbox_loss_op+            radio_landmark_loss*landmark_loss_op+L2_loss_op
    total_loss_op=radio_cls_loss*cls_loss_op+radio_bbox_loss*bbox_loss_op+L2_loss_op


    train_op,lr_op=optimize(base_lr,total_loss_op,num)
    
    
    tf.summary.scalar("cls_loss",cls_loss_op)#cls_loss
    tf.summary.scalar("bbox_loss",bbox_loss_op)#bbox_loss
    # tf.summary.scalar("landmark_loss",landmark_loss_op)#landmark_loss
    tf.summary.scalar("cls_accuracy",accuracy_op)#cls_acc
    tf.summary.scalar("total_loss",total_loss_op)#cls_loss, bbox loss, landmark loss and L2 loss add together
    summary_op = tf.summary.merge_all()

    ###
    if suffix is None:
        logs_dir = "../graph_mine/%s" %(net)
    else:
        logs_dir = "../graph_mine/%s/%s" %(suffix, net)

    if os.path.exists(logs_dir) == False:
        os.makedirs(logs_dir)
    #模型训练
    ### 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True


    init = tf.global_variables_initializer()
    # sess = tf.Session()

    ###
    sess = tf.Session(config=config)

    saver = tf.train.Saver(max_to_keep=3)


    # pdb.set_trace()
    assert pretrained is None or resume is None
    if pretrained is not None:
        # model_file=tf.train.latest_checkpoint(pretrained)
        # # pdb.set_trace()
        # saver.restore(sess,model_file)

        # variables = tf.global_variables()
        # variable_names = [v.name for v in variables]
        # v_id = variable_names.index('Variable:0')

        # # pdb.set_trace()

        # sess.run(tf.variables_initializer([variables[v_id], ], name='finetune_init'))   # 将迭代器的计数重新置零

        # pdb.set_trace()

        model_file=tf.train.latest_checkpoint(pretrained)

        if exclude_vars is None:
            exclude_vars = ['Variable']
        else:
            exclude_vars += ['Variable']

        variables_to_restore = tf.contrib.slim.get_variables_to_restore(exclude=exclude_vars)
        tf.train.init_from_checkpoint(model_file, {v.name.split(':')[0]: v for v in variables_to_restore})
        sess.run(tf.global_variables_initializer())

        start_iter = 0
        epoch = 0

    elif resume is not None:
        model_file=tf.train.latest_checkpoint(resume)
        # pdb.set_trace()
        saver.restore(sess,model_file)

        variables = tf.global_variables()
        variable_names = [v.name for v in variables]
        v_id = variable_names.index('Variable:0')

        start_iter = sess.run(variables[v_id])
        epoch = int(osp.basename(model_file).split('-')[-1])

    else:
        sess.run(tf.global_variables_initializer())

        start_iter = 0
        epoch = 0

    #模型的graph
    writer = tf.summary.FileWriter(logs_dir,sess.graph)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    i = 0

    MAX_STEP = int(num / FLAGS.batch_size + 1) * end_epoch
    # epoch = 0
    sess.graph.finalize()
    try:

        start_time = time()
        # for step in range(MAX_STEP):
        for step in range(start_iter, MAX_STEP):

            i = i + 1
            if coord.should_stop():
                break
            image_batch_array, label_batch_array, bbox_batch_array,landmark_batch_array = sess.run([image_batch, label_batch, bbox_batch,landmark_batch])

            # import pdb
            # pdb.set_trace()

            #随机翻转图像
            image_batch_array,landmark_batch_array = random_flip_images(image_batch_array,label_batch_array,landmark_batch_array)
           


            # _,_,summary = sess.run([train_op, lr_op ,summary_op], feed_dict={input_image: image_batch_array, label: label_batch_array, bbox_target: bbox_batch_array,landmark_target:landmark_batch_array})
            _,_,summary = sess.run([train_op, lr_op ,summary_op], feed_dict={input_image: image_batch_array, label: label_batch_array, bbox_target: bbox_batch_array})

            #展示训练过程
            if (step+1) % display == 0:
                # cls_loss, bbox_loss,landmark_loss,L2_loss,lr,acc = sess.run([cls_loss_op, bbox_loss_op,landmark_loss_op,L2_loss_op,lr_op,accuracy_op],
                #                                              feed_dict={input_image: image_batch_array, label: label_batch_array, bbox_target: bbox_batch_array, landmark_target: landmark_batch_array})

                # total_loss = radio_cls_loss*cls_loss + radio_bbox_loss*bbox_loss + radio_landmark_loss*landmark_loss + L2_loss
                # print('epoch:%d/%d'%(epoch+1,end_epoch))
                # print("Step: %d/%d, accuracy: %3f, cls loss: %4f, bbox loss: %4f,Landmark loss :%4f,L2 loss: %4f, Total Loss: %4f ,lr:%f " % (step+1,MAX_STEP, acc, cls_loss, bbox_loss,landmark_loss, L2_loss,total_loss, lr))

                end_time = time()
                time_cost = end_time - start_time

                cls_loss, bbox_loss,L2_loss,lr,acc = sess.run([cls_loss_op, bbox_loss_op,L2_loss_op,lr_op,accuracy_op],
                                                             feed_dict={input_image: image_batch_array, label: label_batch_array, bbox_target: bbox_batch_array})

                total_loss = radio_cls_loss*cls_loss + radio_bbox_loss*bbox_loss + L2_loss
                print('epoch:%d/%d Time: %fs'%(epoch+1,end_epoch,time_cost))
                print("Step: %d/%d, accuracy: %3f, cls loss: %4f, bbox loss: %4f,L2 loss: %4f, Total Loss: %4f ,lr:%f " % (step+1,MAX_STEP, acc, cls_loss, bbox_loss, L2_loss,total_loss, lr))

                start_time = time()

            #每一次epoch保留一次模型
            if i * FLAGS.batch_size > num:
                epoch = epoch + 1
                i = 0
                path_prefix = saver.save(sess, prefix, global_step=epoch)
            writer.add_summary(summary,global_step=step)
    except tf.errors.OutOfRangeError:
        print("完成！！！")
    finally:
        coord.request_stop()
        writer.close()
    coord.join(threads)
    sess.close()



def train_2cls(net_factory,prefix,end_epoch,base_dir_list,display,base_lr,\
    suffix_list=None,pretrained=None,resume=None,size=12):
    '''训练模型，人头和上半身两个类同时训练'''

    if size==12:
        net='PNet'
        radio_cls_loss = 1.0;radio_bbox_loss = 0.5;radio_landmark_loss = 0.5;
    elif size==9:
        net='PNet_9'
        # pdb.set_trace()
        radio_cls_loss = 1.0;radio_bbox_loss = 0.5;radio_landmark_loss = 0.5;
    elif size==8:
        net='PNet_8'
        # pdb.set_trace()
        radio_cls_loss = 1.0;radio_bbox_loss = 0.5;radio_landmark_loss = 0.5;
    elif size==24:
        net='RNet'
        radio_cls_loss = 1.0;radio_bbox_loss = 0.5;radio_landmark_loss = 0.5;
    elif size==48:
        net='ONet'
        radio_cls_loss = 1.0;radio_bbox_loss = 0.5;radio_landmark_loss = 1;
        
    if net=='PNet':

        label_files = [osp.join(base_dir_list[i][0], 'train_pnet_landmark.txt') for i in range(2)]

        num = 0
        for label_file in label_files:
            f = open(label_file, 'r')
            num += len(f.readlines())
            f.close()
        dataset_dirs = [osp.join(base_dir_list[i][0], 'tfrecord/train_PNet_landmark.tfrecord_shuffle') for i in range(2)]

        batch_sizes = [int(FLAGS_2CLS.batch_size*FLAGS_2CLS.head_upperbody_ratio[i]) for i in range(2)]

        # pdb.set_trace()

        head_image_batch,head_label_batch,head_bbox_batch,head_landmark_batch = \
            read_single_tfrecord(dataset_dirs[0],batch_sizes[0],net)

        upperbody_image_batch,upperbody_label_batch,upperbody_bbox_batch,upperbody_landmark_batch = \
            read_single_tfrecord(dataset_dirs[1],batch_sizes[1],net,is_upperbody=True)

        image_batch,label_batch,bbox_batch,landmark_batch = \
            concat_batch(
                [head_image_batch,head_label_batch,head_bbox_batch,head_landmark_batch],
                [upperbody_image_batch,upperbody_label_batch,upperbody_bbox_batch,upperbody_landmark_batch]
                )

    elif net=='PNet_9' or net=='PNet_8':
        raise "To be filled."
    else:
        base_dir = base_dir_list[0]
        label_file1=glob(os.path.join(base_dir,'pos_*_{}.txt'.format(size)))
        label_file2=glob(os.path.join(base_dir,'part_*_{}.txt'.format(size)))
        label_file3=glob(os.path.join(base_dir,'neg_*_{}.txt'.format(size)))
        label_files = label_file1 + label_file2 + label_file3

        num = 0
        for label_file in label_files:
            f = open(label_file, 'r')
            num += len(f.readlines())
            f.close()

        # pdb.set_trace()

    
        pos_dirs = [
                    os.path.join(base_dir,'tfrecord/pos_landmark_head.tfrecord_shuffle'),
                    os.path.join(base_dir,'tfrecord/pos_landmark_upperbody.tfrecord_shuffle'),
                    ]
        part_dirs = [
                    os.path.join(base_dir,'tfrecord/part_landmark_head.tfrecord_shuffle'),
                    os.path.join(base_dir,'tfrecord/part_landmark_upperbody.tfrecord_shuffle'),
                    ]
        neg_dirs = [
                    os.path.join(base_dir,'tfrecord/neg_landmark_head.tfrecord_shuffle'),
                    os.path.join(base_dir,'tfrecord/neg_landmark_upperbody.tfrecord_shuffle'),
                    ]

        dataset_dirs=[pos_dirs,part_dirs,neg_dirs]

        #各数据占比
        #目的是使每一个batch的数据占比都相同
        pos_radio,part_radio,neg_radio=1.5/6,1.5/6,3.0/6

        pos_batch_size=int(np.ceil(FLAGS_2CLS.batch_size*pos_radio))
        assert pos_batch_size != 0,"Batch Size 有误 "
        part_batch_size = int(np.ceil(FLAGS_2CLS.batch_size*part_radio))
        assert part_batch_size != 0,"BBatch Size 有误 "
        neg_batch_size = int(np.ceil(FLAGS_2CLS.batch_size*neg_radio))
        assert neg_batch_size != 0,"Batch Size 有误 "

        batch_sizes = [
                        [int(pos_batch_size*FLAGS_2CLS.head_upperbody_ratio[0]), int(pos_batch_size*FLAGS_2CLS.head_upperbody_ratio[1])],
                        [int(part_batch_size*FLAGS_2CLS.head_upperbody_ratio[0]), int(part_batch_size*FLAGS_2CLS.head_upperbody_ratio[1])],
                        [int(neg_batch_size*FLAGS_2CLS.head_upperbody_ratio[0]), int(neg_batch_size*FLAGS_2CLS.head_upperbody_ratio[1])]
                        ]

        # pdb.set_trace()
        image_batch, label_batch, bbox_batch, landmark_batch = \
            read_multi_tfrecords_general(dataset_dirs,batch_sizes, net, list_list=True)  
        # pdb.set_trace()

    input_image=tf.placeholder(tf.float32,shape=[FLAGS_2CLS.batch_size,size,size,3],name='input_image')
    label=tf.placeholder(tf.float32,shape=[FLAGS_2CLS.batch_size],name='label')
    bbox_target=tf.placeholder(tf.float32,shape=[FLAGS_2CLS.batch_size,4],name='bbox_target')
    # landmark_target=tf.placeholder(tf.float32,shape=[FLAGS_2CLS.batch_size,10],name='landmark_target')
    #图像色相变换
    input_image=image_color_distort(input_image)
    # cls_loss_op,bbox_loss_op,landmark_loss_op,L2_loss_op,accuracy_op=net_factory(input_image,
    #                     label,bbox_target,landmark_target,training=True)
    cls_loss_op,bbox_loss_op,L2_loss_op,accuracy_op=net_factory(input_image,
                        label,bbox_target,training=True)

    # total_loss_op=radio_cls_loss*cls_loss_op+radio_bbox_loss*bbox_loss_op+            radio_landmark_loss*landmark_loss_op+L2_loss_op
    total_loss_op=radio_cls_loss*cls_loss_op+radio_bbox_loss*bbox_loss_op+L2_loss_op


    train_op,lr_op=optimize(base_lr,total_loss_op,num)
    
    
    tf.summary.scalar("cls_loss",cls_loss_op)#cls_loss
    tf.summary.scalar("bbox_loss",bbox_loss_op)#bbox_loss
    # tf.summary.scalar("landmark_loss",landmark_loss_op)#landmark_loss
    tf.summary.scalar("cls_accuracy",accuracy_op)#cls_acc
    tf.summary.scalar("total_loss",total_loss_op)#cls_loss, bbox loss, landmark loss and L2 loss add together
    summary_op = tf.summary.merge_all()

    ###
    if suffix_list is None:
        raise "suffix_list should not be None"
    else:
        logs_dir = "../graph_mine/2cls/{}/{}".format('-'.join(suffix_list), net)

    if os.path.exists(logs_dir) == False:
        os.makedirs(logs_dir)
    #模型训练
    ###
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True


    init = tf.global_variables_initializer()
    # sess = tf.Session()

    ###
    sess = tf.Session(config=config)

    saver = tf.train.Saver(max_to_keep=3)


    # pdb.set_trace()

    # 两者至少有一个为None
    assert pretrained is None or resume is None

    if pretrained is not None:
        model_file=tf.train.latest_checkpoint(pretrained)
        # pdb.set_trace()
        saver.restore(sess,model_file)

        variables = tf.global_variables()
        variable_names = [v.name for v in variables]
        v_id = variable_names.index('Variable:0')

        # pdb.set_trace()

        sess.run(tf.variables_initializer([variables[v_id], ], name='finetune_init'))   # 将迭代器的计数重新置零

        # pdb.set_trace()
        start_iter = 0
        epoch = 0
    elif resume is not None:
        model_file=tf.train.latest_checkpoint(resume)
        # pdb.set_trace()
        saver.restore(sess,model_file)

        variables = tf.global_variables()
        variable_names = [v.name for v in variables]
        v_id = variable_names.index('Variable:0')

        start_iter = sess.run(variables[v_id])
        epoch = int(osp.basename(model_file).split('-')[-1])
    else:
        sess.run(init)
        start_iter = 0
        epoch = 0


    #模型的graph
    writer = tf.summary.FileWriter(logs_dir,sess.graph)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    i = 0

    MAX_STEP = int(num / FLAGS_2CLS.batch_size + 1) * end_epoch
    sess.graph.finalize()
    try:

        for step in range(start_iter, MAX_STEP):
            i = i + 1
            if coord.should_stop():
                break
            image_batch_array, label_batch_array, bbox_batch_array,landmark_batch_array = sess.run([image_batch, label_batch, bbox_batch,landmark_batch])

            # import pdb
            # pdb.set_trace()

            #随机翻转图像
            image_batch_array,landmark_batch_array = random_flip_images(image_batch_array,label_batch_array,landmark_batch_array)
           


            # _,_,summary = sess.run([train_op, lr_op ,summary_op], feed_dict={input_image: image_batch_array, label: label_batch_array, bbox_target: bbox_batch_array,landmark_target:landmark_batch_array})
            _,_,summary = sess.run([train_op, lr_op ,summary_op], feed_dict={input_image: image_batch_array, label: label_batch_array, bbox_target: bbox_batch_array})

            #展示训练过程
            if (step+1) % display == 0:
                # cls_loss, bbox_loss,landmark_loss,L2_loss,lr,acc = sess.run([cls_loss_op, bbox_loss_op,landmark_loss_op,L2_loss_op,lr_op,accuracy_op],
                #                                              feed_dict={input_image: image_batch_array, label: label_batch_array, bbox_target: bbox_batch_array, landmark_target: landmark_batch_array})

                # total_loss = radio_cls_loss*cls_loss + radio_bbox_loss*bbox_loss + radio_landmark_loss*landmark_loss + L2_loss
                # print('epoch:%d/%d'%(epoch+1,end_epoch))
                # print("Step: %d/%d, accuracy: %3f, cls loss: %4f, bbox loss: %4f,Landmark loss :%4f,L2 loss: %4f, Total Loss: %4f ,lr:%f " % (step+1,MAX_STEP, acc, cls_loss, bbox_loss,landmark_loss, L2_loss,total_loss, lr))

                cls_loss, bbox_loss,L2_loss,lr,acc = sess.run([cls_loss_op, bbox_loss_op,L2_loss_op,lr_op,accuracy_op],
                                                             feed_dict={input_image: image_batch_array, label: label_batch_array, bbox_target: bbox_batch_array})

                total_loss = radio_cls_loss*cls_loss + radio_bbox_loss*bbox_loss + L2_loss
                print('epoch:%d/%d'%(epoch+1,end_epoch))
                print("Step: %d/%d, accuracy: %3f, cls loss: %4f, bbox loss: %4f,L2 loss: %4f, Total Loss: %4f ,lr:%f " % (step+1,MAX_STEP, acc, cls_loss, bbox_loss, L2_loss,total_loss, lr))

            #每一次epoch保留一次模型
            if i * FLAGS_2CLS.batch_size > num:
                epoch = epoch + 1
                i = 0
                path_prefix = saver.save(sess, prefix, global_step=epoch)
            writer.add_summary(summary,global_step=step)
    except tf.errors.OutOfRangeError:
        print("完成！！！")
    finally:
        coord.request_stop()
        writer.close()
    coord.join(threads)
    sess.close()


def train_multi_tfrecords(net_factory,prefix,end_epoch,base_dirs,display,base_lr, \
        suffix_list,batch_ratio,pretrained=None,resume=None,size=12,net=None,exclude_vars=None):
    '''训练模型'''
 
    if size==12:
        # net='PNet'
        radio_cls_loss = 1.0;radio_bbox_loss = 0.5;radio_landmark_loss = 0.5;
    elif size==9:
        # net='PNet_9'
        # pdb.set_trace()
        radio_cls_loss = 1.0;radio_bbox_loss = 0.5;radio_landmark_loss = 0.5;
    elif size==8:
        # net='PNet_8'
        # pdb.set_trace()
        radio_cls_loss = 1.0;radio_bbox_loss = 0.5;radio_landmark_loss = 0.5;
    elif size==24:
        # net='RNet'
        radio_cls_loss = 1.0;radio_bbox_loss = 0.5;radio_landmark_loss = 0.5;
    elif size==48:
        # net='ONet'
        radio_cls_loss = 1.0;radio_bbox_loss = 0.5;radio_landmark_loss = 1;
        
    if net=='PNet' or 'P_Net' in net:

        # pdb.set_trace()

        label_files = [osp.join(base_dir,'train_pnet_landmark.txt') for base_dir in base_dirs]
        num = 0
        for label_file in label_files:
            f = open(label_file, 'r')
            num += len(f.readlines())
            f.close()

        dataset_dirs = [osp.join(base_dir,'tfrecord/train_PNet_landmark.tfrecord_shuffle') for base_dir in base_dirs]
        batch_sizes = [int(x*FLAGS.batch_size) for x in batch_ratio]

        image_batch, label_batch, bbox_batch, landmark_batch = read_multi_tfrecords_general(dataset_dirs, batch_sizes, net)
        
        # pdb.set_trace()

    elif net=='PNet_9' or net=='PNet_8':
        #计算一共多少组数据
        label_file=os.path.join(base_dir, '../12/', 'train_pnet_landmark.txt')
        f = open(label_file, 'r')
   
        num = len(f.readlines())
        dataset_dir=os.path.join(base_dir,'tfrecord/train_{}_landmark.tfrecord_shuffle'.format(net))
        #从tfrecord读取数据
        image_batch,label_batch,bbox_batch,landmark_batch=read_single_tfrecord(dataset_dir,FLAGS.batch_size,net)
    else:

        num = 0
        for base_dir in base_dirs:
            for img_type in ['pos', 'part', 'neg']:
                label_file = osp.join(base_dir, '{}_{}.txt'.format(img_type, size))
                f = open(label_file, 'r')
                num += len(f.readlines())

    
        pos_dirs = [os.path.join(base_dir,'tfrecord/pos_landmark.tfrecord_shuffle') for base_dir in base_dirs]
        part_dirs = [os.path.join(base_dir,'tfrecord/part_landmark.tfrecord_shuffle') for base_dir in base_dirs]
        neg_dirs = [os.path.join(base_dir,'tfrecord/neg_landmark.tfrecord_shuffle') for base_dir in base_dirs]
        dataset_dirs=[pos_dirs,part_dirs,neg_dirs]

        #各数据占比
        #目的是使每一个batch的数据占比都相同
        # pos_radio,part_radio,landmark_radio,neg_radio=1.0/6,1.0/6,1.0/6,3.0/6
        pos_radio,part_radio,neg_radio=1.5/6,1.5/6,3.0/6

        pos_batch_size=int(np.ceil(FLAGS.batch_size*pos_radio))
        assert pos_batch_size != 0,"Batch Size 有误 "
        part_batch_size = int(np.ceil(FLAGS.batch_size*part_radio))
        assert part_batch_size != 0,"BBatch Size 有误 "
        neg_batch_size = int(np.ceil(FLAGS.batch_size*neg_radio))
        assert neg_batch_size != 0,"Batch Size 有误 "
        # landmark_batch_size = int(np.ceil(FLAGS.batch_size*landmark_radio))
        # assert landmark_batch_size != 0,"Batch Size 有误 "
        # batch_sizes = [pos_batch_size,part_batch_size,neg_batch_size,landmark_batch_size]
        # batch_sizes = [pos_batch_size,part_batch_size,neg_batch_size]

        batch_sizes = [
                        [int(x*pos_batch_size) for x in batch_ratio], \
                        [int(x*part_batch_size) for x in batch_ratio], \
                        [int(x*neg_batch_size) for x in batch_ratio], \
                    ]


        # pdb.set_trace()

        image_batch, label_batch, bbox_batch,landmark_batch = \
            read_multi_tfrecords_general(dataset_dirs,batch_sizes, net, list_list=True)  

    input_image=tf.placeholder(tf.float32,shape=[FLAGS.batch_size,size,size,3],name='input_image')
    label=tf.placeholder(tf.float32,shape=[FLAGS.batch_size],name='label')
    bbox_target=tf.placeholder(tf.float32,shape=[FLAGS.batch_size,4],name='bbox_target')
    # landmark_target=tf.placeholder(tf.float32,shape=[FLAGS.batch_size,10],name='landmark_target')
    #图像色相变换
    input_image=image_color_distort(input_image)
    # cls_loss_op,bbox_loss_op,landmark_loss_op,L2_loss_op,accuracy_op=net_factory(input_image,
    #                     label,bbox_target,landmark_target,training=True)
    cls_loss_op,bbox_loss_op,L2_loss_op,accuracy_op=net_factory(input_image,
                        label,bbox_target,training=True)

    # total_loss_op=radio_cls_loss*cls_loss_op+radio_bbox_loss*bbox_loss_op+            radio_landmark_loss*landmark_loss_op+L2_loss_op
    total_loss_op=radio_cls_loss*cls_loss_op+radio_bbox_loss*bbox_loss_op+L2_loss_op


    train_op,lr_op=optimize(base_lr,total_loss_op,num)
    
    
    tf.summary.scalar("cls_loss",cls_loss_op)#cls_loss
    tf.summary.scalar("bbox_loss",bbox_loss_op)#bbox_loss
    # tf.summary.scalar("landmark_loss",landmark_loss_op)#landmark_loss
    tf.summary.scalar("cls_accuracy",accuracy_op)#cls_acc
    tf.summary.scalar("total_loss",total_loss_op)#cls_loss, bbox loss, landmark loss and L2 loss add together
    summary_op = tf.summary.merge_all()

    logs_dir = "../graph_mine/%s/%s" %('-'.join(suffix_list), net)

    if os.path.exists(logs_dir) == False:
        os.makedirs(logs_dir)
    #模型训练
    ### 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True


    # init = tf.global_variables_initializer()
    # sess = tf.Session()

    ###
    sess = tf.Session(config=config)

    saver = tf.train.Saver(max_to_keep=3)


    # pdb.set_trace()
    assert pretrained is None or resume is None
    if pretrained is not None:
        model_file=tf.train.latest_checkpoint(pretrained)
        # pdb.set_trace()
        # saver.restore(sess,model_file)

        # variables = tf.global_variables()
        # variable_names = [v.name for v in variables]
        # v_id = variable_names.index('Variable:0')

        # sess.run(tf.variables_initializer([variables[v_id], ], name='finetune_init'))   # 将迭代器的计数重新置零

        if exclude_vars is None:
            exclude_vars = ['Variable']
        else:
            exclude_vars += ['Variable']

        variables_to_restore = tf.contrib.slim.get_variables_to_restore(exclude=exclude_vars)
        tf.train.init_from_checkpoint(model_file, {v.name.split(':')[0]: v for v in variables_to_restore})
        sess.run(tf.global_variables_initializer())

        start_iter = 0
        epoch = 0

        # pdb.set_trace()

    elif resume is not None:
        model_file=tf.train.latest_checkpoint(resume)
        # pdb.set_trace()
        saver.restore(sess,model_file)

        variables = tf.global_variables()
        variable_names = [v.name for v in variables]
        v_id = variable_names.index('Variable:0')

        start_iter = sess.run(variables[v_id])
        epoch = int(osp.basename(model_file).split('-')[-1])


    else:
        sess.run(tf.global_variables_initializer())

        start_iter = 0
        epoch = 0

    #模型的graph
    writer = tf.summary.FileWriter(logs_dir,sess.graph)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    i = 0

    MAX_STEP = int(num / FLAGS.batch_size + 1) * end_epoch
    # epoch = 0
    sess.graph.finalize()
    try:

        start_time = time()
        # for step in range(MAX_STEP):
        for step in range(start_iter, MAX_STEP):

            i = i + 1
            if coord.should_stop():
                break
            image_batch_array, label_batch_array, bbox_batch_array,landmark_batch_array = sess.run([image_batch, label_batch, bbox_batch,landmark_batch])

            # import pdb
            # pdb.set_trace()

            #随机翻转图像
            image_batch_array,landmark_batch_array = random_flip_images(image_batch_array,label_batch_array,landmark_batch_array)
           


            # _,_,summary = sess.run([train_op, lr_op ,summary_op], feed_dict={input_image: image_batch_array, label: label_batch_array, bbox_target: bbox_batch_array,landmark_target:landmark_batch_array})
            _,_,summary = sess.run([train_op, lr_op ,summary_op], feed_dict={input_image: image_batch_array, label: label_batch_array, bbox_target: bbox_batch_array})

            #展示训练过程
            if (step+1) % display == 0:
                end_time = time()

                time_cost = end_time - start_time
                # cls_loss, bbox_loss,landmark_loss,L2_loss,lr,acc = sess.run([cls_loss_op, bbox_loss_op,landmark_loss_op,L2_loss_op,lr_op,accuracy_op],
                #                                              feed_dict={input_image: image_batch_array, label: label_batch_array, bbox_target: bbox_batch_array, landmark_target: landmark_batch_array})

                # total_loss = radio_cls_loss*cls_loss + radio_bbox_loss*bbox_loss + radio_landmark_loss*landmark_loss + L2_loss
                # print('epoch:%d/%d'%(epoch+1,end_epoch))
                # print("Step: %d/%d, accuracy: %3f, cls loss: %4f, bbox loss: %4f,Landmark loss :%4f,L2 loss: %4f, Total Loss: %4f ,lr:%f " % (step+1,MAX_STEP, acc, cls_loss, bbox_loss,landmark_loss, L2_loss,total_loss, lr))

                cls_loss, bbox_loss,L2_loss,lr,acc = sess.run([cls_loss_op, bbox_loss_op,L2_loss_op,lr_op,accuracy_op],
                                                             feed_dict={input_image: image_batch_array, label: label_batch_array, bbox_target: bbox_batch_array})

                total_loss = radio_cls_loss*cls_loss + radio_bbox_loss*bbox_loss + L2_loss
                print('epoch:%d/%d. Time: %fs'%(epoch+1,end_epoch,time_cost))
                print("Step: %d/%d, accuracy: %3f, cls loss: %4f, bbox loss: %4f,L2 loss: %4f, Total Loss: %4f ,lr:%f " % (step+1,MAX_STEP, acc, cls_loss, bbox_loss, L2_loss,total_loss, lr))

                start_time = time()

            #每一次epoch保留一次模型
            if i * FLAGS.batch_size > num:
                epoch = epoch + 1
                i = 0
                path_prefix = saver.save(sess, prefix, global_step=epoch)
            writer.add_summary(summary,global_step=step)
    except tf.errors.OutOfRangeError:
        print("完成！！！")
    finally:
        coord.request_stop()
        writer.close()
    coord.join(threads)
    sess.close()


def train_2cls_multi_tfrecords(net_factory,prefix,end_epoch,base_dir_list,display,base_lr, \
        suffix_list,batch_ratio,pretrained=None,resume=None,size=12):
    '''训练模型'''
 
    if size==12:
        net='PNet'
        radio_cls_loss = 1.0;radio_bbox_loss = 0.5;radio_landmark_loss = 0.5;
    elif size==9:
        net='PNet_9'
        # pdb.set_trace()
        radio_cls_loss = 1.0;radio_bbox_loss = 0.5;radio_landmark_loss = 0.5;
    elif size==8:
        net='PNet_8'
        # pdb.set_trace()
        radio_cls_loss = 1.0;radio_bbox_loss = 0.5;radio_landmark_loss = 0.5;
    elif size==24:
        net='RNet'
        radio_cls_loss = 1.0;radio_bbox_loss = 0.5;radio_landmark_loss = 0.5;
    elif size==48:
        net='ONet'
        radio_cls_loss = 1.0;radio_bbox_loss = 0.5;radio_landmark_loss = 1;
        
    if net=='PNet':

        label_files = [osp.join(x, 'train_pnet_landmark.txt') for x in base_dir_list[0]+base_dir_list[1]]

        num = 0
        for label_file in label_files:
            f = open(label_file, 'r')
            # pdb.set_trace()
            num += len(f.readlines())
            f.close()
        dataset_dirs = [
                        [osp.join(x, 'tfrecord/train_PNet_landmark.tfrecord_shuffle') for x in base_dir_list[0]],
                        [osp.join(x, 'tfrecord/train_PNet_landmark.tfrecord_shuffle') for x in base_dir_list[1]],
                        ]

        batch_sizes = [
                        [int(FLAGS_2CLS.batch_size*FLAGS_2CLS.head_upperbody_ratio[0]*FLAGS_2CLS.batch_ratio[i]) \
                            for i in range(len(FLAGS_2CLS.batch_ratio))], # batch size for head
                        [int(FLAGS_2CLS.batch_size*FLAGS_2CLS.head_upperbody_ratio[1]*FLAGS_2CLS.batch_ratio[i]) \
                            for i in range(len(FLAGS_2CLS.batch_ratio))], # batch size for upperbody
                        ]

        head_image_batch,head_label_batch,head_bbox_batch,head_landmark_batch = \
            read_multi_tfrecords_general(dataset_dirs[0], batch_sizes[0], net)  

        upperbody_image_batch,upperbody_label_batch,upperbody_bbox_batch,upperbody_landmark_batch = \
            read_multi_tfrecords_general(dataset_dirs[1], batch_sizes[1], net, is_upperbody=True)  

        image_batch,label_batch,bbox_batch,landmark_batch = \
            concat_batch(
                [head_image_batch,head_label_batch,head_bbox_batch,head_landmark_batch],
                [upperbody_image_batch,upperbody_label_batch,upperbody_bbox_batch,upperbody_landmark_batch]
                )

        # pdb.set_trace()

    elif net=='PNet_9' or net=='PNet_8':
        #计算一共多少组数据
        raise "To be filled."
    else:

        cls_names = ('head', 'upperbody')
        label_files = []
        for base_dir in base_dir_list:
            label_file1=glob(os.path.join(base_dir,'pos_*_{}.txt'.format(size)))
            label_file2=glob(os.path.join(base_dir,'part_*_{}.txt'.format(size)))
            label_file3=glob(os.path.join(base_dir,'neg_*_{}.txt'.format(size)))
            label_files.extend(label_file1 + label_file2 + label_file3)

        num = 0
        for label_file in label_files:
            f = open(label_file, 'r')
            num += len(f.readlines())
            f.close()

        # pdb.set_trace()

        image_batch_list = []
        label_batch_list = []
        bbox_batch_list = []
        landmark_batch_list = []


        for i, cls_name in enumerate(cls_names):

            pos_dirs = [
                        os.path.join(base_dir,'tfrecord/pos_landmark_{}.tfrecord_shuffle'.format(cls_name)) \
                            for base_dir in base_dir_list
                        ]
            part_dirs = [
                        os.path.join(base_dir,'tfrecord/part_landmark_{}.tfrecord_shuffle'.format(cls_name)) \
                            for base_dir in base_dir_list
                        ]
            neg_dirs = [
                        os.path.join(base_dir,'tfrecord/neg_landmark_{}.tfrecord_shuffle'.format(cls_name)) \
                            for base_dir in base_dir_list
                        ]

            dataset_dirs=[pos_dirs,part_dirs,neg_dirs]

            pos_radio,part_radio,neg_radio=1.5/6,1.5/6,3.0/6

            pos_batch_size=int(np.ceil(FLAGS_2CLS.batch_size*FLAGS_2CLS.head_upperbody_ratio[i]*pos_radio))
            assert pos_batch_size != 0,"Batch Size 有误 "
            part_batch_size = int(np.ceil(FLAGS_2CLS.batch_size*FLAGS_2CLS.head_upperbody_ratio[i]*part_radio))
            assert part_batch_size != 0,"BBatch Size 有误 "
            neg_batch_size = int(np.ceil(FLAGS_2CLS.batch_size*FLAGS_2CLS.head_upperbody_ratio[i]*neg_radio))
            assert neg_batch_size != 0,"Batch Size 有误 "

            batch_sizes = [
                            [int(pos_batch_size*x) for x in FLAGS_2CLS.batch_ratio],
                            [int(part_batch_size*x) for x in FLAGS_2CLS.batch_ratio],
                            [int(neg_batch_size*x) for x in FLAGS_2CLS.batch_ratio]
                            ]

            image_batch, label_batch, bbox_batch, landmark_batch = \
                read_multi_tfrecords_general(dataset_dirs,batch_sizes, net, list_list=True) 

            # pdb.set_trace()

            image_batch_list.append(image_batch)
            label_batch_list.append(label_batch)
            bbox_batch_list.append(bbox_batch)
            landmark_batch_list.append(landmark_batch)

        image_batch,label_batch,bbox_batch,landmark_batch = \
            concat_batch(
                [x[0] for x in (image_batch_list, label_batch_list, bbox_batch_list, landmark_batch_list)],
                [x[1] for x in (image_batch_list, label_batch_list, bbox_batch_list, landmark_batch_list)]
                )

        # pdb.set_trace()

    input_image=tf.placeholder(tf.float32,shape=[FLAGS.batch_size,size,size,3],name='input_image')
    label=tf.placeholder(tf.float32,shape=[FLAGS.batch_size],name='label')
    bbox_target=tf.placeholder(tf.float32,shape=[FLAGS.batch_size,4],name='bbox_target')
    # landmark_target=tf.placeholder(tf.float32,shape=[FLAGS.batch_size,10],name='landmark_target')
    #图像色相变换
    input_image=image_color_distort(input_image)
    # cls_loss_op,bbox_loss_op,landmark_loss_op,L2_loss_op,accuracy_op=net_factory(input_image,
    #                     label,bbox_target,landmark_target,training=True)
    cls_loss_op,bbox_loss_op,L2_loss_op,accuracy_op=net_factory(input_image,
                        label,bbox_target,training=True)

    # total_loss_op=radio_cls_loss*cls_loss_op+radio_bbox_loss*bbox_loss_op+            radio_landmark_loss*landmark_loss_op+L2_loss_op
    total_loss_op=radio_cls_loss*cls_loss_op+radio_bbox_loss*bbox_loss_op+L2_loss_op


    train_op,lr_op=optimize(base_lr,total_loss_op,num)
    
    
    tf.summary.scalar("cls_loss",cls_loss_op)#cls_loss
    tf.summary.scalar("bbox_loss",bbox_loss_op)#bbox_loss
    # tf.summary.scalar("landmark_loss",landmark_loss_op)#landmark_loss
    tf.summary.scalar("cls_accuracy",accuracy_op)#cls_acc
    tf.summary.scalar("total_loss",total_loss_op)#cls_loss, bbox loss, landmark loss and L2 loss add together
    summary_op = tf.summary.merge_all()

    logs_dir = "../graph_mine/2cls/%s/%s" %('-'.join(suffix_list), net)

    if os.path.exists(logs_dir) == False:
        os.makedirs(logs_dir)
    #模型训练
    ### 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True


    init = tf.global_variables_initializer()
    # sess = tf.Session()

    ###
    sess = tf.Session(config=config)

    saver = tf.train.Saver(max_to_keep=3)


    # pdb.set_trace()

    if pretrained is not None:
        model_file=tf.train.latest_checkpoint(pretrained)
        # pdb.set_trace()
        saver.restore(sess,model_file)

        variables = tf.global_variables()
        variable_names = [v.name for v in variables]
        v_id = variable_names.index('Variable:0')

        # pdb.set_trace()

        sess.run(tf.variables_initializer([variables[v_id], ], name='finetune_init'))   # 将迭代器的计数重新置零

        # pdb.set_trace()

    else:
        sess.run(init)


    #模型的graph
    writer = tf.summary.FileWriter(logs_dir,sess.graph)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    i = 0

    MAX_STEP = int(num / FLAGS.batch_size + 1) * end_epoch
    epoch = 0
    sess.graph.finalize()
    try:

        for step in range(MAX_STEP):
            i = i + 1
            if coord.should_stop():
                break
            image_batch_array, label_batch_array, bbox_batch_array,landmark_batch_array = sess.run([image_batch, label_batch, bbox_batch,landmark_batch])

            # import pdb
            # pdb.set_trace()

            #随机翻转图像
            image_batch_array,landmark_batch_array = random_flip_images(image_batch_array,label_batch_array,landmark_batch_array)
           


            # _,_,summary = sess.run([train_op, lr_op ,summary_op], feed_dict={input_image: image_batch_array, label: label_batch_array, bbox_target: bbox_batch_array,landmark_target:landmark_batch_array})
            _,_,summary = sess.run([train_op, lr_op ,summary_op], feed_dict={input_image: image_batch_array, label: label_batch_array, bbox_target: bbox_batch_array})

            #展示训练过程
            if (step+1) % display == 0:
                # cls_loss, bbox_loss,landmark_loss,L2_loss,lr,acc = sess.run([cls_loss_op, bbox_loss_op,landmark_loss_op,L2_loss_op,lr_op,accuracy_op],
                #                                              feed_dict={input_image: image_batch_array, label: label_batch_array, bbox_target: bbox_batch_array, landmark_target: landmark_batch_array})

                # total_loss = radio_cls_loss*cls_loss + radio_bbox_loss*bbox_loss + radio_landmark_loss*landmark_loss + L2_loss
                # print('epoch:%d/%d'%(epoch+1,end_epoch))
                # print("Step: %d/%d, accuracy: %3f, cls loss: %4f, bbox loss: %4f,Landmark loss :%4f,L2 loss: %4f, Total Loss: %4f ,lr:%f " % (step+1,MAX_STEP, acc, cls_loss, bbox_loss,landmark_loss, L2_loss,total_loss, lr))

                cls_loss, bbox_loss,L2_loss,lr,acc = sess.run([cls_loss_op, bbox_loss_op,L2_loss_op,lr_op,accuracy_op],
                                                             feed_dict={input_image: image_batch_array, label: label_batch_array, bbox_target: bbox_batch_array})

                total_loss = radio_cls_loss*cls_loss + radio_bbox_loss*bbox_loss + L2_loss
                print('epoch:%d/%d'%(epoch+1,end_epoch))
                print("Step: %d/%d, accuracy: %3f, cls loss: %4f, bbox loss: %4f,L2 loss: %4f, Total Loss: %4f ,lr:%f " % (step+1,MAX_STEP, acc, cls_loss, bbox_loss, L2_loss,total_loss, lr))

            #每一次epoch保留一次模型
            if i * FLAGS.batch_size > num:
                epoch = epoch + 1
                i = 0
                path_prefix = saver.save(sess, prefix, global_step=epoch)
            writer.add_summary(summary,global_step=step)
    except tf.errors.OutOfRangeError:
        print("完成！！！")
    finally:
        coord.request_stop()
        writer.close()
    coord.join(threads)
    sess.close()


# In[5]:


def optimize(base_lr,loss,data_num):
    '''参数优化'''
    lr_factor=0.1
    global_step = tf.Variable(0, trainable=False)
    boundaries = [int(epoch * data_num / FLAGS.batch_size) for epoch in FLAGS.LR_EPOCH]
    lr_values = [base_lr * (lr_factor ** x) for x in range(0, len(FLAGS.LR_EPOCH) + 1)]
    lr_op = tf.train.piecewise_constant(global_step, boundaries, lr_values)
    optimizer = tf.train.MomentumOptimizer(lr_op, 0.9)
    train_op = optimizer.minimize(loss, global_step)
    return train_op, lr_op


# In[2]:


def read_single_tfrecord(tfrecord_file,batch_size,net, is_upperbody=False, aspect=None):
    '''
        读取tfrecord数据，如果is_upperbody为真(仅在训练2 class MTCNN的时候才需要使用)，
        需要对label进行修改（1->2, -1->-2）

    '''
    filename_queue=tf.train.string_input_producer([tfrecord_file],shuffle=True)
    reader=tf.TFRecordReader()
    _,serialized_example=reader.read(filename_queue)
    image_features=tf.parse_single_example(serialized_example,
                        features={
                        'image/encoded': tf.FixedLenFeature([], tf.string),
                        'image/label': tf.FixedLenFeature([], tf.int64),
                        'image/roi': tf.FixedLenFeature([4], tf.float32),
                        'image/landmark': tf.FixedLenFeature([10],tf.float32)
                    }
                )
    # pdb.set_trace()
    if net=='PNet' or 'P_Net' in net:
        image_size=12
    elif net=='PNet_9':
        image_size=9
    elif net=='PNet_8':
        image_size=8
    elif net=='RNet' or 'R_Net' in net:
        image_size=24
    elif net=='ONet' or 'O_Net' in net:
        image_size=48
    image=tf.decode_raw(image_features['image/encoded'],tf.uint8)

    # pdb.set_trace()
    if aspect is None:
        image=tf.reshape(image,[image_size,image_size,3])
    else:
        # pdb.set_trace()
        image=tf.reshape(image,[aspect[0],aspect[1],3])
    # pdb.set_trace()


    #将值规划在[-1,1]内
    image=(tf.cast(image,tf.float32)-127.5)/128
    
    label=tf.cast(image_features['image/label'],tf.float32)
    roi=tf.cast(image_features['image/roi'],tf.float32)
    landmark=tf.cast(image_features['image/landmark'],tf.float32)
    # image,label,roi,landmark=tf.train.batch([image,label,roi,landmark],
    #                                        batch_size=batch_size,
    #                                        num_threads=2,
    #                                        capacity=batch_size)

    image,label,roi,landmark=tf.train.batch([image,label,roi,landmark],
                                           batch_size=batch_size,
                                           num_threads=16,
                                           capacity=16*batch_size)

    # image,label,roi=tf.train.batch([image,label,roi],
    #                                        batch_size=batch_size,
    #                                        num_threads=16,
    #                                        capacity=16*batch_size)

    label=tf.reshape(label,[batch_size])
    roi=tf.reshape(roi,[batch_size,4])
    # landmark=tf.reshape(landmark,[batch_size,10])

    # pdb.set_trace()

    if is_upperbody:

        twos = tf.add(tf.zeros_like(label), 2)
        neg_twos = tf.add(tf.zeros_like(label), -2)

        # pdb.set_trace()

        label = tf.where(tf.equal(label, 1), twos, label)
        label = tf.where(tf.equal(label, -1), neg_twos, label)

        # label = tf.cast(label, tf.int32)

        # pdb.set_trace()



    return image,label,roi,landmark
    # return image,label,roi,_



# In[3]:


def read_multi_tfrecords(tfrecord_files, batch_sizes, net):
    '''读取多个tfrecord文件放一起'''
    pos_dir,part_dir,neg_dir,landmark_dir = tfrecord_files
    pos_batch_size,part_batch_size,neg_batch_size,landmark_batch_size = batch_sizes
   
    pos_image,pos_label,pos_roi,pos_landmark = read_single_tfrecord(pos_dir, pos_batch_size, net)
  
    part_image,part_label,part_roi,part_landmark = read_single_tfrecord(part_dir, part_batch_size, net)
  
    neg_image,neg_label,neg_roi,neg_landmark = read_single_tfrecord(neg_dir, neg_batch_size, net)

    landmark_image,landmark_label,landmark_roi,landmark_landmark = read_single_tfrecord(landmark_dir, landmark_batch_size, net)
 

    images = tf.concat([pos_image,part_image,neg_image,landmark_image], 0, name="concat/image")
   
    labels = tf.concat([pos_label,part_label,neg_label,landmark_label],0,name="concat/label")
 
    assert isinstance(labels, object)

    rois = tf.concat([pos_roi,part_roi,neg_roi,landmark_roi],0,name="concat/roi")
    
    landmarks = tf.concat([pos_landmark,part_landmark,neg_landmark,landmark_landmark],0,name="concat/landmark")
    return images,labels,rois,landmarks
    

def read_multi_tfrecords_xiao(tfrecord_files, batch_sizes, net):
    '''读取多个tfrecord文件放一起'''
    # pos_dir,part_dir,neg_dir,landmark_dir = tfrecord_files
    # pos_batch_size,part_batch_size,neg_batch_size,landmark_batch_size = batch_sizes
    pos_dir,part_dir,neg_dir = tfrecord_files
    pos_batch_size,part_batch_size,neg_batch_size = batch_sizes 

    pos_image,pos_label,pos_roi,pos_landmark = read_single_tfrecord(pos_dir, pos_batch_size, net)
  
    part_image,part_label,part_roi,part_landmark = read_single_tfrecord(part_dir, part_batch_size, net)
  
    neg_image,neg_label,neg_roi,neg_landmark = read_single_tfrecord(neg_dir, neg_batch_size, net)

    # landmark_image,landmark_label,landmark_roi,landmark_landmark = read_single_tfrecord(landmark_dir, landmark_batch_size, net)
 

    # images = tf.concat([pos_image,part_image,neg_image,landmark_image], 0, name="concat/image")
   
    # labels = tf.concat([pos_label,part_label,neg_label,landmark_label],0,name="concat/label")
 
    images = tf.concat([pos_image,part_image,neg_image], 0, name="concat/image")
   
    labels = tf.concat([pos_label,part_label,neg_label],0,name="concat/label")

    assert isinstance(labels, object)

    # rois = tf.concat([pos_roi,part_roi,neg_roi,landmark_roi],0,name="concat/roi")
    
    # landmarks = tf.concat([pos_landmark,part_landmark,neg_landmark,landmark_landmark],0,name="concat/landmark")
    
    rois = tf.concat([pos_roi,part_roi,neg_roi],0,name="concat/roi")
    
    landmarks = tf.concat([pos_landmark,part_landmark,neg_landmark],0,name="concat/landmark")

    return images,labels,rois,landmarks


def read_multi_tfrecords_general(tfrecord_files, batch_sizes, net, \
    list_list=False, is_upperbody=False):
    '''读取多个tfrecord文件放一起'''

    res_list = []

    if list_list:
        tfrecord_files = [x[i] for x in tfrecord_files for i in range(len(x))]
        batch_sizes = [x[i] for x in batch_sizes for i in range(len(x))]


    for i, tfrecord_file in enumerate(tfrecord_files):
        res_cur = read_single_tfrecord(tfrecord_file, batch_sizes[i], net)
        res_list.append(res_cur)

    images = tf.concat([res[0] for res in res_list], 0, name="concat/image")
   
    labels = tf.concat([res[1] for res in res_list], 0, name="concat/label")

    assert isinstance(labels, object)

    rois = tf.concat([res[2] for res in res_list], 0, name="concat/roi")
    
    landmarks = tf.concat([res[3] for res in res_list], 0, name="concat/landmark")

    if is_upperbody:

        twos = tf.add(tf.zeros_like(labels), 2)
        neg_twos = tf.add(tf.zeros_like(labels), -2)

        # pdb.set_trace()

        labels = tf.where(tf.equal(labels, 1), twos, labels)
        labels = tf.where(tf.equal(labels, -1), neg_twos, labels)

        # label = tf.cast(label, tf.int32)

        # pdb.set_trace()


    return images,labels,rois,landmarks


    

    



# In[4]:


def image_color_distort(inputs):
    inputs = tf.image.random_contrast(inputs, lower=0.5, upper=1.5)
    inputs = tf.image.random_brightness(inputs, max_delta=0.2)
    inputs = tf.image.random_hue(inputs,max_delta= 0.2)
    inputs = tf.image.random_saturation(inputs,lower = 0.5, upper= 1.5)

    return inputs


# In[6]:


def random_flip_images(image_batch,label_batch,landmark_batch):
    '''随机翻转图像'''
    if random.choice([0,1]) > 0:
        num_images = image_batch.shape[0]
        fliplandmarkindexes = np.where(label_batch==-2)[0]
        flipposindexes = np.where(label_batch==1)[0]
        
        flipindexes = np.concatenate((fliplandmarkindexes,flipposindexes))
          
        for i in flipindexes:
            cv2.flip(image_batch[i],1,image_batch[i])        
        
           
        for i in fliplandmarkindexes:
            landmark_ = landmark_batch[i].reshape((-1,2))
            landmark_ = np.asarray([(1-x, y) for (x, y) in landmark_])
            landmark_[[0, 1]] = landmark_[[1, 0]]
            landmark_[[3, 4]] = landmark_[[4, 3]]       
            landmark_batch[i] = landmark_.ravel()
        
    return image_batch,landmark_batch

def concat_batch(batches_1, batches_2):

    len_1 = len(batches_1)
    len_2 = len(batches_2)

    assert len_1 == len_2

    res = (tf.concat([batches_1[i], batches_2[i]], 0, name="concat/batch/{}".format(i)) \
        for i in range(len_1))

    return res



