
# coding: utf-8

# In[1]:


import tensorflow as tf
slim=tf.contrib.slim
import numpy as np
#只把70%数据用作参数更新
num_keep_radio=0.7


# In[7]:
def P_Net_org(inputs,label=None,bbox_target=None,landmark_target=None,training=True):
    '''pnet的结构'''
    with tf.variable_scope('PNet'):
        with slim.arg_scope([slim.conv2d],activation_fn=prelu,
                           weights_initializer=slim.xavier_initializer(),
                           weights_regularizer=slim.l2_regularizer(0.0005),
                           padding='VALID'):
            net=slim.conv2d(inputs,10,3,scope='conv1')
            net=slim.max_pool2d(net,kernel_size=[2,2],stride=2,padding='SAME',scope='pool1')
            net=slim.conv2d(net,16,3,scope='conv2')
            net=slim.conv2d(net,32,3,scope='conv3')
            #二分类输出通道数为2
            conv4_1=slim.conv2d(net,2,1,activation_fn=tf.nn.softmax,scope='conv4_1')
            bbox_pred=slim.conv2d(net,4,1,activation_fn=None,scope='conv4_2')
            landmark_pred=slim.conv2d(net,10,1,activation_fn=None,scope='conv4_3')
            
            if training:
                cls_prob=tf.squeeze(conv4_1,[1,2],name='cls_prob')#[batch,2]
                cls_loss=cls_ohem(cls_prob,label)
                
                bbox_pred=tf.squeeze(bbox_pred,[1,2],name='bbox_pred')#[bacth,4]
                bbox_loss=bbox_ohem(bbox_pred,bbox_target,label)
                
                landmark_pred=tf.squeeze(landmark_pred,[1,2],name='landmark_pred')#[batch,10]
                landmark_loss=landmark_ohem(landmark_pred,landmark_target,label)
                
                accuracy=cal_accuracy(cls_prob,label)
                L2_loss=tf.add_n(slim.losses.get_regularization_losses())
                return cls_loss,bbox_loss,landmark_loss,L2_loss,accuracy
            else:
                #测试时batch_size=1
                cls_pro_test=tf.squeeze(conv4_1,axis=0)
                bbox_pred_test=tf.squeeze(bbox_pred,axis=0)
                landmark_pred_test=tf.squeeze(landmark_pred,axis=0)
                return cls_pro_test,bbox_pred_test,landmark_pred_test


def R_Net_org(inputs,label=None,bbox_target=None,landmark_target=None,training=True):
    '''RNet结构'''
    with tf.variable_scope('RNet'):
        with slim.arg_scope([slim.conv2d],
                           activation_fn=prelu,
                           weights_initializer=slim.xavier_initializer(),
                           weights_regularizer=slim.l2_regularizer(0.0005),
                           padding='VALID'):
            net=slim.conv2d(inputs,28,3,scope='conv1')
            net=slim.max_pool2d(net,kernel_size=[3,3],stride=2,padding='SAME',scope='pool1')
            net=slim.conv2d(net,48,3,scope='conv2')
            net=slim.max_pool2d(net,kernel_size=[3,3],stride=2,scope='pool2')
            net=slim.conv2d(net,64,2,scope='conv3')
            fc_flatten=slim.flatten(net)
            fc1=slim.fully_connected(fc_flatten,num_outputs=128,scope='fc1')
            
            cls_prob=slim.fully_connected(fc1,num_outputs=2,activation_fn=tf.nn.softmax,scope='cls_fc')
            bbox_pred=slim.fully_connected(fc1,num_outputs=4,activation_fn=None,scope='bbox_fc')
            landmark_pred=slim.fully_connected(fc1,num_outputs=10,activation_fn=None,scope='landmark_fc')
            if training:
                cls_loss=cls_ohem(cls_prob,label)
                
                bbox_loss=bbox_ohem(bbox_pred,bbox_target,label)
                
                landmark_loss=landmark_ohem(landmark_pred,landmark_target,label)
                
                accuracy=cal_accuracy(cls_prob,label)
                L2_loss=tf.add_n(slim.losses.get_regularization_losses())
                return cls_loss,bbox_loss,landmark_loss,L2_loss,accuracy
            else:
                return cls_prob,bbox_pred,landmark_pred



def P_Net(inputs,label=None,bbox_target=None,landmark_target=None,training=True):
    '''pnet的结构'''
    with tf.variable_scope('PNet'):
        with slim.arg_scope([slim.conv2d],activation_fn=prelu,
                           weights_initializer=slim.xavier_initializer(),
                           weights_regularizer=slim.l2_regularizer(0.0005),
                           padding='VALID'):
            net=slim.conv2d(inputs,10,3,scope='conv1')
            net=slim.max_pool2d(net,kernel_size=[2,2],stride=2,padding='SAME',scope='pool1')
            net=slim.conv2d(net,16,3,scope='conv2')
            net=slim.conv2d(net,32,3,scope='conv3')
            #二分类输出通道数为2
            conv4_1=slim.conv2d(net,2,1,activation_fn=tf.nn.softmax,scope='conv4_1')
            bbox_pred=slim.conv2d(net,4,1,activation_fn=None,scope='conv4_2')
            # landmark_pred=slim.conv2d(net,10,1,activation_fn=None,scope='conv4_3')
            
            if training:
                cls_prob=tf.squeeze(conv4_1,[1,2],name='cls_prob')#[batch,2]
                cls_loss=cls_ohem(cls_prob,label)
                
                bbox_pred=tf.squeeze(bbox_pred,[1,2],name='bbox_pred')#[bacth,4]
                bbox_loss=bbox_ohem(bbox_pred,bbox_target,label)
                
                # landmark_pred=tf.squeeze(landmark_pred,[1,2],name='landmark_pred')#[batch,10]
                # landmark_loss=landmark_ohem(landmark_pred,landmark_target,label)
                
                accuracy=cal_accuracy(cls_prob,label)
                L2_loss=tf.add_n(slim.losses.get_regularization_losses())
                # return cls_loss,bbox_loss,landmark_loss,L2_loss,accuracy
                return cls_loss,bbox_loss,L2_loss,accuracy

            else:
                #测试时batch_size=1
                cls_pro_test=tf.squeeze(conv4_1,axis=0)
                bbox_pred_test=tf.squeeze(bbox_pred,axis=0)
                # landmark_pred_test=tf.squeeze(landmark_pred,axis=0)
                # return cls_pro_test,bbox_pred_test,landmark_pred_test
                return cls_pro_test,bbox_pred_test,None



def P_Net_9(inputs,label=None,bbox_target=None,landmark_target=None,training=True):
    '''pnet_9的结构'''
    with tf.variable_scope('PNet_9'):
        with slim.arg_scope([slim.conv2d],activation_fn=prelu,
                           weights_initializer=slim.xavier_initializer(),
                           weights_regularizer=slim.l2_regularizer(0.0005),
                           padding='VALID'):
            net=slim.conv2d(inputs,10,3,scope='conv1')
            # net=slim.max_pool2d(net,kernel_size=[2,2],stride=2,padding='SAME',scope='pool1')
            net=slim.conv2d(net,10,3,scope='conv1_1')

            net=slim.conv2d(net,16,3,scope='conv2')
            net=slim.conv2d(net,32,3,scope='conv3')
            #二分类输出通道数为2
            conv4_1=slim.conv2d(net,2,1,activation_fn=tf.nn.softmax,scope='conv4_1')
            bbox_pred=slim.conv2d(net,4,1,activation_fn=None,scope='conv4_2')
            # landmark_pred=slim.conv2d(net,10,1,activation_fn=None,scope='conv4_3')
            
            if training:
                cls_prob=tf.squeeze(conv4_1,[1,2],name='cls_prob')#[batch,2]
                cls_loss=cls_ohem(cls_prob,label)
                
                bbox_pred=tf.squeeze(bbox_pred,[1,2],name='bbox_pred')#[bacth,4]
                bbox_loss=bbox_ohem(bbox_pred,bbox_target,label)
                
                # landmark_pred=tf.squeeze(landmark_pred,[1,2],name='landmark_pred')#[batch,10]
                # landmark_loss=landmark_ohem(landmark_pred,landmark_target,label)
                
                accuracy=cal_accuracy(cls_prob,label)
                L2_loss=tf.add_n(slim.losses.get_regularization_losses())
                # return cls_loss,bbox_loss,landmark_loss,L2_loss,accuracy
                return cls_loss,bbox_loss,L2_loss,accuracy

            else:
                #测试时batch_size=1
                cls_pro_test=tf.squeeze(conv4_1,axis=0)
                bbox_pred_test=tf.squeeze(bbox_pred,axis=0)
                # landmark_pred_test=tf.squeeze(landmark_pred,axis=0)
                # return cls_pro_test,bbox_pred_test,landmark_pred_test
                return cls_pro_test,bbox_pred_test,None


def P_Net_8(inputs,label=None,bbox_target=None,landmark_target=None,training=True):
    '''pnet_8的结构'''
    with tf.variable_scope('PNet_8'):
        with slim.arg_scope([slim.conv2d],activation_fn=prelu,
                           weights_initializer=slim.xavier_initializer(),
                           weights_regularizer=slim.l2_regularizer(0.0005),
                           padding='VALID'):
            net=slim.conv2d(inputs,10,3,scope='conv1')
            net=slim.max_pool2d(net,kernel_size=[2,2],stride=2,padding='SAME',scope='pool1')
            # net=slim.conv2d(net,16,3,scope='conv2')
            net=slim.conv2d(net,32,3,scope='conv3')
            #二分类输出通道数为2
            conv4_1=slim.conv2d(net,2,1,activation_fn=tf.nn.softmax,scope='conv4_1')
            bbox_pred=slim.conv2d(net,4,1,activation_fn=None,scope='conv4_2')
            # landmark_pred=slim.conv2d(net,10,1,activation_fn=None,scope='conv4_3')
            
            if training:
                cls_prob=tf.squeeze(conv4_1,[1,2],name='cls_prob')#[batch,2]
                cls_loss=cls_ohem(cls_prob,label)
                
                bbox_pred=tf.squeeze(bbox_pred,[1,2],name='bbox_pred')#[bacth,4]
                bbox_loss=bbox_ohem(bbox_pred,bbox_target,label)
                
                # landmark_pred=tf.squeeze(landmark_pred,[1,2],name='landmark_pred')#[batch,10]
                # landmark_loss=landmark_ohem(landmark_pred,landmark_target,label)
                
                accuracy=cal_accuracy(cls_prob,label)
                L2_loss=tf.add_n(slim.losses.get_regularization_losses())
                # return cls_loss,bbox_loss,landmark_loss,L2_loss,accuracy
                return cls_loss,bbox_loss,L2_loss,accuracy

            else:
                #测试时batch_size=1
                cls_pro_test=tf.squeeze(conv4_1,axis=0)
                bbox_pred_test=tf.squeeze(bbox_pred,axis=0)
                # landmark_pred_test=tf.squeeze(landmark_pred,axis=0)
                # return cls_pro_test,bbox_pred_test,landmark_pred_test
                return cls_pro_test,bbox_pred_test,None




# In[8]:


def R_Net(inputs,label=None,bbox_target=None,landmark_target=None,training=True):
    '''RNet结构'''
    with tf.variable_scope('RNet'):
        with slim.arg_scope([slim.conv2d],
                           activation_fn=prelu,
                           weights_initializer=slim.xavier_initializer(),
                           weights_regularizer=slim.l2_regularizer(0.0005),
                           padding='VALID'):
            net=slim.conv2d(inputs,28,3,scope='conv1')
            net=slim.max_pool2d(net,kernel_size=[3,3],stride=2,padding='SAME',scope='pool1')
            net=slim.conv2d(net,48,3,scope='conv2')
            net=slim.max_pool2d(net,kernel_size=[3,3],stride=2,scope='pool2')
            net=slim.conv2d(net,64,2,scope='conv3')
            fc_flatten=slim.flatten(net)
            fc1=slim.fully_connected(fc_flatten,num_outputs=128,scope='fc1')
            
            cls_prob=slim.fully_connected(fc1,num_outputs=2,activation_fn=tf.nn.softmax,scope='cls_fc')
            bbox_pred=slim.fully_connected(fc1,num_outputs=4,activation_fn=None,scope='bbox_fc')
            # landmark_pred=slim.fully_connected(fc1,num_outputs=10,activation_fn=None,scope='landmark_fc')
            if training:
                cls_loss=cls_ohem(cls_prob,label)
                
                bbox_loss=bbox_ohem(bbox_pred,bbox_target,label)
                
                # landmark_loss=landmark_ohem(landmark_pred,landmark_target,label)
                
                accuracy=cal_accuracy(cls_prob,label)
                L2_loss=tf.add_n(slim.losses.get_regularization_losses())
                # return cls_loss,bbox_loss,landmark_loss,L2_loss,accuracy
                return cls_loss,bbox_loss,L2_loss,accuracy

            else:
                # return cls_prob,bbox_pred,landmark_pred
                return cls_prob,bbox_pred,None



# In[9]:


def O_Net(inputs,label=None,bbox_target=None,landmark_target=None,training=True):
    '''ONet结构'''
    with tf.variable_scope('ONet'):
        with slim.arg_scope([slim.conv2d],
                           activation_fn=prelu,
                           weights_initializer=slim.xavier_initializer(),
                           weights_regularizer=slim.l2_regularizer(0.0005),
                           padding='VALID'):
            net=slim.conv2d(inputs,32,3,scope='conv1')
            net=slim.max_pool2d(net,kernel_size=[3,3],stride=2,padding='SAME',scope='pool1')
            net=slim.conv2d(net,64,3,scope='conv2')
            net=slim.max_pool2d(net,kernel_size=[3,3],stride=2,scope='pool2')
            net=slim.conv2d(net,64,3,scope='conv3')
            net=slim.max_pool2d(net,kernel_size=[2,2],stride=2,padding='SAME',scope='pool3')
            net=slim.conv2d(net,128,2,scope='conv4')
            fc_flatten=slim.flatten(net)
            fc1=slim.fully_connected(fc_flatten,num_outputs=256,scope='fc1')
            
            cls_prob=slim.fully_connected(fc1,num_outputs=2,activation_fn=tf.nn.softmax,scope='cls_fc')
            bbox_pred=slim.fully_connected(fc1,num_outputs=4,activation_fn=None,scope='bbox_fc')
            # landmark_pred=slim.fully_connected(fc1,num_outputs=10,activation_fn=None,scope='landmark_fc')
            if training:
                cls_loss=cls_ohem(cls_prob,label)
                
                bbox_loss=bbox_ohem(bbox_pred,bbox_target,label)
                
                # landmark_loss=landmark_ohem(landmark_pred,landmark_target,label)
                
                accuracy=cal_accuracy(cls_prob,label)
                L2_loss=tf.add_n(slim.losses.get_regularization_losses())
                # return cls_loss,bbox_loss,landmark_loss,L2_loss,accuracy
                return cls_loss,bbox_loss,L2_loss,accuracy
            else:
                # return cls_prob,bbox_pred,landmark_pred
                return cls_prob,bbox_pred,None




def P_Net_2cls(inputs,label=None,bbox_target=None,landmark_target=None,training=True):
    '''pnet的结构'''
    with tf.variable_scope('PNet'):
        with slim.arg_scope([slim.conv2d],activation_fn=prelu,
                           weights_initializer=slim.xavier_initializer(),
                           weights_regularizer=slim.l2_regularizer(0.0005),
                           padding='VALID'):
            net=slim.conv2d(inputs,10,3,scope='conv1')
            net=slim.max_pool2d(net,kernel_size=[2,2],stride=2,padding='SAME',scope='pool1')
            net=slim.conv2d(net,16,3,scope='conv2')
            net=slim.conv2d(net,32,3,scope='conv3')
            #三分类 bg, head, upperbody
            conv4_1=slim.conv2d(net,3,1,activation_fn=tf.nn.softmax,scope='conv4_1')
            bbox_pred=slim.conv2d(net,8,1,activation_fn=None,scope='conv4_2')
            # landmark_pred=slim.conv2d(net,10,1,activation_fn=None,scope='conv4_3')
            
            if training:
                cls_prob=tf.squeeze(conv4_1,[1,2],name='cls_prob')#[batch,3]
                cls_loss=cls_ohem_2cls(cls_prob,label)
                
                bbox_pred=tf.squeeze(bbox_pred,[1,2],name='bbox_pred')#[bacth,8]
                bbox_loss=bbox_ohem_2cls(bbox_pred,bbox_target,label)
                
                # landmark_pred=tf.squeeze(landmark_pred,[1,2],name='landmark_pred')#[batch,10]
                # landmark_loss=landmark_ohem(landmark_pred,landmark_target,label)
                
                accuracy=cal_accuracy_2cls(cls_prob,label)
                L2_loss=tf.add_n(slim.losses.get_regularization_losses())
                # return cls_loss,bbox_loss,landmark_loss,L2_loss,accuracy
                return cls_loss,bbox_loss,L2_loss,accuracy

            else:
                #测试时batch_size=1
                cls_pro_test=tf.squeeze(conv4_1,axis=0)
                bbox_pred_test=tf.squeeze(bbox_pred,axis=0)
                # landmark_pred_test=tf.squeeze(landmark_pred,axis=0)
                # return cls_pro_test,bbox_pred_test,landmark_pred_test
                return cls_pro_test,bbox_pred_test,None




def R_Net_2cls(inputs,label=None,bbox_target=None,landmark_target=None,training=True):
    '''RNet结构'''
    with tf.variable_scope('RNet'):
        with slim.arg_scope([slim.conv2d],
                           activation_fn=prelu,
                           weights_initializer=slim.xavier_initializer(),
                           weights_regularizer=slim.l2_regularizer(0.0005),
                           padding='VALID'):
            net=slim.conv2d(inputs,28,3,scope='conv1')
            net=slim.max_pool2d(net,kernel_size=[3,3],stride=2,padding='SAME',scope='pool1')
            net=slim.conv2d(net,48,3,scope='conv2')
            net=slim.max_pool2d(net,kernel_size=[3,3],stride=2,scope='pool2')
            net=slim.conv2d(net,64,2,scope='conv3')
            fc_flatten=slim.flatten(net)
            fc1=slim.fully_connected(fc_flatten,num_outputs=128,scope='fc1')
            
            cls_prob=slim.fully_connected(fc1,num_outputs=3,activation_fn=tf.nn.softmax,scope='cls_fc')
            bbox_pred=slim.fully_connected(fc1,num_outputs=8,activation_fn=None,scope='bbox_fc')
            # landmark_pred=slim.fully_connected(fc1,num_outputs=10,activation_fn=None,scope='landmark_fc')
            if training:
                cls_loss=cls_ohem_2cls(cls_prob,label)
                
                bbox_loss=bbox_ohem_2cls(bbox_pred,bbox_target,label)
                
                # landmark_loss=landmark_ohem(landmark_pred,landmark_target,label)
                
                accuracy=cal_accuracy_2cls(cls_prob,label)
                L2_loss=tf.add_n(slim.losses.get_regularization_losses())
                # return cls_loss,bbox_loss,landmark_loss,L2_loss,accuracy
                return cls_loss,bbox_loss,L2_loss,accuracy

            else:
                # return cls_prob,bbox_pred,landmark_pred
                return cls_prob,bbox_pred,None



def O_Net_2cls(inputs,label=None,bbox_target=None,landmark_target=None,training=True):
    '''ONet结构'''
    with tf.variable_scope('ONet'):
        with slim.arg_scope([slim.conv2d],
                           activation_fn=prelu,
                           weights_initializer=slim.xavier_initializer(),
                           weights_regularizer=slim.l2_regularizer(0.0005),
                           padding='VALID'):
            net=slim.conv2d(inputs,32,3,scope='conv1')
            net=slim.max_pool2d(net,kernel_size=[3,3],stride=2,padding='SAME',scope='pool1')
            net=slim.conv2d(net,64,3,scope='conv2')
            net=slim.max_pool2d(net,kernel_size=[3,3],stride=2,scope='pool2')
            net=slim.conv2d(net,64,3,scope='conv3')
            net=slim.max_pool2d(net,kernel_size=[2,2],stride=2,padding='SAME',scope='pool3')
            net=slim.conv2d(net,128,2,scope='conv4')
            fc_flatten=slim.flatten(net)
            fc1=slim.fully_connected(fc_flatten,num_outputs=256,scope='fc1')
            
            cls_prob=slim.fully_connected(fc1,num_outputs=3,activation_fn=tf.nn.softmax,scope='cls_fc')
            bbox_pred=slim.fully_connected(fc1,num_outputs=8,activation_fn=None,scope='bbox_fc')
            # landmark_pred=slim.fully_connected(fc1,num_outputs=10,activation_fn=None,scope='landmark_fc')
            if training:
                cls_loss=cls_ohem_2cls(cls_prob,label)
                
                bbox_loss=bbox_ohem_2cls(bbox_pred,bbox_target,label)
                
                # landmark_loss=landmark_ohem(landmark_pred,landmark_target,label)
                
                accuracy=cal_accuracy_2cls(cls_prob,label)
                L2_loss=tf.add_n(slim.losses.get_regularization_losses())
                # return cls_loss,bbox_loss,landmark_loss,L2_loss,accuracy
                return cls_loss,bbox_loss,L2_loss,accuracy
            else:
                # return cls_prob,bbox_pred,landmark_pred
                return cls_prob,bbox_pred,None



def R_Net_conv3(inputs,label=None,bbox_target=None,landmark_target=None,training=True):
    '''RNet结构，将原始的2*2卷积改成3*3卷积'''
    with tf.variable_scope('RNet'):
        with slim.arg_scope([slim.conv2d],
                           activation_fn=prelu,
                           weights_initializer=slim.xavier_initializer(),
                           weights_regularizer=slim.l2_regularizer(0.0005),
                           padding='VALID'):
            net=slim.conv2d(inputs,28,3,scope='conv1')
            net=slim.max_pool2d(net,kernel_size=[3,3],stride=2,padding='SAME',scope='pool1')
            net=slim.conv2d(net,48,3,scope='conv2')
            net=slim.max_pool2d(net,kernel_size=[3,3],stride=2,scope='pool2')
            # net=slim.conv2d(net,64,2,scope='conv3')
            net=slim.conv2d(net,64,3,scope='conv3')

            fc_flatten=slim.flatten(net)
            fc1=slim.fully_connected(fc_flatten,num_outputs=128,scope='fc1')
            
            cls_prob=slim.fully_connected(fc1,num_outputs=2,activation_fn=tf.nn.softmax,scope='cls_fc')
            bbox_pred=slim.fully_connected(fc1,num_outputs=4,activation_fn=None,scope='bbox_fc')
            # landmark_pred=slim.fully_connected(fc1,num_outputs=10,activation_fn=None,scope='landmark_fc')
            if training:
                cls_loss=cls_ohem(cls_prob,label)
                
                bbox_loss=bbox_ohem(bbox_pred,bbox_target,label)
                
                # landmark_loss=landmark_ohem(landmark_pred,landmark_target,label)
                
                accuracy=cal_accuracy(cls_prob,label)
                L2_loss=tf.add_n(slim.losses.get_regularization_losses())
                # return cls_loss,bbox_loss,landmark_loss,L2_loss,accuracy
                return cls_loss,bbox_loss,L2_loss,accuracy

            else:
                # return cls_prob,bbox_pred,landmark_pred
                return cls_prob,bbox_pred,None


def O_Net_conv3(inputs,label=None,bbox_target=None,landmark_target=None,training=True):
    '''ONet结构，将原始的2*2卷积改成3*3卷积'''
    with tf.variable_scope('ONet'):
        with slim.arg_scope([slim.conv2d],
                           activation_fn=prelu,
                           weights_initializer=slim.xavier_initializer(),
                           weights_regularizer=slim.l2_regularizer(0.0005),
                           padding='VALID'):
            net=slim.conv2d(inputs,32,3,scope='conv1')
            net=slim.max_pool2d(net,kernel_size=[3,3],stride=2,padding='SAME',scope='pool1')
            net=slim.conv2d(net,64,3,scope='conv2')
            net=slim.max_pool2d(net,kernel_size=[3,3],stride=2,scope='pool2')
            net=slim.conv2d(net,64,3,scope='conv3')
            net=slim.max_pool2d(net,kernel_size=[2,2],stride=2,padding='SAME',scope='pool3')
            # net=slim.conv2d(net,128,2,scope='conv4')
            net=slim.conv2d(net,128,3,scope='conv4')

            fc_flatten=slim.flatten(net)
            fc1=slim.fully_connected(fc_flatten,num_outputs=256,scope='fc1')
            
            cls_prob=slim.fully_connected(fc1,num_outputs=2,activation_fn=tf.nn.softmax,scope='cls_fc')
            bbox_pred=slim.fully_connected(fc1,num_outputs=4,activation_fn=None,scope='bbox_fc')
            # landmark_pred=slim.fully_connected(fc1,num_outputs=10,activation_fn=None,scope='landmark_fc')
            if training:
                cls_loss=cls_ohem(cls_prob,label)
                
                bbox_loss=bbox_ohem(bbox_pred,bbox_target,label)
                
                # landmark_loss=landmark_ohem(landmark_pred,landmark_target,label)
                
                accuracy=cal_accuracy(cls_prob,label)
                L2_loss=tf.add_n(slim.losses.get_regularization_losses())
                # return cls_loss,bbox_loss,landmark_loss,L2_loss,accuracy
                return cls_loss,bbox_loss,L2_loss,accuracy
            else:
                # return cls_prob,bbox_pred,landmark_pred
                return cls_prob,bbox_pred,None


def P_Net_v1(inputs,label=None,bbox_target=None,landmark_target=None,training=True):
    '''
        在全志芯片上的版本。主要有下面的特点：
        1. conv层的激活函数从prelu改为relu；
        2. 将pooling去掉，把stride放到conv里面；
        3. 卷积核数量改为8的倍数；
        4. 把conv_4_1的softmax去掉，在cls_ohem中使用softmax；
    '''
    with tf.variable_scope('PNet'):
        with slim.arg_scope([slim.conv2d],activation_fn=tf.nn.relu,
                           weights_initializer=slim.xavier_initializer(),
                           weights_regularizer=slim.l2_regularizer(0.0005),
                           padding='VALID'):
            net=slim.conv2d(inputs,8,3,stride=2,scope='conv1')
            net=slim.conv2d(net,16,3,scope='conv2')
            net=slim.conv2d(net,32,3,scope='conv3')
            #二分类输出通道数为2
            conv4_1=slim.conv2d(net,2,1,activation_fn=None,scope='conv4_1')
            bbox_pred=slim.conv2d(net,4,1,activation_fn=None,scope='conv4_2')
            
            if training:
                cls_prob=tf.squeeze(conv4_1,[1,2],name='cls_prob')#[batch,2]
                cls_loss=cls_ohem(cls_prob,label,use_softmax=True)
                
                bbox_pred=tf.squeeze(bbox_pred,[1,2],name='bbox_pred')#[bacth,4]
                bbox_loss=bbox_ohem(bbox_pred,bbox_target,label)
                
                accuracy=cal_accuracy(cls_prob,label)
                L2_loss=tf.add_n(slim.losses.get_regularization_losses())
                
                return cls_loss,bbox_loss,L2_loss,accuracy

            else:
                #测试时batch_size=1
                cls_pro_test=tf.squeeze(conv4_1,axis=0)
                bbox_pred_test=tf.squeeze(bbox_pred,axis=0)

                cls_pro_test = tf.nn.softmax(cls_pro_test)

                return cls_pro_test,bbox_pred_test,None



def R_Net_v1(inputs,label=None,bbox_target=None,landmark_target=None,training=True):
    '''
        在全志芯片上的版本。主要有下面的特点：
        1. conv层的激活函数从prelu改为relu；
        2. 将pooling去掉，把stride放到conv里面；
        3. 卷积核数量改为8的倍数；
        4. 把conv_4_1的softmax去掉，在cls_ohem中使用softmax；
        5. 把conv3的2*2卷积改为3*3卷积
    '''
    with tf.variable_scope('RNet'):
        with slim.arg_scope([slim.conv2d],
                           activation_fn=tf.nn.relu,
                           weights_initializer=slim.xavier_initializer(),
                           weights_regularizer=slim.l2_regularizer(0.0005),
                           padding='VALID'):
            net=slim.conv2d(inputs,32,3,stride=2,scope='conv1')
            net=slim.conv2d(net,48,3,stride=2,scope='conv2')
            net=slim.conv2d(net,64,3,scope='conv3')
            fc_flatten=slim.flatten(net)
            fc1=slim.fully_connected(fc_flatten,num_outputs=128,scope='fc1')
            
            cls_prob=slim.fully_connected(fc1,num_outputs=2,activation_fn=None,scope='cls_fc')
            bbox_pred=slim.fully_connected(fc1,num_outputs=4,activation_fn=None,scope='bbox_fc')
            # landmark_pred=slim.fully_connected(fc1,num_outputs=10,activation_fn=None,scope='landmark_fc')
            if training:
                cls_loss=cls_ohem(cls_prob,label,use_softmax=True)
                
                bbox_loss=bbox_ohem(bbox_pred,bbox_target,label)
                
                # landmark_loss=landmark_ohem(landmark_pred,landmark_target,label)
                
                accuracy=cal_accuracy(cls_prob,label)
                L2_loss=tf.add_n(slim.losses.get_regularization_losses())
                # return cls_loss,bbox_loss,landmark_loss,L2_loss,accuracy
                return cls_loss,bbox_loss,L2_loss,accuracy

            else:
                # return cls_prob,bbox_pred,landmark_pred
                cls_prob = tf.nn.softmax(cls_prob)

                return cls_prob,bbox_pred,None



# In[9]:


def O_Net_v1(inputs,label=None,bbox_target=None,landmark_target=None,training=True):
    '''
        在全志芯片上的版本。主要有下面的特点：
        1. conv层的激活函数从prelu改为relu；
        2. 将pooling去掉，把stride放到conv里面；
        3. 卷积核数量改为8的倍数；
        4. 把conv_4_1的softmax去掉，在cls_ohem中使用softmax；
        5. 把conv4的2*2卷积改为3*3卷积
    '''
    with tf.variable_scope('ONet'):
        with slim.arg_scope([slim.conv2d],
                           activation_fn=tf.nn.relu,
                           weights_initializer=slim.xavier_initializer(),
                           weights_regularizer=slim.l2_regularizer(0.0005),
                           padding='VALID'):
            net=slim.conv2d(inputs,32,3,stride=2,scope='conv1')
            # net=slim.max_pool2d(net,kernel_size=[3,3],stride=2,padding='SAME',scope='pool1')
            net=slim.conv2d(net,64,3,stride=2,scope='conv2')
            # net=slim.max_pool2d(net,kernel_size=[3,3],stride=2,scope='pool2')
            net=slim.conv2d(net,64,3,stride=2,scope='conv3')
            # net=slim.max_pool2d(net,kernel_size=[2,2],stride=2,padding='SAME',scope='pool3')
            net=slim.conv2d(net,128,3,scope='conv4')
            fc_flatten=slim.flatten(net)
            fc1=slim.fully_connected(fc_flatten,num_outputs=256,scope='fc1')
            
            cls_prob=slim.fully_connected(fc1,num_outputs=2,activation_fn=None,scope='cls_fc')
            bbox_pred=slim.fully_connected(fc1,num_outputs=4,activation_fn=None,scope='bbox_fc')
            # landmark_pred=slim.fully_connected(fc1,num_outputs=10,activation_fn=None,scope='landmark_fc')
            if training:
                cls_loss=cls_ohem(cls_prob,label,use_softmax=True)
                
                bbox_loss=bbox_ohem(bbox_pred,bbox_target,label)
                
                # landmark_loss=landmark_ohem(landmark_pred,landmark_target,label)
                
                accuracy=cal_accuracy(cls_prob,label)
                L2_loss=tf.add_n(slim.losses.get_regularization_losses())
                # return cls_loss,bbox_loss,landmark_loss,L2_loss,accuracy
                return cls_loss,bbox_loss,L2_loss,accuracy
            else:
                # return cls_prob,bbox_pred,landmark_pred
                cls_prob = tf.nn.softmax(cls_prob)

                return cls_prob,bbox_pred,None



def R_Net_fcn(inputs,label=None,bbox_target=None,landmark_target=None,training=True):
    '''
        @新栋师兄在群里发的版本；
        将原始的RNet改为全卷积的形式：
        1. 通道数改为8的倍数；
        2. 去掉max pooling的padding='SAME'操作；
        3. 将全连接层改为1*1卷积层；
    '''
    with tf.variable_scope('RNet'):
        with slim.arg_scope([slim.conv2d],
                           activation_fn=prelu,
                           weights_initializer=slim.xavier_initializer(),
                           weights_regularizer=slim.l2_regularizer(0.0005),
                           padding='VALID'):
            net=slim.conv2d(inputs,24,3,scope='conv1')
            net=slim.max_pool2d(net,kernel_size=[3,3],stride=2,scope='pool1')
            net=slim.conv2d(net,32,3,scope='conv2')
            net=slim.max_pool2d(net,kernel_size=[3,3],stride=2,scope='pool2')
            net=slim.conv2d(net,48,3,scope='conv3')
            net=slim.conv2d(net,128,1,scope='conv4')

            cls_prob = slim.conv2d(net, 2, 1, activation_fn=tf.nn.softmax,scope='cls_conv')
            bbox_pred = slim.conv2d(net, 4, 1, scope='bbox_conv')

            if training:
                cls_prob = tf.squeeze(cls_prob, [1, 2], name='cls_prob')
                bbox_pred = tf.squeeze(bbox_pred, [1, 2], name='bbox_pred')

                cls_loss=cls_ohem(cls_prob,label)
                bbox_loss=bbox_ohem(bbox_pred,bbox_target,label)
                accuracy=cal_accuracy(cls_prob,label)
                L2_loss=tf.add_n(slim.losses.get_regularization_losses())
                # return cls_loss,bbox_loss,landmark_loss,L2_loss,accuracy
                return cls_loss,bbox_loss,L2_loss,accuracy

            else:
                cls_prob = tf.squeeze(cls_prob, [1, 2], name='cls_prob')
                bbox_pred = tf.squeeze(bbox_pred, [1, 2], name='bbox_pred')

                cls_prob = tf.nn.softmax(cls_prob)

                # return cls_prob,bbox_pred,landmark_pred
                return cls_prob,bbox_pred,None


def R_Net_fcn_v1(inputs,label=None,bbox_target=None,landmark_target=None,training=True):
    '''
        将原始的RNet改为全卷积的形式：
        1. 通道数改为8的倍数；
        2. 去掉max pooling的padding='SAME'操作；
        3. 将全连接层改为1*1卷积层；
        4. conv层的激活函数从prelu改为relu；
        5. 把cls_conv的softmax去掉，在cls_ohem中使用softmax；
    '''
    with tf.variable_scope('RNet'):
        with slim.arg_scope([slim.conv2d],
                           activation_fn=tf.nn.relu,
                           weights_initializer=slim.xavier_initializer(),
                           weights_regularizer=slim.l2_regularizer(0.0005),
                           padding='VALID'):
            net=slim.conv2d(inputs,24,3,scope='conv1')
            net=slim.max_pool2d(net,kernel_size=[3,3],stride=2,scope='pool1')
            net=slim.conv2d(net,32,3,scope='conv2')
            net=slim.max_pool2d(net,kernel_size=[3,3],stride=2,scope='pool2')
            net=slim.conv2d(net,48,3,scope='conv3')
            net=slim.conv2d(net,128,1,scope='conv4')

            cls_prob = slim.conv2d(net, 2, 1, activation_fn=None,scope='cls_conv')
            bbox_pred = slim.conv2d(net, 4, 1, scope='bbox_conv')

            if training:
                cls_prob = tf.squeeze(cls_prob, [1, 2], name='cls_prob')
                bbox_pred = tf.squeeze(bbox_pred, [1, 2], name='bbox_pred')

                cls_loss=cls_ohem(cls_prob,label, use_softmax=True)
                bbox_loss=bbox_ohem(bbox_pred,bbox_target,label)
                accuracy=cal_accuracy(cls_prob,label)
                L2_loss=tf.add_n(slim.losses.get_regularization_losses())
                # return cls_loss,bbox_loss,landmark_loss,L2_loss,accuracy
                return cls_loss,bbox_loss,L2_loss,accuracy

            else:
                cls_prob = tf.squeeze(cls_prob, [1, 2], name='cls_prob')
                bbox_pred = tf.squeeze(bbox_pred, [1, 2], name='bbox_pred')
                cls_prob = tf.nn.softmax(cls_prob)

                # return cls_prob,bbox_pred,landmark_pred
                return cls_prob,bbox_pred,None



def O_Net_fcn_v1(inputs,label=None,bbox_target=None,landmark_target=None,training=True):
    '''
        将原始的ONet改为全卷积的形式：
        1. 通道数改为8的倍数；
        2. 去掉max pooling的padding='SAME'操作；
        3. 将全连接层改为1*1卷积层；
        4. conv层的激活函数从prelu改为relu；
        5. 把cls_conv的softmax去掉，在cls_ohem中使用softmax；
    '''
    with tf.variable_scope('ONet'):
        with slim.arg_scope([slim.conv2d],
                           activation_fn=tf.nn.relu,
                           weights_initializer=slim.xavier_initializer(),
                           weights_regularizer=slim.l2_regularizer(0.0005),
                           padding='VALID'):
            net=slim.conv2d(inputs,32,3,scope='conv1')
            net=slim.max_pool2d(net,kernel_size=[3,3],stride=2,scope='pool1')
            net=slim.conv2d(net,64,3,scope='conv2')
            net=slim.max_pool2d(net,kernel_size=[3,3],stride=2,scope='pool2')
            net=slim.conv2d(net,64,3,scope='conv3')
            net=slim.max_pool2d(net,kernel_size=[3,3],stride=2,scope='pool3')
            net=slim.conv2d(net,128,3,scope='conv4')
            # fc_flatten=slim.flatten(net)
            # fc1=slim.fully_connected(fc_flatten,num_outputs=256,scope='fc1')
            net=slim.conv2d(net,256,1,scope='conv5')

            cls_prob = slim.conv2d(net, 2, 1, activation_fn=None,scope='cls_conv')
            bbox_pred = slim.conv2d(net, 4, 1, scope='bbox_conv')

            # landmark_pred=slim.fully_connected(fc1,num_outputs=10,activation_fn=None,scope='landmark_fc')
            if training:
                cls_prob = tf.squeeze(cls_prob, [1, 2], name='cls_prob')
                bbox_pred = tf.squeeze(bbox_pred, [1, 2], name='bbox_pred')

                cls_loss=cls_ohem(cls_prob,label,use_softmax=True)
                
                bbox_loss=bbox_ohem(bbox_pred,bbox_target,label)
                
                # landmark_loss=landmark_ohem(landmark_pred,landmark_target,label)
                
                accuracy=cal_accuracy(cls_prob,label)
                L2_loss=tf.add_n(slim.losses.get_regularization_losses())
                # return cls_loss,bbox_loss,landmark_loss,L2_loss,accuracy
                return cls_loss,bbox_loss,L2_loss,accuracy
            else:
                # return cls_prob,bbox_pred,landmark_pred
                cls_prob = tf.nn.softmax(cls_prob)
                
                return cls_prob,bbox_pred,None


def P_Net_aspect_24_12(inputs,label=None,bbox_target=None,landmark_target=None,training=True):
    '''
        感受野为24*12的PNet：
        1. conv2的卷积核改为[5,3]
        2. conv3的卷积核改为[5,3]
        3. 增加一个conv4，卷积核大小为[3,1]

    '''
    with tf.variable_scope('PNet'):
        with slim.arg_scope([slim.conv2d],activation_fn=prelu,
                           weights_initializer=slim.xavier_initializer(),
                           weights_regularizer=slim.l2_regularizer(0.0005),
                           padding='VALID'):
            print(inputs.shape)
            net=slim.conv2d(inputs,10,3,scope='conv1')
            net=slim.max_pool2d(net,kernel_size=[2,2],stride=2,padding='SAME',scope='pool1')
            net=slim.conv2d(net,16,[5,3],scope='conv2')
            net=slim.conv2d(net,32,[5,3],scope='conv3')
            net=slim.conv2d(net,32,[3,1],scope='conv4')

            #二分类输出通道数为2
            conv5_1=slim.conv2d(net,2,1,activation_fn=tf.nn.softmax,scope='conv5_1')
            bbox_pred=slim.conv2d(net,4,1,activation_fn=None,scope='conv5_2')
            
            if training:
                cls_prob=tf.squeeze(conv5_1,[1,2],name='cls_prob')#[batch,2]
                cls_loss=cls_ohem(cls_prob,label)
                
                bbox_pred=tf.squeeze(bbox_pred,[1,2],name='bbox_pred')#[bacth,4]
                bbox_loss=bbox_ohem(bbox_pred,bbox_target,label)
                
                accuracy=cal_accuracy(cls_prob,label)
                L2_loss=tf.add_n(slim.losses.get_regularization_losses())
                return cls_loss,bbox_loss,L2_loss,accuracy

            else:
                #测试时batch_size=1
                cls_pro_test=tf.squeeze(conv5_1,axis=0)
                bbox_pred_test=tf.squeeze(bbox_pred,axis=0)
                return cls_pro_test,bbox_pred_test,None



def prelu(inputs):
    '''prelu函数定义'''
    alphas=tf.get_variable('alphas',shape=inputs.get_shape()[-1],dtype=tf.float32,
                          initializer=tf.constant_initializer(0.25))
    pos=tf.nn.relu(inputs)
    neg=alphas*(inputs-abs(inputs))*0.5
    return pos+neg


# In[3]:


def cls_ohem(cls_prob,label,use_softmax=False):
    '''计算类别损失
    参数：
      cls_prob：预测类别，是否有人
      label：真实值
    返回值：
      损失
    '''
    if use_softmax:
        cls_prob = tf.nn.softmax(cls_prob)

    zeros=tf.zeros_like(label)
    #只把pos的label设定为1,其余都为0
    label_filter_invalid=tf.where(tf.less(label,0),zeros,label)
    #类别size[2*batch]
    num_cls_prob=tf.size(cls_prob)
    cls_prob_reshpae=tf.reshape(cls_prob,[num_cls_prob,-1])
    label_int=tf.cast(label_filter_invalid,tf.int32)
    #获取batch数
    num_row=tf.to_int32(cls_prob.get_shape()[0])
    #对应某一batch而言，batch*2为非人类别概率，batch*2+1为人概率类别,indices为对应 cls_prob_reshpae
    #应该的真实值，后续用交叉熵计算损失
    row=tf.range(num_row)*2
    indices_=row+label_int
    #真实标签对应的概率
    label_prob=tf.squeeze(tf.gather(cls_prob_reshpae,indices_))
    loss=-tf.log(label_prob+1e-10)
    zeros=tf.zeros_like(label_prob,dtype=tf.float32)
    ones=tf.ones_like(label_prob,dtype=tf.float32)
    #统计neg和pos的数量
    valid_inds=tf.where(label<zeros,zeros,ones)
    num_valid=tf.reduce_sum(valid_inds)
    #选取70%的数据
    keep_num=tf.cast(num_valid*num_keep_radio,dtype=tf.int32)
    #只选取neg，pos的70%损失
    loss=loss*valid_inds
    loss,_=tf.nn.top_k(loss,k=keep_num)
    return tf.reduce_mean(loss)


def cls_ohem_2cls(cls_prob,label):
    '''计算类别损失
    参数：
      cls_prob：预测类别，背景、人头、上半身
      label：真实值，背景-0、人头-1、上半身-2
    返回值：
      损失
    '''
    zeros=tf.zeros_like(label)
    #只把pos的label设定为1或2，其余为0
    label_filter_invalid=tf.where(tf.less(label,0),zeros,label)
    #类别size[3*batch]
    num_cls_prob=tf.size(cls_prob)
    cls_prob_reshpae=tf.reshape(cls_prob,[num_cls_prob,-1])
    label_int=tf.cast(label_filter_invalid,tf.int32)
    #获取batch数
    num_row=tf.to_int32(cls_prob.get_shape()[0])
    #对应某一batch而言，batch*3为非人类别概率，batch*2+1为人头概率类别,batch*2+2为上半身概率类别,indices为对应 cls_prob_reshpae
    #应该的真实值，后续用交叉熵计算损失
    row=tf.range(num_row)*3
    indices_=row+label_int
    #真实标签对应的概率
    label_prob=tf.squeeze(tf.gather(cls_prob_reshpae,indices_))
    loss=-tf.log(label_prob+1e-10)
    zeros=tf.zeros_like(label_prob,dtype=tf.float32)
    ones=tf.ones_like(label_prob,dtype=tf.float32)
    #统计neg和pos的数量
    valid_inds=tf.where(label<zeros,zeros,ones)
    num_valid=tf.reduce_sum(valid_inds)
    #选取70%的数据
    keep_num=tf.cast(num_valid*num_keep_radio,dtype=tf.int32)
    #只选取neg，pos的70%损失
    loss=loss*valid_inds
    loss,_=tf.nn.top_k(loss,k=keep_num)
    return tf.reduce_mean(loss)


# In[4]:


def bbox_ohem(bbox_pred,bbox_target,label):
    '''计算box的损失'''
    zeros_index=tf.zeros_like(label,dtype=tf.float32)
    ones_index=tf.ones_like(label,dtype=tf.float32)
    #保留pos和part的数据
    valid_inds=tf.where(tf.equal(tf.abs(label),1),ones_index,zeros_index)
    #计算平方差损失
    square_error=tf.square(bbox_pred-bbox_target)
    square_error=tf.reduce_sum(square_error,axis=1)
    #保留的数据的个数
    num_valid=tf.reduce_sum(valid_inds)
    keep_num=tf.cast(num_valid,dtype=tf.int32)
    #保留pos和part部分的损失
    square_error=square_error*valid_inds
    square_error,_=tf.nn.top_k(square_error,k=keep_num)
    return tf.reduce_mean(square_error)
    

def bbox_ohem_2cls(bbox_pred,bbox_target,label):
    '''计算box的损失'''

    zeros_index=tf.zeros_like(label,dtype=tf.float32)
    ones_index=tf.ones_like(label,dtype=tf.float32)


    bbox_pred_reshpae=tf.reshape(bbox_pred,[-1,4])

    # label绝对值小于2的话，置0，否则置1
    label_filter_invalid = tf.where(tf.less(tf.abs(label), 2), zeros_index, ones_index)

    label_int=tf.cast(label_filter_invalid,tf.int32)
    #获取batch数
    num_row=tf.to_int32(bbox_pred.get_shape()[0])
    row=tf.range(num_row)*2
    indices_=row+label_int
    # 对应的回归目标，背景和人头为前四个神经元，上半身为后四个神经元，背景在下面会过滤掉
    bbox_pred=tf.gather(bbox_pred_reshpae, indices_, axis=0)


    #保留pos和part的数据
    valid_inds_1=tf.where(tf.equal(label,-2),ones_index,zeros_index)    # 这里-2表示上半身part
    valid_inds_2=tf.where(tf.equal(label,-1),ones_index,zeros_index)    # -1表示人头part
    valid_inds_3=tf.where(tf.equal(label,1),ones_index,zeros_index)
    valid_inds_4=tf.where(tf.equal(label,2),ones_index,zeros_index)

    valid_inds = tf.add_n([valid_inds_1, valid_inds_2, valid_inds_3, valid_inds_4])

    #计算平方差损失
    square_error=tf.square(bbox_pred-bbox_target)
    square_error=tf.reduce_sum(square_error,axis=1)
    #保留的数据的个数
    num_valid=tf.reduce_sum(valid_inds)
    keep_num=tf.cast(num_valid,dtype=tf.int32)
    #保留pos和part部分的损失
    square_error=square_error*valid_inds
    square_error,_=tf.nn.top_k(square_error,k=keep_num)
    return tf.reduce_mean(square_error)
    


# In[5]:


def landmark_ohem(landmark_pred,landmark_target,label):
    '''计算关键点损失'''
    ones=tf.ones_like(label,dtype=tf.float32)
    zeros=tf.zeros_like(label,dtype=tf.float32)
    #只保留landmark数据
    valid_inds=tf.where(tf.equal(label,-2),ones,zeros)
    #计算平方差损失
    square_error=tf.square(landmark_pred-landmark_target)
    square_error=tf.reduce_sum(square_error,axis=1)
    #保留数据个数
    num_valid=tf.reduce_sum(valid_inds)
    keep_num=tf.cast(num_valid,dtype=tf.int32)
    #保留landmark部分数据损失
    square_error=square_error*valid_inds
    square_error,_=tf.nn.top_k(square_error,k=keep_num)
    return tf.reduce_mean(square_error)


# In[6]:


def cal_accuracy(cls_prob,label):
    '''计算分类准确率'''
    #预测最大概率的类别，0代表无人，1代表有人
    pred=tf.argmax(cls_prob,axis=1)
    label_int=tf.cast(label,tf.int64)
    #保留label>=0的数据，即pos和neg的数据
    cond=tf.where(tf.greater_equal(label_int,0))
    picked=tf.squeeze(cond)
    #获取pos和neg的label值
    label_picked=tf.gather(label_int,picked)
    pred_picked=tf.gather(pred,picked)
    #计算准确率
    accuracy_op=tf.reduce_mean(tf.cast(tf.equal(label_picked,pred_picked),tf.float32))
    return accuracy_op


def cal_accuracy_2cls(cls_prob,label):
    '''计算分类准确率'''
    #预测最大概率的类别，0代表无人，1代表有人头，2代表上半身
    pred=tf.argmax(cls_prob,axis=1)
    label_int=tf.cast(label,tf.int64)
    #保留label>=0的数据，即pos和neg的数据
    cond=tf.where(tf.greater_equal(label_int,0))
    picked=tf.squeeze(cond)
    #获取pos和neg的label值
    label_picked=tf.gather(label_int,picked)
    pred_picked=tf.gather(pred,picked)
    #计算准确率
    accuracy_op=tf.reduce_mean(tf.cast(tf.equal(label_picked,pred_picked),tf.float32))
    return accuracy_op





