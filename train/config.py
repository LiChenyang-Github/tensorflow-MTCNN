
# coding: utf-8

# In[ ]:

#是否导入预训练模型的参数
##################################
pretrained=None
# pretrained="../model_mine/xiao-6-scenes_head_all/PNet/"
# pretrained="../model_mine/xiao-6-scenes_head_all/P_Net_v1/"


# pretrained="../model_mine/xiao-6-scenes_upperbody_all/PNet/"
# pretrained="../model_mine/xiao-6-scenes_upperbody_all/P_Net_v1/"




#是否不导入某些参数
##################################
exclude_vars = None
# exclude_vars = ['RNet/conv3', 'RNet/fc1']
# exclude_vars = ['ONet/conv4', 'ONet/fc1']

#是否导入chpt继续训练
##################################
resume=None
# resume="../model/head_all-fenglian-1-2_head-holly_head/R_Net_v1/"
# resume="../model/head_all-fenglian-1-2_head-holly_head/R_Net_fcn/"



#迭代次数
end_epoch=[30,22,22]
#经过多少batch显示数据
display=100
#初始学习率
lr=0.001

batch_size=384
###
# batch_size=2048

#学习率减少的迭代次数
LR_EPOCH=[6,14,20]
#最小脸大小设定
min_face=20
# min_face=12


#生成hard_example的batch
batches=[2048,256,16]
#pent对图像缩小倍数
stride=2
#三个网络的阈值
thresh=[0.6,0.7,0.7]

###gen_hard_example.py使用的thred，PNet和RNet使用较低的阈值，以产生更多的proposals，增大recall
thresh_hard_examples=[0.3, 0.1, 0.7]


#当使用train_multi_tfrecords的时候，具体使用哪些后缀的tfrecords
##################################
# suffix_list = None
# suffix_list = ["fenglian-1-2_head_all", "xiao_gym_subway_head_all"]
suffix_list = ["fenglian-1-2_upperbody_all", "xiao_gym_subway_upperbody_all"]


##################################
#构成每个batch的各个suffix的tfrecords的比例
# batch_ratio = [2/6, 4/6]
batch_ratio = [3/6, 3/6]

