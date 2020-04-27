# coding: utf-8


#测试图片的根目录
##################
img_root='/home/LiChenyang/Projects/tensorflow-MTCNN/picture/'

#测试图片标签放置位置，如果test_dir为None，就直接对img_root下面的图片进行forward
##################
test_dir = None
# test_dir='data/xiao_head_val.txt'

#保存结果的路径
##################
out_path='../output_mine/wider_face_mine_test'

#图片的数量
##################
img_num=None

#模型的后缀名称
##################
# suffix=None
suffix='wider_face_mine'

##################
model_path=None
# model_path=['../model/PNet/','../model/RNet/','../model/ONet/']

##################
net_name=None
# net_name=[]






#最小脸大小设定
##################
min_face=20

#生成hard_example的batch
batches=[2048,256,16]
#pent对图像缩小倍数
##################
stride=2

#三个网络的阈值
thresh=[0.6,0.7,0.7]
#最后测试选择的网络
test_mode='ONet'


#是否将输入进行reisze到符合芯片带宽
##################
resize_input=False
# resize_input=True

#是否对超过芯片带宽的图片进行滑窗
##################
sliding_win=False
win_size=(300,300)