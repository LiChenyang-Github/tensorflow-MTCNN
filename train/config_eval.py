# coding: utf-8


#最小脸大小设定
##################
min_face=20
# min_face=12
# min_face=8
# min_face=9





#生成hard_example的batch
batches=[2048,256,16]
#pent对图像缩小倍数
##################
stride=2
# stride=1

#三个网络的阈值
thresh=[0.6,0.7,0.1]
#最后测试选择的网络
test_mode='ONet'


#模型的后缀名称
##################
# suffix=None
suffix='all'








#测试图片的根目录
##################
# img_root='/disk3/hjy/work/data/aic_upper_head/'
# img_root='/disk2/lichenyang/lichenyang/Datasets/fenglian/'
img_root='/disk2/lichenyang/lichenyang/Projects/tensorflow-MTCNN_quanzhi_8P100/output_mine/tmp/xiao/test_imgs/'

#测试图片标签放置位置
##################
test_dir = None
# test_dir='data/xiao_head_val.txt'
# test_dir='data/xiao_upperbody_val.txt'
# test_dir='data/fenglian/fenglian_head.txt'
# test_dir='data/fenglian/fenglian_upperbody.txt'
# test_dir='data/fenglian/fenglian_head_val.txt'
# test_dir='data/fenglian/fenglian_upperbody_val.txt'





#保存结果的txt的位置
##################
# out_path='output_mine/'
# out_path='output_mine/minface-12/fix-bandwidth/'
# out_path='output_mine/wo-delsize/upperbody/minface-12/fix-bandwidth/'
# out_path='output_mine/wo-delsize/upperbody/'
# out_path='output_mine/wo-delsize/head/'
# out_path='output_mine/wo-delsize/head/minface-12/fix-bandwidth/'
# out_path='output_mine/fenglian/head/'
# out_path='output_mine/fenglian/upperbody/minface-12/fix-bandwidth/'
# out_path='output_mine/fenglian/upperbody/'
# out_path='output_mine/wo-delsize/head/minface-9/fix-bandwidth/'
# out_path='output_mine/fenglian/'
out_path='output_mine/tmp/xiao/'
# out_path='output_mine/wo-delsize/upperbody/sliding_win_300-300/'
# out_path='output_mine/fenglian/sliding_win_300-300/'
# out_path='output_mine/fenglian/minface-12/'



#是否可视化检测结果，并保存为图片形式
##################
save_img=False

#是否可视化检测结果，并保存为tensorboard
##################
save_tensorboard=False


#目标的名称
##################
obj_name='head'
# obj_name='upperbody'


#是否将输入进行reisze到符合芯片带宽
##################
resize_input=False
# resize_input=True


#是否对超过芯片带宽的图片进行滑窗
##################
sliding_win=False
win_size=(300,300)


##################
# model_path=None
# model_path=['model/head_all/PNet_9/','model/all/RNet/','model/all/ONet/']

net_name=None
# net_name=[]
