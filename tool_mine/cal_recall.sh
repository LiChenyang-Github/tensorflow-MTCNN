
GPU_IDS=2
INPUT_SIZE=12

### xiao 6 scenes head/upperbody all
# CUDA_VISIBLE_DEVICES=${GPU_IDS} python cal_recall.py ${INPUT_SIZE} \
#     --suffix xiao-6-scenes_upperbody_all \
#     --filename ../data_mine/xiao_6_scenes/xiao_upperbody_val.txt \
#     --base_dir /disk3/hjy/work/data/aic_upper_head/ \
#     --net_name P_Net_v1 R_Net_fcn_v1

    # --img_num 5000

    # --net_name P_Net_v1 R_Net_v1


### wider_face_v1 model (https://github.com/LeslieZhoa/tensorflow-MTCNN)
# CUDA_VISIBLE_DEVICES=${GPU_IDS} python cal_recall.py ${INPUT_SIZE} \
#     --suffix wider_face_v1 \
#     --filename ../data/wider_face_train.txt \
#     --base_dir ../data/WIDER_train/images/ \
#     --net_name P_Net_org R_Net_org\
#     --img_num 5000 \
#     --add_img_suffix


### wider_face_mine 自己基于wider_face训练的MTCNN
# CUDA_VISIBLE_DEVICES=${GPU_IDS} python cal_recall.py ${INPUT_SIZE} \
#     --suffix wider_face_mine \
#     --filename ../data/wider_face_train.txt \
#     --base_dir ../data/WIDER_train/images/ \
#     --img_num 5000 \
#     --add_img_suffix


### wider_face_v1 和 wider_face_mine模型在FDDB上的recall计算 
# CUDA_VISIBLE_DEVICES=${GPU_IDS} python cal_recall.py ${INPUT_SIZE} \
#     --suffix fddb \
#     --filename ../data_mine/FDDB/FDDB.txt \
#     --base_dir /home/LiChenyang/Datasets/FDDB/images/ 
    # --net_name P_Net_org


### uppperbody_4
# CUDA_VISIBLE_DEVICES=${GPU_IDS} python cal_recall.py ${INPUT_SIZE} \
#     --suffix upperBody_annotation_4_82_upperbody_all \
#     --filename ../data_mine/upperBody_annotation_4_82/upperBody_annotation_4_82_upperbody_train.txt \
#     --base_dir /home/LiChenyang/Datasets/xi_ao/ \
#     --img_num 5000 