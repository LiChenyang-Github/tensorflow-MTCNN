
GPU_IDS=1
INPUT_SIZE=24

### xiao 6 scenes head/upperbody all
# CUDA_VISIBLE_DEVICES=${GPU_IDS} python gen_hard_example.py ${INPUT_SIZE} \
#     --suffix xiao-6-scenes_head_all \
#     --filename ../data_mine/xiao_6_scenes/xiao_head_train.txt \
#     --base_dir /disk3/hjy/work/data/aic_upper_head/ \
#     --net_name P_Net_v1 R_Net_v1


### wider face
# CUDA_VISIBLE_DEVICES=${GPU_IDS} python gen_hard_example.py ${INPUT_SIZE} \
#     --suffix wider_face_mine \
#     --filename ../data/wider_face_train.txt \
#     --base_dir ../data/WIDER_train/images/ \
#     --add_img_suffix


### upperbody_4
# CUDA_VISIBLE_DEVICES=${GPU_IDS} python gen_hard_example.py ${INPUT_SIZE} \
#     --suffix upperBody_annotation_4_82_upperbody_all \
#     --filename ../data_mine/upperBody_annotation_4_82/upperBody_annotation_4_82_upperbody_train.txt \
#     --base_dir /home/LiChenyang/Datasets/xi_ao/
