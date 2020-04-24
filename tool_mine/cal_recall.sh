
GPU_IDS=5
INPUT_SIZE=24

### xiao 6 scenes head/upperbody all
# CUDA_VISIBLE_DEVICES=${GPU_IDS} python cal_recall.py ${INPUT_SIZE} \
#     --suffix xiao-6-scenes_upperbody_all \
#     --filename ../data_mine/xiao_6_scenes/xiao_upperbody_val.txt \
#     --base_dir /disk3/hjy/work/data/aic_upper_head/ \
#     --net_name P_Net_v1 R_Net_fcn_v1

    # --img_num 5000




    # --net_name P_Net_v1 R_Net_v1
