
GPU_IDS=7
INPUT_SIZE=24

### xiao 6 scenes head/upperbody all
# CUDA_VISIBLE_DEVICES=${GPU_IDS} python gen_hard_example.py ${INPUT_SIZE} \
#     --suffix xiao-6-scenes_head_all \
#     --filename ../data_mine/xiao_6_scenes/xiao_head_train.txt \
#     --base_dir /disk3/hjy/work/data/aic_upper_head/ \
#     --net_name P_Net_v1 R_Net_v1
