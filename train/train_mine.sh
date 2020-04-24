GPU_IDS=4
INPUT_SIZE=12

### xiao 6 scenes head all
# CUDA_VISIBLE_DEVICES=${GPU_IDS} python train_mine.py ${INPUT_SIZE} \
#     --suffix xiao-6-scenes_head_all \
    # --net_name O_Net_fcn_v1

### xiao 6 scenes upperbody all
# CUDA_VISIBLE_DEVICES=${GPU_IDS} python train_mine.py ${INPUT_SIZE} \
#     --suffix xiao-6-scenes_upperbody_all \
#     --net_name O_Net_fcn_v1

### fenglian-1-2 + xiao 3 scenes head/upperbody all
# CUDA_VISIBLE_DEVICES=${GPU_IDS} python train_mine.py ${INPUT_SIZE} \
#     --use_multi_tfrecords True \
#     --net_name P_Net_v1