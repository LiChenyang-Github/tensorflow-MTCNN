GPU_IDS=0
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

### wider face
# CUDA_VISIBLE_DEVICES=${GPU_IDS} python train_mine.py ${INPUT_SIZE} \
#     --suffix wider_face_mine

### upperBody_annotation_4_82_upperbody_all
# CUDA_VISIBLE_DEVICES=${GPU_IDS} python train_mine.py ${INPUT_SIZE} \
#     --suffix upperBody_annotation_4_82_upperbody_all

### upperBody_annotation_4_82_upperbody_all aspect-24-12, 125wx6, lrx100 (the lr in ICC-CNN),
## 训练RNet和ONet的时候不需要设置--aspect参数(只有PNet需要)，否则输入的placehold的形状会设置为aspect的尺寸
CUDA_VISIBLE_DEVICES=${GPU_IDS} python train_mine.py ${INPUT_SIZE} \
    --suffix upperBody_annotation_4_82-125wx6-lrx100_upperbody_all_aspect-24-12 \
    --net_name P_Net_aspect_24_12 \
    --aspect 24 12

### upperBody_annotation_4_82_upperbody_all aspect-18-12
# CUDA_VISIBLE_DEVICES=${GPU_IDS} python train_mine.py ${INPUT_SIZE} \
#     --suffix upperBody_annotation_4_82_upperbody_all_aspect-18-12 \
#     --net_name P_Net_aspect_18_12 \
#     --aspect 18 12
