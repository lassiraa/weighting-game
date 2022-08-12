# Training parameters for replication
model_name="resnet50"  # options: resnet50, swin_t, vit_b_32, vgg16_bn
lr="1e-3"
gamma=0.95
batch_size=64
epochs=50

# Desired paths and num workers
images_dir="CHANGE"  # e.g. ./coco/images/
ann_dir="CHANGE"  # e.g. ./coco/annotations/
checkpoint_dir="CHANGE"
num_workers=16

python fine_tune.py --images_dir ${images_dir} \
    --ann_dir ${ann_dir} \
    --checkpoint_dir ${checkpoint_dir} \
    --model_name ${model_name} \
    --lr ${lr} \
    --gamma ${gamma} \
    --epochs ${epochs} \
    --batch_size ${batch_size} \
    --num_workers ${num_workers} \
