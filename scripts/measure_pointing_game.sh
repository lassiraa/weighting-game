# Directory of validation images and annotations file path
images_dir="CHANGE"  # e.g. ./coco/images/val2017/
ann_path="CHANGE"  # e.g. ./coco/annotations/instances_val2017.json

# Desired model/method combination to process
method="gradcam" # options: gradcam, gradcam++, xgradcam, ablationcam, layercam, guidedbackprop
model="resnet50" # options: resnet50, swin_t, vit_b_32, vgg16_bn

# Batch size for batched CAM methods
batch_size=32

python pointing_game.py \
    --method ${method} \
    --model ${model} \
    --images_dir ${images_dir} \
    --ann_path ${ann_path} \
    --batch_size ${batch_size}
