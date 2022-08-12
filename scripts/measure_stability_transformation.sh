# Directory of validation images and annotations file path
images_dir="CHANGE"  # e.g. ./coco/images/val2017/
ann_path="CHANGE"  # e.g. ./coco/annotations/instances_val2017.json

# Desired model/method combination to process
method="gradcam" # options: gradcam, gradcam++, xgradcam, ablationcam, layercam, guidedbackprop
model="resnet50" # options: resnet50, swin_t, vit_b_32, vgg16_bn

batch_size=32

python explanation_stability_transformation.py \
    --method ${method} \
    --model ${model} \
    --images_dir ${images_dir} \
    --ann_path ${ann_path} \
    --batch_size ${batch_size}
