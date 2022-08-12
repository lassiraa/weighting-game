# Directory of videos containing 3D effect (150 frames each)
video_dir="CHANGE"

# Desired model/method combination to process
method="gradcam" # options: gradcam, gradcam++, xgradcam, ablationcam, layercam, guidedbackprop
model="resnet50" # options: resnet50, swin_t, vit_b_32, vgg16_bn

#  Batch size for batched explainability methods
batch_size=32

python explanation_stability_video.py \
    --method ${method} \
    --model ${model} \
    --in_path ${video_dir} \
    --batch_size ${batch_size}
