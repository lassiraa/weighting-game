from typing import Callable
import os
import json

import moviepy.editor as mpy 
import torch
import torchvision.transforms as transforms
import numpy as np
from scipy import stats
from tqdm import tqdm
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    LayerCAM, \
    FullGrad, \
    GuidedBackpropReLUModel
from pytorch_grad_cam.ablation_layer import AblationLayerVit
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from utils import reshape_transform_vit, load_model_with_target_layers, scale_image


def calculate_mean_correlation(
    video: mpy.VideoFileClip,
    device: torch.device,
    saliency_method: Callable,
    class_idx: int,
    image_transform: transforms.Compose,
    image_normalize: transforms.Normalize,
    is_backprop: bool
) -> tuple[np.ndarray]:
    correlations = []
    prev_saliency = None

    for i, frame in enumerate(video.iter_frames()):
        
        #  Process 5 pairs of frames per video, so 10 frames in total
        if ((i+2) % 29 != 0) and ((i + 1) % 29 != 0):
            prev_saliency = None
            continue

        #  Preprocess frame for model
        frame = image_transform(frame / 255)
        input = image_normalize(frame).to(device=device, dtype=torch.float32).unsqueeze(0)

        #  Process saliency map
        if is_backprop:
            saliency_map = saliency_method(input, target_category=class_idx)
            saliency_map = saliency_map.sum(axis=2).reshape(224, 224)
            saliency_map = np.where(saliency_map > 0, saliency_map, 0)
            saliency_map = scale_image(saliency_map, 1)
        else:
            saliency_map = saliency_method(input, [ClassifierOutputTarget(class_idx)])[0, :]

        #  Calculate Spearman correlation of saliency map from previous frame to current frame
        if prev_saliency is not None:
            corr, _ = stats.spearmanr(prev_saliency, saliency_map, axis=None)
            correlations.append(corr)

        prev_saliency = np.copy(saliency_map)
    
    return np.mean(correlations)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Measure stability of explanation method')
    parser.add_argument('--in_path', type=str, required=True,
                        help='path folder containing videos')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size for cam methods')
    parser.add_argument('--model_name', type=str, default='resnet50',
                        help='name of model used for inference',
                        choices=['vit_b_32', 'swin_t', 'vgg16_bn', 'resnet50'])
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=['gradcam', 'gradcam++',
                                 'scorecam', 'xgradcam',
                                 'ablationcam', 'eigencam',
                                 'eigengradcam', 'layercam', 'fullgrad',
                                 'guidedbackprop'],
                        help='Can be gradcam/gradcam++/scorecam/xgradcam'
                             '/ablationcam/eigencam/eigengradcam/layercam')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model, target_layers = load_model_with_target_layers(args.model_name, device)

    reshape_transform = None
    is_vit = args.model_name in ['vit_b_32', 'swin_t']

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if is_vit:
        reshape_transform = reshape_transform_vit

    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(224),
        transforms.CenterCrop(224)
    ])

    image_normalize = transforms.Normalize(mean=mean, std=std)

    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "fullgrad": FullGrad,
         "layercam": LayerCAM,
         "guidedbackprop": GuidedBackpropReLUModel}

    method = methods[args.method]
    is_backprop = False
    if args.method == 'guidedbackprop':
        saliency_method = method(model=model,
                                 use_cuda=torch.cuda.is_available())
        is_backprop = True
    elif args.method == 'ablationcam' and is_vit:
        saliency_method = method(model=model,
                                 target_layers=target_layers,
                                 reshape_transform=reshape_transform,
                                 use_cuda=torch.cuda.is_available(),
                                 ablation_layer=AblationLayerVit())
        saliency_method.batch_size = args.batch_size
    else:
        saliency_method = method(model=model,
                                 target_layers=target_layers,
                                 reshape_transform=reshape_transform,
                                 use_cuda=torch.cuda.is_available())
        saliency_method.batch_size = args.batch_size
    
    corrs = []
    for fname in tqdm(os.listdir(args.in_path)):
        video = mpy.VideoFileClip(args.in_path+fname)
        #  Use highest probability class from first frame for CAM
        start_frame = image_transform(video.get_frame(0)).to(device).unsqueeze(0)
        start_output = model(start_frame)
        class_idx = start_output.argmax().item()

        corr = calculate_mean_correlation(
            video=video,
            device=device,
            saliency_method=saliency_method,
            class_idx=class_idx,
            image_transform=image_transform,
            image_normalize=image_normalize,
            is_backprop=is_backprop
        )
    
        corrs.append(corr)
    
    #  Print mean correlation
    print(np.nanmean(corrs))


if __name__ == '__main__':
    main()