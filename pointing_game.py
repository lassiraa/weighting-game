import json
from typing import Callable
from functools import partial

import cv2
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
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
from tqdm import tqdm

from utils import CocoExplainabilityMeasurement, pointing_game_hit, reshape_transform_vit, \
    load_model_with_target_layers, scale_image


def measure_pointing_game(
    coco_loader: DataLoader,
    device: torch.device,
    saliency_method: Callable,
    is_backprop: bool
) -> tuple[np.ndarray]:
    accuracy = []

    for inputs, class_to_targets in tqdm(coco_loader):
        inputs = inputs.to(device)

        for idx, target in class_to_targets.items():
            mask = target['mask'].to(device=device, dtype=torch.bool)

            #  Process saliency map
            if is_backprop:
                saliency_map = saliency_method(inputs, target_category=idx)
                saliency_map = saliency_map.sum(axis=2).reshape(1, 224, 224)
                saliency_map = np.where(saliency_map > 0, saliency_map, 0)
                saliency_map = scale_image(saliency_map, 1)
                saliency_map = torch.from_numpy(saliency_map).to(device)
            else:
                saliency_map = saliency_method(inputs, [ClassifierOutputTarget(idx)])
                saliency_map = torch.from_numpy(saliency_map).to(device)

            #  Calculate whether maximum saliency point is within correct class.
            hit = pointing_game_hit(saliency_map, mask)
            accuracy.append(hit)
    
    return accuracy


def get_dataloader(
    path2data: str,
    path2json: str,
    num_workers: int
) -> None:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    mask_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(224),
        transforms.CenterCrop(224)
    ])

    coco_dset = CocoExplainabilityMeasurement(
        root=path2data,
        annFile=path2json,
        transform=image_transform,
        target_transform=mask_transform
    )
    coco_loader = DataLoader(
        coco_dset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )
    
    return coco_loader


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Measure accuracy of explanation method')
    parser.add_argument('--images_dir', type=str,
                        default='/media/lassi/Data/datasets/coco/images/val2017/',
                        help='path to coco root directory containing image folders')
    parser.add_argument('--ann_path', type=str,
                        default='/media/lassi/Data/datasets/coco/annotations/instances_val2017.json',
                        help='path to root directory containing annotations')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='batch size for cam methods')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='workers for dataloader')
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

    if is_vit:
        reshape_transform = reshape_transform_vit

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
            
    coco_loader = get_dataloader(
        path2data=args.images_dir,
        path2json=args.ann_path,
        num_workers=args.num_workers
    )

    results = measure_pointing_game(
        coco_loader=coco_loader,
        device=device,
        saliency_method=saliency_method,
        is_backprop=is_backprop
    )

    #  Save image to annotation dictionary as json
    with open(f'data/{args.model_name}-{args.method}-pointing_game.json', 'w') as fp:
        json.dump(results, fp)