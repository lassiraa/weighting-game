import json
from typing import Callable
from functools import partial

import cv2
import torch
import torchvision.transforms as transforms
from torchvision.transforms.functional import resized_crop
from torch.utils.data import DataLoader
import numpy as np
from scipy import stats
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

from utils import CocoStability, reshape_transform_vit, load_model_with_target_layers, scale_image


#  Setting manual seed for RandomResizedCrop
torch.manual_seed(52)


def calculate_saliency(
    saliency_method: Callable,
    input: torch.tensor,
    idx: int,
    is_backprop: bool
) -> np.ndarray:
    if is_backprop:
        saliency_map = saliency_method(input, target_category=idx)
        saliency_map = saliency_map.sum(axis=2).reshape(1, 224, 224)
        saliency_map = np.where(saliency_map > 0, saliency_map, 0)
        saliency_map = scale_image(saliency_map, 1)
    else:
        saliency_map = saliency_method(input, [ClassifierOutputTarget(idx)])
    return saliency_map


def measure_stability(
    coco_loader: DataLoader,
    device: torch.device,
    model: torch.nn.Module,
    saliency_method: Callable,
    is_backprop: bool
) -> tuple[np.ndarray]:
    correlations = []

    for input in tqdm(coco_loader):
        #  Process randomized zoom and pan via random resized crop.
        i, j, h, w = transforms.RandomResizedCrop.get_params(input, scale=(0.75, 0.9), ratio=(1,1))
        cropped_input = resized_crop(input, i, j, h, w, size=(224,224))

        input = input.to(device)
        cropped_input = cropped_input.to(device)
        
        #  Calculate highest probability class for class-guided methods.
        idx = model(input).argmax().item()

        #  Process saliency maps.
        saliency_map = calculate_saliency(saliency_method, input, idx, is_backprop)
        cropped_saliency_map = calculate_saliency(saliency_method, cropped_input, idx, is_backprop)

        #  Process same zoom/pan to original image's saliency map to line up with cropped_saliency_map.
        saliency_map = torch.from_numpy(saliency_map)
        saliency_map = resized_crop(saliency_map, i, j, h, w, size=(224,224))
        saliency_map = saliency_map.numpy()

        #  Calculate the correlation.
        corr, _ = stats.spearmanr(cropped_saliency_map, saliency_map, axis=None)
        
        #  Correlation is not defined for constant arrays, which can happen.
        #  Therefore we skip those.
        if corr:
            correlations.append(corr)

    return correlations


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
    mask_dilation = partial(cv2.dilate, kernel=np.ones((9,9)), iterations=1)
    mask_transform = transforms.Compose([
        mask_dilation,
        transforms.ToTensor(),
        transforms.Resize(224),
        transforms.CenterCrop(224)
    ])

    coco_dset = CocoStability(
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
    parser = argparse.ArgumentParser(description='Measure explanation stability using zoom/pan')
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

    results = measure_stability(
        coco_loader=coco_loader,
        device=device,
        model=model,
        saliency_method=saliency_method,
        is_backprop=is_backprop
    )

    #  Save image to annotation dictionary as json
    with open(f'data/{args.model_name}-{args.method}-transformation_stability.json', 'w') as fp:
        json.dump(results, fp)