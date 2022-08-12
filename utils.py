from typing import Any, Callable, Optional, Tuple, List
import os

import numpy as np
import torch
import torchvision.models as models
import torch.nn as nn
from pycocotools.coco import COCO
from torchvision.datasets import VisionDataset
from PIL import Image


rng = np.random.default_rng(51)


def scale_image(
    img: Any,
    max: int = 1
) -> Any:
    return (img - img.min()) * (1/(img.max() - img.min()) * max)


def load_model_with_target_layers(
    model_name: str,
    device: torch.device
) -> nn.Module:
    if model_name == 'vgg16_bn':
        model = models.vgg16_bn(weights=None)
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, 80)
        target_layers = [model.features[-1]]

    if model_name == 'vit_b_32':
        model = models.vit_b_32(weights=None)
        in_features = model.heads[0].in_features
        model.heads[0] = nn.Linear(in_features, 80)
        target_layers = [model.encoder.layers[-1].ln_1]

    if model_name == 'swin_t':
        model = models.swin_t(weights=None)
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, 80)
        target_layers = [model.features[-1][-1].norm1]

    if model_name == 'resnet50':
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 80)
        target_layers = [model.layer4[-1]]
    
    model.load_state_dict(torch.load(f'{model_name}_coco.pt', map_location=device))
    return model, target_layers


def calculate_mass_within(
    saliency_map: torch.tensor,
    class_mask: torch.tensor
) -> float:
    mass = saliency_map.sum()
    mass_within = (saliency_map * class_mask).sum()
    return (mass_within / mass).item()


def pointing_game_hit(
    saliency_map: torch.tensor,
    class_mask: torch.tensor
) -> float:
    max_idx = saliency_map.argmax()
    return class_mask.flatten()[max_idx].item() 


def reshape_transform_vit(tensor, dim=7):
    #  Needed for ViT but not for Swin
    if tensor.shape[1] == (dim * dim + 1):
        tensor = tensor[:, 1:, :]
    
    result = tensor.reshape(tensor.size(0),
                            dim, dim, -1)

    #  Bring the channels to the first dimension,
    #  like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


class CocoClassification(VisionDataset):
    def __init__(
        self,
        root: str,
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)

        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        categories = self.coco.getCatIds()
        self.num_categories = len(categories)
        self.categories = {cat: idx for idx, cat in enumerate(categories)}

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, id: int) -> List[Any]:
        anns = self.coco.loadAnns(self.coco.getAnnIds(id))
        labels = np.zeros(self.num_categories, dtype=np.float32)

        for ann in anns:
            assert('category_id' in ann)
            idx = self.categories[ann['category_id']]
            labels[idx] = 1
        
        return labels

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)

        if self.transform is not None:
            image = self.transform(image)

        return image, target

    def __len__(self) -> int:
        return len(self.ids)


class CocoExplainabilityMeasurement(VisionDataset):
    def __init__(
        self,
        root: str,
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)

        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        categories = self.coco.getCatIds()
        self.num_categories = len(categories)
        self.categories = {cat: idx for idx, cat in enumerate(categories)}

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, id: int) -> List[Any]:
        anns = self.coco.loadAnns(self.coco.getAnnIds(id))
        class_to_targets = dict()

        for ann in anns:
            if 'category_id' not in ann:
                continue
            
            mask = self.coco.annToMask(ann).astype(bool)
            idx = self.categories[ann['category_id']]
            
            if idx in class_to_targets:
                class_to_targets[idx]['mask'] += mask
                continue
            
            labels = np.zeros(self.num_categories, dtype=np.float32)
            labels[idx] = 1
            class_to_targets[idx] = dict(
                mask=mask,
                labels=labels
            )
        
        return class_to_targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        class_to_targets = self._load_target(id)

        #  Transform both the masks and the images.
        #  If one is transformed, the other one needs to be as well.
        if self.transform is not None and self.target_transform is not None:
            image = self.transform(image)
            class_to_targets = {
                idx: {
                    'mask': self.target_transform(target['mask'].astype('float')),
                    'labels': target['labels']
                }
                for idx, target in class_to_targets.items()
            }

        return image, class_to_targets

    def __len__(self) -> int:
        return len(self.ids)


class CocoStability(VisionDataset):
    def __init__(
        self,
        root: str,
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)

        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        categories = self.coco.getCatIds()
        self.num_categories = len(categories)
        self.categories = {cat: idx for idx, cat in enumerate(categories)}

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)

        #  Transform original image to normalized and center cropped.
        if self.transform is not None:
            image = self.transform(image)

        return image

    def __len__(self) -> int:
        return len(self.ids)
