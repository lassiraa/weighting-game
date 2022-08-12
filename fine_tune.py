import time

import torchvision.models as models
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
from utils import CocoClassification


def get_model_to_fine_tune(model_name: str, device: torch.device):

    if model_name == 'vgg16_bn':
        model = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
        #  Using feature extraction so only output layer is fine-tuned
        for param in model.parameters():
            param.requires_grad = False
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, 80)

    if model_name == 'vit_b_32':
        model = models.vit_b_32(weights=models.ViT_B_32_Weights.IMAGENET1K_V1)
        #  Using feature extraction so only output layer is fine-tuned
        for param in model.parameters():
            param.requires_grad = False
        in_features = model.heads[0].in_features
        model.heads[0] = nn.Linear(in_features, 80)

    if model_name == 'swin_t':
        model = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
        #  Using feature extraction so only output layer is fine-tuned
        for param in model.parameters():
            param.requires_grad = False
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, 80)

    if model_name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        #  Using feature extraction so only output layer is fine-tuned
        for param in model.parameters():
            param.requires_grad = False
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 80)

    model = model.to(device)
    #  Getting all parameters that need to be optimized
    params_to_update = []
    for _, param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)

    return model, params_to_update


def fine_tune(model: nn.Module,
              optimizer: optim.Adam,
              scheduler: ExponentialLR,
              coco_loader_train: DataLoader,
              coco_loader_val: DataLoader,
              params: dict,
              model_name: str,
              device: torch.device,
              checkpoint_dir: str):
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(params['epochs']):
        t0 = time.time()
        tr_loss = []
        val_loss = []

        model.train()
        
        for inputs, labels in coco_loader_train:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            tr_loss.append(loss.item())

        model.eval()

        with torch.no_grad():

            for inputs, labels in coco_loader_val:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                val_loss.append(loss.item())

        scheduler.step()
        tr_loss = np.mean(tr_loss)
        val_loss = np.mean(val_loss)
        print(f'Time for epoch {epoch}: {time.time()-t0}')
        print(f'Tr. loss {tr_loss} | Val. loss {val_loss}')

        #  Save model state every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(
                model.state_dict(),
                f'{checkpoint_dir}{model_name}_coco_ep{epoch}.pt'
            )
    
    return model


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Finetune network')
    parser.add_argument('--images_dir', type=str,
                        help='path to coco root directory containing image folders')
    parser.add_argument('--ann_dir', type=str,
                        help='path to root directory containing annotations')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.90,
                        help='gamma for exponential lr scheduler')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of training iterations')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size for training')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='workers for dataloader')
    parser.add_argument('--model_name', type=str, default='vit_b_32',
                        help='name of model used for training',
                        choices=['vit_b_32', 'vgg16_bn', 'resnet50', 'swin_t'])
    parser.add_argument('--checkpoint_dir', type=str,
                        help='path to save checkpoint')
    args = parser.parse_args()

    training_params = dict(
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        gamma=args.gamma,
        model_name=args.model_name
    )

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,
                             std=std)
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,
                             std=std)
    ])


    coco_dset_train = CocoClassification(
        root=args.images_dir + 'train2017/',
        annFile=args.ann_dir + 'instances_train2017.json',
        transform=train_transform,
        target_transform=None
    )
    coco_loader_train = DataLoader(
        coco_dset_train,
        batch_size=training_params['batch_size'],
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers
    )

    coco_dset_val = CocoClassification(
        root=args.images_dir + 'val2017/',
        annFile=args.ann_dir + 'instances_val2017.json',
        transform=val_transform,
        target_transform=None
    )
    coco_loader_val = DataLoader(
        coco_dset_val,
        batch_size=training_params['batch_size'],
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers
    )

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model, params_to_update = get_model_to_fine_tune(
        args.model_name, device
    )

    optimizer = optim.Adam(params_to_update, lr=training_params['lr'])
    scheduler = ExponentialLR(optimizer, gamma=0.9)

    model = fine_tune(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        coco_loader_train=coco_loader_train,
        coco_loader_val=coco_loader_val,
        params=training_params,
        model_name=args.model_name,
        device=device,
        checkpoint_dir=args.checkpoint_dir
    )


if __name__ == '__main__':
    main()
