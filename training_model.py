import time
import os
import argparse
from typing import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

import torchvision
from torchvision import models

from datasets import FishDataset

import matplotlib.pyplot as plt
import numpy as np

def image_fix(arr, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    '''
    Clips, transposes and normalizes images to be displayed.
    '''
    return np.clip(np.transpose(arr, (1, 2, 0)) * std + mean, 0, 1)  # clip to visible rgb space

def train_model(dataloaders, dataset_sizes, device, model, crit_seg, crit_cls, optimizer, num_epochs):
    since = time.time()

    print('-' * 10)
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')

        # each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # set model to training mode
            else:
                model.eval()   # set model to evaluate mode

            running_loss = 0.0

            # iterate over data
            for inputs, masks, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                masks = masks.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase=='train'):
                    # get ouputs separately
                    outputs_seg = model[:-1](inputs)
                    outputs_cls = model[-1](outputs_seg).squeeze(3).squeeze(2)
                    interpolated_seg = nn.functional.interpolate(outputs_seg, size=(inputs.shape[2], inputs.shape[3]), mode='nearest')

                    # get losses separately and combine
                    Lseg = crit_seg(interpolated_seg, masks.type(torch.int64)).mean()
                    Lcls = crit_cls(outputs_cls, labels)
                    total_loss = Lseg + Lcls

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        total_loss.backward()
                        optimizer.step()

                # stats
                running_loss += total_loss.item() * inputs.size(0)

            # get stats for this epoch
            epoch_loss = running_loss / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f}')

        print('-' * 10)

    # finished, display stats
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    return model

if __name__ == '__main__':
    # get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='datasets/fish-recognition-ground-truth-data/fish_image', help="directory of the images")
    parser.add_argument('--mask_dir', type=str, default='datasets/fish-recognition-ground-truth-data/mask_image', help="directory of the masks")
    parser.add_argument('--train_val_split', type=float, default=0.7, help="decimal portion of data for training")
    parser.add_argument('--crop_size', type=int, default=128, help="crop size for the dataset")
    parser.add_argument('--batch_size', type=int, default=16, help="batch size")
    parser.add_argument('--num_workers', type=int, default=4, help="number of workers")
    parser.add_argument('--lr', type=float, default=0.0001, help="learning rate")
    parser.add_argument('--beta1', type=float, default=0.9, help="beta-1 for adam optimizer")
    parser.add_argument('--beta2', type=float, default=0.999, help="beta-2 for adam optimizer")
    parser.add_argument('--epochs', type=int, default=12, help="total number of epochs")
    args = parser.parse_args()

    # set up device
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(f'using: {torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"}')
    print(f'cpu cores available: {os.cpu_count()}')

    # get data
    print('loading data...')
    dataset = FishDataset(images_dir=args.image_dir, masks_dir=args.mask_dir, crop_size=args.crop_size)
    class_names = dataset.classes
    class_weights = dataset.class_weights

    train_size = args.train_val_split
    val_size = (1.0 - args.train_val_split) * (2.0/3.0)  # validation is 2/3 of remaining data
    test_size = 1.0 - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset), 'test': len(test_dataset)}

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    print('data loaded!')

    # build model on top of pretrained resnet18
    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # encoder is up to and including layer 3
    encoder = nn.Sequential(
        *list(resnet.children())[:-3]
    )
    # segmenter and classifier as described in table A.1
    # note that the output of encoder and segmenter needs to be \
    # interpolated by nearest-neighbour sampling to input image size \
    # when calculating loss
    segmenter = nn.Sequential(
        nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        nn.Conv2d(64, len(class_names), kernel_size=1, stride=1, padding=0)
    )
    classifier = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Conv2d(len(class_names), 8, kernel_size=1, stride=1, padding=0),
        nn.BatchNorm2d(8),
        nn.ReLU(),
        nn.Conv2d(8, len(class_names), kernel_size=1, stride=1, padding=0)
    )
    model = nn.Sequential(OrderedDict([
        ('encoder', encoder),
        ('segmenter', segmenter),
        ('classifier', classifier),
    ]))
    model = model.to(device)

    # optimizer acts on all parameters
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    
    # separate loss functions for segmentation and classification
    crit_seg = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights).type(torch.float32).to(device), reduction='none')
    crit_cls = nn.CrossEntropyLoss()

    print('starting training...')
    model = train_model(dataloaders, dataset_sizes, device, model, crit_seg, crit_cls, optimizer, args.epochs)
    print('finished training!')

    # testing data
    print('starting testing...')
    model.eval()
    for inputs, masks, labels in dataloaders['test']:
        inputs = inputs.to(device)
        masks = masks.to(device)
        labels = labels.to(device)

        # forward
        with torch.set_grad_enabled(False):
            # get ouputs separately
            outputs_seg = model[:-1](inputs)
            outputs_cls = model[-1](outputs_seg).squeeze(3).squeeze(2)
            interpolated_seg = nn.functional.interpolate(outputs_seg, size=(inputs.shape[2], inputs.shape[3]), mode='nearest')

            fig, axes = plt.subplots(nrows=args.batch_size, ncols=3, figsize=(9, 72))
            for i in range(args.batch_size):
                # computed mask
                _, indices = torch.max(interpolated_seg[i], dim=0)
                img = (indices / 24.0).cpu().numpy()

                axes[i, 0].imshow(image_fix(inputs[i].cpu().numpy()))
                axes[i, 0].set_title(f'Actual image of class: {class_names[labels[i].cpu().numpy()]}')
                axes[i, 1].imshow(masks[i].cpu().numpy())
                axes[i, 1].set_title(f'Actual mask')
                axes[i, 2].imshow(img)
                axes[i, 2].set_title(f'Predicted mask of class: {class_names[outputs_cls[i].cpu().numpy().argmax()]}')
            # can only show by saving
            plt.savefig(f'output.png', bbox_inches='tight')

        # only one batch
        break
    print('finished testing!')