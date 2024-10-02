import os
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
import torchvision.transforms.functional

import matplotlib.pyplot as plt

class FishDataset(Dataset):
    def __init__(self, images_dir, masks_dir, crop_size=128):
        # get directories
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.fish_folders = sorted(os.listdir(images_dir))
        self.classes = self.fish_folders.copy()  # folder names are the labels
        self.classes.insert(0, 'background')
        # create transforms
        self.image_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((crop_size, crop_size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.mask_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((crop_size, crop_size)),
            torchvision.transforms.ToTensor()
        ])

        # get class distributions
        self.class_weights = []
        background_counts = 0
        for fish_folder in self.fish_folders:
            mask_folder = fish_folder.replace('fish', 'mask')
            fish_mask_dir = os.path.join(masks_dir, mask_folder)

            # iterate over each mask in the fish folder
            counts = 0
            for mask_name in os.listdir(fish_mask_dir):
                mask_path = os.path.join(fish_mask_dir, mask_name)

                # load the mask and count mask values
                mask = np.array(Image.open(mask_path).convert("L"))
                counts += np.sum(mask==255)
                background_counts += np.sum(mask==0)
            self.class_weights.append(counts)
        self.class_weights.insert(0, background_counts)
        self.class_weights = 1.0 - (np.array(self.class_weights) / np.sum(self.class_weights))

    def __len__(self):
        return sum(len(os.listdir(os.path.join(self.images_dir, fish))) for fish in self.fish_folders)

    def __getitem__(self, idx):
        # used for converting from total index to each fishes image index
        cumulative_len = 0
        # label will be in order, 0 = background, 1 = fish_01, 2 = fish_02, etc.
        for i, fish_folder in enumerate(self.fish_folders):
            mask_folder = fish_folder.replace('fish', 'mask')
            fish_images = os.listdir(os.path.join(self.images_dir, fish_folder))
            # if index is in this fishes set
            if idx < cumulative_len + len(fish_images):
                # get which image
                local_idx = idx - cumulative_len
                image_name = fish_images[local_idx]
                image_path = os.path.join(self.images_dir, fish_folder, image_name)
                mask_name = image_name.replace('fish', 'mask')
                mask_path = os.path.join(self.masks_dir, mask_folder, mask_name)

                image = Image.open(image_path).convert("RGB")
                mask = Image.open(mask_path).convert("RGB")

                # apply transforms
                if self.image_transform:
                    image = self.image_transform(image)
                if self.mask_transform:
                    mask = self.mask_transform(mask)[0]
                    mask.type(torch.int64)
                    mask[mask==1] = i+1

                return image, mask, i+1
            
            # next fish
            cumulative_len += len(fish_images)

        # otherwise, not valid
        raise IndexError("Index out of range")
    
if __name__ == "__main__":
    # set up device
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(f'using: {torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"}')
    print(f'cpu cores available: {os.cpu_count()}')

    # FISH RECOGNITION TESTS
    print("FISH RECOGNITION GROUND TRUTH TESTS")

    # create dataset and split into dataloaders
    dataset = FishDataset(images_dir="datasets/fish-recognition-ground-truth-data/fish_image", masks_dir="datasets/fish-recognition-ground-truth-data/mask_image")

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    # stats and display
    train_images, train_masks, train_labels = next(iter(train_loader))
    print(f"Images batch shape: {train_images.size()}")
    print(f"Masks batch shape: {train_masks.size()}")
    print(f"Labels batch shape: {train_labels.size()}")

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(9, 9))
    for y in range(3):
        sample_idx = torch.randint(len(dataset), size=(1,)).item()
        img, mask, label = dataset[sample_idx]
        axes[y, 0].imshow(img.numpy().transpose(1,2,0))
        axes[y, 0].set_axis_off()
        axes[y, 0].set_title(dataset.classes[label-1])
        axes[y, 1].imshow(mask.numpy())
        axes[y, 1].set_axis_off()

    # can only show by saving
    # plt.savefig('fishes.png', bbox_inches='tight')