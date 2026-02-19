import torch
import os
import random
from PIL import Image, ImageOps
from torch.utils.data import Dataset


import torch
import os
import random
from PIL import Image, ImageOps
from torch.utils.data import Dataset


class SimpleNetDataset(Dataset):
    def __init__(self, path="", transform=None, useHFlip=False, useVFlip=False, num_images=None):
        self.path = path
        self.transform = transform
        self.useHFlip = useHFlip
        self.useVFlip = useVFlip
        self.num_images = num_images

        self.image_paths = []
        self.labels = []

        # Create label mapping from sorted folder names
        class_names = sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}

        # Collect images and labels
        for class_name in class_names:
            class_dir = os.path.join(path, class_name)
            image_files = sorted([
                os.path.join(class_dir, f)
                for f in os.listdir(class_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ])

            if self.num_images is not None:
                image_files = image_files[:self.num_images]

            self.image_paths.extend(image_files)
            self.labels.extend([self.class_to_idx[class_name]] * len(image_files))

        self.total_images = len(self.image_paths)

    def __len__(self):
        return self.total_images

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]
        image = Image.open(image_path).convert('RGB')

        # Optional flipping
        if self.useHFlip and random.random() > 0.5:
            image = ImageOps.mirror(image)
        if self.useVFlip and random.random() > 0.5:
            image = ImageOps.flip(image)

        if self.transform:
            image = self.transform(image)
            image = image[[2, 1, 0], :, :]  # RGB to BGR if needed

        return image, label, image_path

