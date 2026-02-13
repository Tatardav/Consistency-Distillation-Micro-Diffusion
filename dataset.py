import os
import json
import csv
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class ImageCaptionDataset(Dataset):
    def __init__(self, data_dir, resolution=512, max_samples=None):
        self.data_dir = data_dir

        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        self.captions = {}
        json_path = os.path.join(data_dir, "captions.json")
        csv_path = os.path.join(data_dir, "captions.csv")

        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                self.captions = json.load(f)
        elif os.path.exists(csv_path):
            with open(csv_path, 'r', newline='\n') as f:
                reader = csv.reader(f)
                for i, row in enumerate(reader):
                    if i == 0:
                        continue
                    if len(row) >= 2:
                        self.captions[row[0]] = row[1]
        else:
            raise FileNotFoundError(f"No captions.json or captions.csv in {data_dir}")

        images_dir = os.path.join(data_dir, "images")
        all_images = sorted([
            f for f in os.listdir(images_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
        ])
        self.image_files = [f for f in all_images if f in self.captions]
        if max_samples:
            self.image_files = self.image_files[:max_samples]

        print(f"Loaded {len(self.image_files)} image-caption pairs from {data_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        fname = self.image_files[idx]
        image = Image.open(os.path.join(self.data_dir, "images", fname)).convert("RGB")
        image = self.transform(image)
        return {"image": image, "caption": self.captions[fname]}


class InfiniteSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, shuffle=True, seed=0):
        super().__init__(dataset)
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        order = np.arange(len(self.dataset))
        rnd = np.random.RandomState(self.seed)
        if self.shuffle:
            rnd.shuffle(order)
        idx = 0
        while True:
            i = idx % len(order)
            yield order[i]
            if self.shuffle and i == len(order) - 1:
                rnd.shuffle(order)
            idx += 1


def get_dataloader(data_dir, resolution=512, batch_size=2, max_samples=None):
    dataset = ImageCaptionDataset(data_dir, resolution, max_samples)
    sampler = InfiniteSampler(dataset, shuffle=True)
    return DataLoader(
        dataset, sampler=sampler,
        batch_size=batch_size, num_workers=4,
        pin_memory=True, drop_last=True,
    )
