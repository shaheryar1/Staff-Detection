from PIL import Image
from ensure import ensure_annotations
import pandas as pd
import numpy as np
from torchvision import transforms,
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
}

class StaffDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    @ensure_annotations
    def __getitem__(self, idx: int):
        image_path = self.dataframe.iloc[idx, 0]
        target = self.dataframe.iloc[idx, 1]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, target

def train_dataloader(train_df : pd.DataFrame,batch_size = 32):
    return DataLoader(StaffDataset(train_df, data_transforms['train']), batch_size=batch_size, shuffle=True)

def val_dataloader(val_df: pd.DataFrame ,batch_size = 32):
    return DataLoader(StaffDataset(val_df, data_transforms['val']), batch_size=batch_size)