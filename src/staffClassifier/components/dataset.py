from PIL import Image
from ensure import ensure_annotations
import pandas as pd
import numpy as np
from torchvision import transforms
import torch.nn as nn
import os
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


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


def prepare_dataset(path, split = True):
    BASE_PATH = path
    NON_STAFF_FOLDER = 'Non-Staff Images'
    STAFF_FOLDER = 'Staff Images'
    all_non_staff_images = os.listdir(os.path.join(BASE_PATH,NON_STAFF_FOLDER))
    all_non_staff_images = [str(os.path.join(BASE_PATH,NON_STAFF_FOLDER,x)) for x in all_non_staff_images]
    all_staff_images = os.listdir(os.path.join(BASE_PATH,STAFF_FOLDER))
    all_staff_images = [str(os.path.join(BASE_PATH,STAFF_FOLDER,x)) for x in all_staff_images]
    df = pd.DataFrame({
        'images': all_non_staff_images + all_staff_images,
        'target': list(np.zeros(len(all_non_staff_images)).astype(int)) + list(np.ones(len(all_staff_images)).astype(int))
    })
    df['images'] = df['images'].astype(str)
    df['target'] = df['target'].astype(int)

    df = pd.DataFrame({
        'images': all_non_staff_images + all_staff_images,
        'target': list(np.zeros(len(all_non_staff_images)).astype(int)) + list(np.ones(len(all_staff_images)).astype(int))
    })
    df['images'] = df['images'].astype(str)
    df['target'] = df['target'].astype(int)
    if split :
        train_df, val_df = train_test_split(df, test_size=0.30,shuffle=True)
        return train_df[0:100], val_df[0:100]
    else:
        return df