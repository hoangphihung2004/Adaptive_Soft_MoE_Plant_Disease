import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):

    def __init__(self, data, transform=None):
        self.df = data
        self.transform = transform

    def __getitem__(self, index):
        label = self.df.loc[index, "Label"]
        path = os.path.join(self.df.loc[index, "Path"])
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.df)

def dataset(data, config):

    df_train = data.loc[data["Type"] == "Train"].copy().reset_index(drop=True)
    df_val = data.loc[data["Type"] == "Validation"].copy().reset_index(drop=True)
    df_test = data.loc[data["Type"] == "Test"].copy().reset_index(drop=True)

    label_encoder = LabelEncoder()
    df_train["Label"] = label_encoder.fit_transform(df_train["Label"])
    df_val["Label"] = label_encoder.transform(df_val["Label"])
    df_test["Label"] = label_encoder.transform(df_test["Label"])

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    data_transform = {
        "Train": transforms.Compose([
            transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(20),
            transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        "Validation": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        "Test": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }

    train_dataset = MyDataset(data=df_train, transform=data_transform["Train"])
    valid_dataset = MyDataset(data=df_val, transform=data_transform["Validation"])
    test_dataset = MyDataset(data=df_test, transform=data_transform["Test"])

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=config.batch_size, shuffle=False)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=config.batch_size, shuffle=False)

    data_loader = {"Train": train_dataloader, "Val": valid_dataloader}

    return data_loader, test_dataloader, label_encoder


