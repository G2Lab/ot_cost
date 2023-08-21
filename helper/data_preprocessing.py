from sklearn.preprocessing import StandardScaler
from torch.utils.data  import DataLoader, random_split, TensorDataset
from sklearn.model_selection import train_test_split
import torch
import numpy as np
from torchvision import transforms
import nibabel as nib
import torchio as tio
from abc import ABC, abstractmethod
from torch.utils.data import ConcatDataset


DATASET_TYPES_TABULAR = {'Synthetic', 'Credit', 'Weather'}
DATASET_TYPES_IMAGE = {'CIFAR', 'EMNIST', 'IXITiny'}
torch.manual_seed(1)
np.random.seed(1)

def get_dataset_handler(dataset_name):
    if dataset_name in DATASET_TYPES_TABULAR:
        return TabularDatasetHandler(dataset_name)
    elif dataset_name in DATASET_TYPES_IMAGE:
        return ImageDatasetHandler(dataset_name)


class AbstractDatasetHandler(ABC):
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    @abstractmethod
    def preprocess_data(self, X, y):
        pass

class TabularDatasetHandler(AbstractDatasetHandler):
    def __init__(self, dataset_name):
        super().__init__(dataset_name)
        self.scaler = StandardScaler()
        self.scaler_label = StandardScaler()
    def preprocess_data(self, dataloader, fit_transform = False):
        X, y = dataloader
        if fit_transform:
            X_tensor = torch.tensor(self.scaler.fit_transform(X), dtype=torch.float32)
            if self.dataset_name == 'Weather':
                y_tensor = torch.tensor(self.scaler_label.fit_transform(y.reshape(-1, 1)), dtype=torch.float32)
            else:
                y_tensor = torch.tensor(y, dtype=torch.float32)    
        else:
            X_tensor = torch.tensor(self.scaler.transform(X), dtype=torch.float32)
            if self.dataset_name == 'Weather':
                y_tensor = torch.tensor(self.scaler_label.transform(y.reshape(-1, 1)), dtype=torch.float32)   
            else:
                y_tensor = torch.tensor(y, dtype=torch.float32)    
        return TensorDataset(X_tensor, y_tensor)
    
class ImageDatasetHandler(AbstractDatasetHandler):
    def preprocess_data(self, X, y, fit_transform = False):
        if self.dataset_name == 'EMNIST':
                X_tensor.unsqueeze_(1)
        elif self.dataset_name == 'CIFAR':
                transform = transforms.Compose([
                                                transforms.ToPILImage(),
                                                transforms.Resize((224, 224)),
                                                transforms.ToTensor()])
                X_tensor = torch.stack([transform(image) for image in X_tensor])
        elif self.dataset_name in ['IXITiny']:
            transform = tio.Compose([
                                tio.ToCanonical(),
                                tio.Resample(4),
                                tio.CropOrPad((48, 60, 48)),
                                tio.OneHot()
                                ])
            image_files = [torch.load(path) for path in X]
            label_files = [torch.tensor(nib.load(path).get_fdata(), dtype=torch.float).unsqueeze(0) for path in y]
            X_tensor = torch.stack(image_files)
            y_tensor = torch.stack([transform(label) for label in label_files])
        return X_tensor, y_tensor

class DataPreprocessor:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.handler = get_dataset_handler(self.dataset)

    def preprocess(self, X, y):
        if self.dataset in DATASET_TYPES_TABULAR:
            train_data, val_data, test_data = self.split(X, y) 
            train_data = self.handler.preprocess_data(train_data, fit_transform= True)
            val_data = self.handler.preprocess_data(val_data, fit_transform= False)
            test_data = self.handler.preprocess_data(test_data, fit_transform= False)
        elif self.dataset in DATASET_TYPES_IMAGE:
            X_tensor, y_tensor = self.handler.preprocess_data(X, y)
            train_data, val_data, test_data = self.split(X_tensor, y_tensor) 
        return self.create_dataloaders(train_data, val_data, test_data)
    
    def split(self, X, y, test_size=0.2, val_size = 0.2):
        if self.dataset in DATASET_TYPES_TABULAR:
            X_train_temp, X_test, y_train_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train, X_val, y_train, y_val = train_test_split(X_train_temp, y_train_temp, test_size=0.2, random_state=42)
            return (X_train, y_train), (X_val, y_val), (X_test, y_test)

        elif self.dataset in DATASET_TYPES_IMAGE:
            full_dataset = TensorDataset(X, y)
            n = len(full_dataset)
            test_size = int(test_size * n)
            val_size = int(val_size * n)
            train_size = n - test_size - val_size
            train_data, val_data, test_data = random_split(full_dataset, [train_size, val_size, test_size])
            return train_data, val_data, test_data
    
    def create_dataloaders(self, train_data, val_data, test_data):
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=self.batch_size)
        test_loader = DataLoader(test_data, batch_size=self.batch_size)
        return train_loader, val_loader, test_loader
    
    def combine(self, train_loader_1, train_loader_2, val_loader_1, val_loader_2):
        train_loader_combined = self.concatenate_data(train_loader_1, train_loader_2, True)
        val_loader_combined = self.concatenate_data(val_loader_1, val_loader_2, False)
        return train_loader_combined, val_loader_combined
    
    def concatenate_data(self, loader_1, loader_2, shuffle):
        X1, y1 = loader_1.dataset.tensors
        X2, y2 = loader_2.dataset.tensors
        X = torch.cat((X1, X2), dim = 0)
        y = torch.cat((y1, y2), dim = 0)
        dataset = TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
        return loader


