from sklearn.preprocessing import StandardScaler
from torch.utils.data  import DataLoader, random_split, TensorDataset
import torch
import numpy as np
from torchvision import transforms
import nibabel as nib
import torchio as tio


DATASET_TYPES_NUMERIC = {'Synthetic', 'Credit', 'Weather', 'IXITiny'}
DATASET_TYPES_CATEGORICAL = {'CIFAR', 'EMNIST'}

class Transformations:
    @staticmethod
    def get_metric(dataset):
        dataset_mapping = {
            'Synthetic': StandardScaler(),
            'Credit':  StandardScaler(),
            'Weather':  StandardScaler(),
            'EMNIST':  None,
            'CIFAR': transforms.Compose([transforms.ToPILImage(),
                                         transforms.Resize((224, 224)),
                                         transforms.ToTensor()]),
            'IXITiny': tio.Compose([
                        tio.ToCanonical(),
                        tio.Resample(4),
                        tio.CropOrPad((48, 60, 48)),
                        tio.OneHot()])
            
        }
        return dataset_mapping[dataset]

class DataPreprocessor:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.transform = Transformations.get_metric(self.dataset)

    def split_and_scale_data(self, X, y):
        if self.dataset in DATASET_TYPES_NUMERIC:
            X_tensor, y_tensor = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        elif self.dataset in DATASET_TYPES_CATEGORICAL:
            X_tensor, y_tensor = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
            if self.dataset == 'EMNIST':
                X_tensor.unsqueeze_(1)
            elif self.dataset == 'CIFAR':
                X_tensor = torch.stack([self.transform(image) for image in X_tensor])
        
        
        #train,val,test split
        train_data, val_data, test_data = self.split(X_tensor, y_tensor) 
        
        #transform
        if self.dataset in ['Synthetic', 'Credit', 'Weather']:
            train_data, val_data, test_data = self.numeric_transformation(train_data, val_data, test_data)

        return train_data, val_data, test_data

    def create_dataloaders(self, train_data, val_data, test_data):
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=self.batch_size)
        test_loader = DataLoader(test_data, batch_size=self.batch_size)
        
        return train_loader, val_loader, test_loader

    def preprocess(self, X, y):
        train_data, val_data, test_data = self.split_and_scale_data(X, y)
        return self.create_dataloaders(train_data, val_data, test_data)
    
    def split(self, X_tensor, y_tensor, test_size=0.2, val_size = 0.2):
        full_dataset = TensorDataset(X_tensor, y_tensor)
        test_size = int(test_size * len(full_dataset))
        train_val_size = len(full_dataset) - test_size
        val_size = int(val_size * train_val_size)
        train_size = train_val_size - val_size
        train_data, test_data = random_split(full_dataset, [train_val_size, test_size])
        train_data, val_data = random_split(train_data, [train_size, val_size])
        return train_data, val_data, test_data

    def numeric_transformation(self, train_data, val_data, test_data):
        X_train = self.transform.fit_transform(train_data[:][0].numpy())
        X_val = self.transform.transform(val_data[:][0].numpy())
        X_test = self.transform.transform(test_data[:][0].numpy())
        train_data = TensorDataset(torch.tensor(X_train, dtype=torch.float32), train_data[:][1])
        val_data = TensorDataset(torch.tensor(X_val, dtype=torch.float32), val_data[:][1])
        test_data = TensorDataset(torch.tensor(X_test, dtype=torch.float32), test_data[:][1])
        return train_data, val_data, test_data
             


