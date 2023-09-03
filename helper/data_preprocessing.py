from sklearn.preprocessing import StandardScaler
from torch.utils.data  import DataLoader, random_split, TensorDataset, Dataset
from sklearn.model_selection import train_test_split
import torch
import numpy as np
from torchvision import transforms
import nibabel as nib
import torchio as tio
from abc import ABC, abstractmethod
from PIL import Image

global ROOT_DIR
ROOT_DIR = '/gpfs/commons/groups/gursoy_lab/aelhussein/ot_cost/otcost_fl_rebase'
DATASET_TYPES_TABULAR = {'Synthetic', 'Credit', 'Weather'}
DATASET_TYPES_IMAGE = {'CIFAR', 'EMNIST', 'IXITiny', 'ISIC'}
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
    def preprocess_data(self, dl, fit_transform = False):
        X, y = dl
        if self.dataset_name in ['EMNIST', 'CIFAR']:
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.long)
            if self.dataset_name == 'EMNIST':
                    X_tensor.unsqueeze_(1)
            elif self.dataset_name == 'CIFAR':
                    transform = transforms.Compose([
                                                    transforms.ToPILImage(),
                                                    transforms.Resize((224, 224)),
                                                    transforms.ToTensor()])
                    X_tensor = torch.stack([transform(image) for image in X_tensor])
            return TensorDataset(X_tensor, y_tensor)
        elif self.dataset_name in ['IXITiny']:
            return IXITinyDataset(dl)
        elif self.dataset_name in ['ISIC']:
            return ISICDataset(dl)
            
        

class IXITinyDataset(Dataset):
    def __init__(self, data, transform=None):
        image_paths, label_paths = data
        self.image_paths = image_paths
        landmarks = tio.HistogramStandardization.train(
                        image_paths,
                        output_path=f'{ROOT_DIR}/data/IXITiny/landmarks.npy')
        self.label_paths = label_paths
        self.transform_image = tio.Compose([
                            tio.ToCanonical(),
                            tio.Resample(4),
                            tio.CropOrPad((48, 60, 48)),
                            #tio.HistogramStandardization({'mri': landmarks}),
                            tio.ZNormalization(masking_method=tio.ZNormalization.mean),
                            tio.OneHot()
                        ])
        
        self.transform_label = tio.Compose([
                            tio.ToCanonical(),
                            tio.Resample(4),
                            tio.CropOrPad((48, 60, 48)),
                            tio.OneHot()
                        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        #image = torch.load(image_path)
        image = torch.tensor(nib.load(image_path).get_fdata(), dtype=torch.float).unsqueeze(0)
        label = torch.tensor(nib.load(label_path).get_fdata(), dtype=torch.float).unsqueeze(0)

        image = self.transform_image(image)
        label = self.transform_label(label)
        return image, label
    

class ISICDataset(Dataset):
    def __init__(self, data, transform=None):
        image_paths, labels = data
        sz = 200
        mean=(0.585, 0.500, 0.486)
        std=(0.229, 0.224, 0.225)
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform if transform else transforms.Compose([transforms.ToTensor(),
                transforms.CenterCrop(sz),
                transforms.Normalize(mean, std)
                ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(label, dtype = torch.int64)
        return image, label
    

class DataPreprocessor:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.handler = get_dataset_handler(self.dataset)

    def preprocess(self, X, y):
        train_data, val_data, test_data = self.split(X, y) 
        return self.create_dataloaders(train_data, val_data, test_data)
    
    def preprocess_joint(self, X1, y1, X2, y2):
        train_data, val_data, test_data = self.split_joint(X1, y1, X2, y2) 
        return self.create_dataloaders(train_data, val_data, test_data)

    def split(self, X, y, test_size=0.2, val_size = 0.2):
        X_train_temp, X_test, y_train_temp, y_test = train_test_split(X, y, test_size = test_size, random_state=np.random.RandomState(42))
        X_train, X_val, y_train, y_val = train_test_split(X_train_temp, y_train_temp, test_size = val_size, random_state=np.random.RandomState(42))
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def split_joint(self, X1, y1, X2, y2):
        (X_train1, y_train1), (X_val1, y_val1), (X_test, y_test)= self.split(X1, y1)
        (X_train2, y_train2), (X_val2, y_val2), (_, _) = self.split(X2, y2)
        X_train = np.concatenate((X_train1, X_train2), axis = 0)
        y_train = np.concatenate((y_train1, y_train2), axis = 0)
        X_val = np.concatenate((X_val1, X_val2), axis = 0)
        y_val = np.concatenate((y_val1, y_val2), axis = 0)
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def create_dataloaders(self, train_data, val_data, test_data):
        train_data = self.handler.preprocess_data(train_data, fit_transform= True)
        val_data = self.handler.preprocess_data(val_data, fit_transform= False)
        test_data = self.handler.preprocess_data(test_data, fit_transform= False)
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle = False)
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle = False)
        return train_loader, val_loader, test_loader