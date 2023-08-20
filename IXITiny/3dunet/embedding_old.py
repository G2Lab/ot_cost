import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchio as tio
from pathlib import Path
import copy
from unet import UNet

ROOT_DIR = '/gpfs/commons/groups/gursoy_lab/aelhussein/ot_cost/otcost_fl'
BATCH_SIZE = 8
LR = 1e-3
EPOCHS = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Autoencoder(nn.Module):
    def __init__(self, n_emb):
        super(Autoencoder, self).__init__()
        self.n_emb = n_emb
        
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1), 
            nn.ReLU(),
            nn.Conv3d(32, 64, 3, padding=1),
            nn.ReLU(), 
            nn.MaxPool3d(2, 2),
            nn.Conv3d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2, 2),
            nn.Conv3d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2, 2)
        )
        
        self.bottleneck = nn.Sequential(
            nn.Linear(256*6*7*6, n_emb),
            nn.ReLU()
        )
        
        self.expand = nn.Sequential(
            nn.Linear(n_emb, 256*6*7*6),
            nn.ReLU()  
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(256, 128, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose3d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(128, 64, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 1, kernel_size=(3, 7, 3), stride=1, padding=1),
            nn.Sigmoid()
        )

        
    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.bottleneck(x)
        x = self.expand(x)
        x = x.view(x.size(0), 256, 6, 7, 6)  
        x = self.decoder(x)
        return x

class Autoencoder(nn.Module):
    def __init__(self, n_emb):
        super().__init__()
        self.n_emb = n_emb
        
        # Encoder
        self.enc_conv1 = nn.Conv3d(1, 64, 3, padding=1) 
        self.enc_conv2 = nn.Conv3d(64, 128, 3, padding=1)
        self.enc_conv3 = nn.Conv3d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool3d(2, 2)
        
        # Bottleneck
        self.bottleneck = nn.Linear(256*6*7*6, self.n_emb)
        
        # Decoder
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec_conv1 = nn.Conv3d(256, 128, 3, padding=1)
        self.dec_conv2 = nn.Conv3d(128, 64, 3, padding=1)
        self.dec_conv3 = nn.ConvTranspose3d(64, 1, 3, padding=1)
        
    def forward(self, x):
        # Encoder
        x = F.relu(self.enc_conv1(x))
        x = self.pool(x)
        x = F.relu(self.enc_conv2(x))
        x = self.pool(x)
        x = F.relu(self.enc_conv3(x))
        x = self.pool(x)
        
        # Bottleneck
        x = x.view(x.size(0),-1)
        x = self.bottleneck(x)
        
        # Decoder
        x = x.view(x.size(0), 256, 6, 7, 6)
        x = self.upsample(x)
        x = F.relu(self.dec_conv1(x))
        x = self.upsample(x)
        x = F.relu(self.dec_conv2(x))
        x = self.upsample(x)
        x = torch.sigmoid(self.dec_conv3(x))
        
        return x

def load_data():
    training_split_ratio = 0.9
    dataset_dir = Path('/gpfs/commons/groups/gursoy_lab/aelhussein/ot_cost/otcost_fl/data/IXITiny')
    images_dir = dataset_dir / 'image'
    labels_dir = dataset_dir / 'label'
    image_paths = sorted(images_dir.glob('*.nii.gz'))
    label_paths = sorted(labels_dir.glob('*.nii.gz'))

    subjects = []
    for (image_path, label_path) in zip(image_paths, label_paths):
        subject = tio.Subject(
            mri=tio.ScalarImage(image_path),
            brain=tio.LabelMap(label_path),
        )
        subjects.append(subject)
    dataset = tio.SubjectsDataset(subjects)
    training_transform = tio.Compose([
        tio.ToCanonical(),
        tio.Resample(4),
        tio.CropOrPad((48, 60, 48)),
        tio.RandomMotion(p=0.2),
        tio.RandomBiasField(p=0.3),
        tio.RandomNoise(p=0.5),
        tio.RandomFlip(),
        tio.OneOf({
            tio.RandomAffine(): 0.8,
            tio.RandomElasticDeformation(): 0.2,
        }),
        tio.OneHot(),
    ])

    validation_transform = tio.Compose([
        tio.ToCanonical(),
        tio.Resample(4),
        tio.CropOrPad((48, 60, 48)),
        tio.OneHot(),
    ])

    num_subjects = len(dataset)
    num_training_subjects = int(training_split_ratio * num_subjects)
    num_validation_subjects = num_subjects - num_training_subjects

    num_split_subjects = num_training_subjects, num_validation_subjects
    training_subjects, validation_subjects = torch.utils.data.random_split(subjects, num_split_subjects)

    training_set = tio.SubjectsDataset(
        training_subjects, transform=training_transform)

    validation_set = tio.SubjectsDataset(
        validation_subjects, transform=validation_transform)

    training_batch_size = 8
    validation_batch_size = training_batch_size

    train_loader = torch.utils.data.DataLoader(
        training_set,
        batch_size=training_batch_size,
        shuffle=True,
    )

    val_loader = torch.utils.data.DataLoader(
        validation_set,
        batch_size=validation_batch_size,
        )

    return train_loader, val_loader

#Train autoencoder
def train_autoencoder(n_emb):
    train_loader, val_loader = load_data()

    model = Autoencoder(n_emb)
    if f'model_checkpoint_{n_emb}.pth' in os.listdir(f'{ROOT_DIR}/data/IXITiny/'):
        state_dict = torch.load(f'{ROOT_DIR}/data/IXITiny/model_checkpoint_{n_emb}.pth')
        model.load_state_dict(state_dict)
    
    model = model.to(DEVICE)
    criterion = nn.MSELoss()
    criterion = criterion.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_val_loss = np.inf
    patience = 10
    no_improvement_count = 0 

    train_losses = []
    val_losses = []
    for epoch in range(EPOCHS):
        train_loss_sum = 0
        num_batches = 0
        for inputs in train_loader:
            inputs = inputs['mri']['data'].to(DEVICE)
            outputs = model(inputs)
            train_loss = criterion(outputs, inputs)
            train_loss_sum += train_loss.item()
            num_batches += 1
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        avg_train_loss = train_loss_sum / num_batches if num_batches > 0 else 0
        train_losses.append(avg_train_loss)

        with torch.no_grad():
            val_loss_sum = 0
            num_batches = 0
            for inputs in val_loader:
                inputs = inputs['mri']['data'].float().to(DEVICE)
                outputs = model(inputs)
                val_loss = criterion(outputs, inputs)
                val_loss_sum += val_loss.item()
                num_batches += 1
        avg_val_loss = val_loss_sum / num_batches if num_batches > 0 else 0
        val_losses.append(avg_val_loss)
        if epoch % 10 == 0:
            print(avg_train_loss, avg_val_loss)

        # Early stopping
        if avg_val_loss < best_val_loss:
            model.to('cpu')
            torch.save(model.state_dict(), f'{ROOT_DIR}/data/IXITiny/model_checkpoint_{n_emb}.pth')
            model.to(DEVICE)
            best_val_loss = avg_val_loss
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count >= patience:
                print(f"Stopping early after {patience} epochs without improvement.")
                break
    return best_val_loss, train_losses, val_losses

def main():
    print(DEVICE)

    n_embs = [2048]
    losses = {}
    for n_emb in n_embs:
        losses[n_emb] = train_autoencoder(n_emb)
    with open(f'{ROOT_DIR}/data/IXITiny/losses.json', 'w') as f:
        json.dump(losses, f)
    print("Losses saved to file.")

if __name__ == '__main__':
    main()
    