global ROOT_DIR
ROOT_DIR = '/gpfs/commons/groups/gursoy_lab/aelhussein/ot_cost/otcost_fl_rebase'
DATA_DIR = f'{ROOT_DIR}/data/ISIC'

import pandas as pd
import torch
import torch.nn as nn
import sys
import os
import numpy as np
sys.path.append(f'{ROOT_DIR}/code/helper')
sys.path.append(f'{ROOT_DIR}/code/ISIC/')
import trainers as tr
import pipeline as pp
import process_results as pr
import dataset
import importlib
importlib.reload(tr)
importlib.reload(pp)
importlib.reload(pr)
import pickle
from multiprocessing import Pool
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR

EPOCHS = 500
BATCH_SIZE = 64
RUNS = 2
DATASET = 'ISIC'
METRIC_TEST = 'Accuracy'
LEARNING_RATE = 1e-2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class efficientnetClassifier(nn.Module):
    def __init__(self):
        super(efficientnetClassifier, self).__init__()
        self.efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
        for name, param in self.efficientnet.named_parameters():
            if 'features' in name or 'classifier' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        self.efficientnet.classifier.fc = nn.Linear(1280, 7)

    def forward(self, x):
        logits = self.efficientnet(x)
        return logits

    def initialize_weights(self):
        nn.init.xavier_normal_(self.classifier.weight.data)
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias.data, 0)

        nn.init.xavier_normal_(self.features.weight.data)
        if self.features.bias is not None:
            nn.init.constant_(self.features.bias.data, 0)

def createModel():
    model = efficientnetClassifier()
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
    lr_scheduler = ExponentialLR(optimizer, gamma=0.9)
    return model, criterion, optimizer, lr_scheduler

def loadData(dataset, cost):
    train_data = dataset.FedIsic2019(train=True, data_path=DATA_DIR)
    val_data = dataset.FedIsic2019(train=False, pooled = True, data_path=DATA_DIR)
    train_loader = DataLoader(train_data, batch_size = BATCH_SIZE, shuffle = True)
    val_loader = DataLoader(val_data, batch_size = BATCH_SIZE, shuffle = True)
        
    
    return np.array(image_files), np.array(label_files)


def get_common_name(full_path):
    return os.path.basename(full_path).split('_')[0]

def align_image_label_files(image_files, label_files):
    labels_dict = {get_common_name(path): path for path in label_files}
    images_dict = {get_common_name(path): path for path in image_files}
    common_keys = sorted(set(labels_dict.keys()) & set(images_dict.keys()))
    sorted_labels = [labels_dict[key] for key in common_keys]
    sorted_images = [images_dict[key] for key in common_keys]
    return sorted_images, sorted_labels

def run_model_for_cost(inputs):
    c, loadData, DATASET, METRIC_TEST, BATCH_SIZE, EPOCHS, DEVICE, RUNS = inputs
    mp = pp.ModelPipeline(c, loadData, DATASET, METRIC_TEST, BATCH_SIZE, EPOCHS, DEVICE, RUNS)
    mp.set_functions(createModel())
    return mp.run_model_for_cost()


def main():
     ##run model on datasets
    cpu = int(os.environ.get('SLURM_CPUS_PER_TASK', 5))
    costs = [0.08, 0.28, 0.30]
    inputs = [(c, loadData, DATASET, METRIC_TEST, BATCH_SIZE, EPOCHS, DEVICE, RUNS) for c in costs]
    results = []
    if DEVICE == 'cpu':
        with Pool(cpu) as pool:
            results = pool.map(run_model_for_cost, inputs)
    else:
        for input in inputs:
            results.append(run_model_for_cost(input))

    losses = {}
    metrics_all = pd.DataFrame()
    for c, loss, metrics in results:
        losses[c] = loss
        metrics_all = pd.concat([metrics_all, metrics], axis=0)
    metrics_all.reset_index(inplace = True, drop = True)
    ##remove tensor
    try:
        for col in metrics_all.columns:
            metrics_all[col] = metrics_all[col].apply(lambda x: x.item() if isinstance(x, (torch.Tensor, np.ndarray)) and x.size == 1 else x)
    except:
        pass
    losses_df, test_losses_df = pp.loss_dictionary_to_dataframe(losses, costs, RUNS)
    

    ##Save results
    path_save = f'{ROOT_DIR}/results/{DATASET}'
    cost = f'{costs[0]}-{costs[-1]}'
    metrics_all.to_csv(f'{path_save}/{METRIC_TEST}_{cost}.csv', index=False)
    test_losses_df.to_csv(f'{path_save}/losses_{cost}.csv', index=False)
    with open(f'{path_save}/losses.pkl', 'wb') as f:
        pickle.dump(losses_df, f)


    ##Process results and graph
    save = True
    pr.process_results(DATASET, METRIC_TEST, costs, save)

if __name__ == '__main__':
    main()