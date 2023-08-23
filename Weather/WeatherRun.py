global ROOT_DIR
ROOT_DIR = '/gpfs/commons/groups/gursoy_lab/aelhussein/ot_cost/otcost_fl_rebase'

import pandas as pd
import os
import torch
import torch.nn as nn
import sys
import numpy as np
sys.path.append(f'{ROOT_DIR}/code/helper')
import pipeline as pp
import graph_results as gr
import importlib
importlib.reload(pp)
importlib.reload(gr)
import pickle
from multiprocessing import Pool
from torch.optim.lr_scheduler import ExponentialLR

EPOCHS = 100
BATCH_SIZE = 2000
RUNS = 500
DATASET = 'Weather'
METRIC_TEST = 'R2'
LEARNING_RATE = 5e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##FF model
class Feedforward(torch.nn.Module):
        def __init__(self, input_size):
                super(Feedforward, self).__init__()
                self.input_size = input_size
                self.hidden_size  = [300,150,30]
                self.fc = nn.Sequential(
                        nn.Linear(self.input_size, self.hidden_size[0]),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(self.hidden_size[0], self.hidden_size[1]),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(self.hidden_size[1], (self.hidden_size[2])),
                        nn.ReLU(),
                        nn.Linear(self.hidden_size[2], 1)
                )
                for layer in self.fc:
                        if isinstance(layer, nn.Linear):
                                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                                nn.init.constant_(layer.bias, 0)
        def forward(self, x):
                output = self.fc(x)
                return output
        
def createModel():
    model = Feedforward(123)
    model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
    lr_scheduler = ExponentialLR(optimizer, gamma=0.9)
    return model, criterion, optimizer, lr_scheduler


def loadData(dataset, cost):
    ##load data
    X = pd.read_csv(f'{ROOT_DIR}/data/{DATASET}/data_{dataset}_{cost:.2f}.csv', sep = ' ', names = [i for i in range(124)])
    X = X.sample(n=1000)
    ##get X and label
    y = X.iloc[:,-1]
    X = X.iloc[:,:-1]
    return X.values, y.values

def run_model_for_cost(inputs):
    c, loadData, DATASET, METRIC_TEST, BATCH_SIZE, EPOCHS, DEVICE, RUNS = inputs
    mp = pp.ModelPipeline(c, loadData, DATASET, METRIC_TEST, BATCH_SIZE, EPOCHS, DEVICE, RUNS)
    mp.set_functions(createModel())
    return mp.run_model_for_cost()


def main():
     ##run model on datasets
    cpu = int(os.environ.get('SLURM_CPUS_PER_TASK', 5))
    costs = [0.11, 0.19, 0.30, 0.40, 0.48]
    inputs = [(c, loadData, DATASET, METRIC_TEST, BATCH_SIZE, EPOCHS, DEVICE, RUNS) for c in costs]
    with Pool(cpu) as pool:
        results = pool.map(run_model_for_cost, inputs)

    losses = {}
    metrics_all = pd.DataFrame()
    for c, loss, metrics in results:
        losses[c] = loss
        metrics_all = pd.concat([metrics_all, metrics], axis=0)
    metrics_all.reset_index(inplace = True, drop = True)
    losses_df, test_losses_df = pp.loss_dictionary_to_dataframe(losses, costs, RUNS)
    
    #REMOVE INSTANCES OF CATASTROPHIC LEARNING. WEATHER ONLY
    for col in metrics_all.columns:
        mask = metrics_all[col] < 0
        metrics_all.loc[mask, col] = np.nan
        test_losses_df.loc[mask, col] = np.nan
    
    ##Save results
    path_save = f'{ROOT_DIR}/results/{DATASET}'
    cost = f'{costs[0]}-{costs[-1]}'
    metrics_all.to_csv(f'{path_save}/{METRIC_TEST}_{cost}.csv', index=False)
    test_losses_df.to_csv(f'{path_save}/losses_{cost}.csv', index=False)
    with open(f'{path_save}/losses.pkl', 'wb') as f:
        pickle.dump(losses_df, f)

    
    ##Save graph
    save = True
    gr.grapher(DATASET, metrics_all, METRIC_TEST, cost, save)
    gr.grapher(DATASET, test_losses_df, 'Loss', cost, save)
    gr.grapher_losses(DATASET, losses_df, costs, save)

if __name__ == '__main__':
    main()