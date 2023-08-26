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
import process_results as pr
import importlib
importlib.reload(pp)
importlib.reload(pr)
import pickle
from multiprocessing import Pool
from torch.optim.lr_scheduler import ExponentialLR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 200
BATCH_SIZE = 2000
RUNS = 500
DATASET = 'Credit'
METRIC_TEST = 'AUPRC'
LEARNING_RATE =  8e-3


##FF model
class Feedforward(torch.nn.Module):
        def __init__(self, input_size):
                super(Feedforward, self).__init__()
                self.input_size = input_size
                self.hidden_size  = [128,128, 56, 12]
                self.fc = nn.Sequential(
                        nn.Linear(self.input_size, self.hidden_size[0]),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(self.hidden_size[0], self.hidden_size[1]),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(self.hidden_size[1], self.hidden_size[2]),
                        nn.ReLU(),
                        nn.Linear(self.hidden_size[2], self.hidden_size[3]),
                        nn.ReLU(),
                        nn.Linear(self.hidden_size[3], 1)
                )
                for layer in self.fc:
                    if isinstance(layer, nn.Linear):
                            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                            nn.init.constant_(layer.bias, 0)


        def forward(self, x):
                output = self.fc(x)
                return output
        
def createModel():
    model = Feedforward(28)
    model.to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
    lr_scheduler = ExponentialLR(optimizer, gamma=0.8)
    return model, criterion, optimizer, lr_scheduler


def loadData(dataset, cost):
    ##load data
    X = pd.read_csv(f'{ROOT_DIR}/data/{DATASET}/data_{dataset}_{cost:.2f}.csv', sep = ' ', names = [i for i in range(29)])
    X = X.sample(150)
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
    costs = [0.12, 0.23, 0.30, 0.41]
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
    

    ##Save results
    path_save = f'{ROOT_DIR}/results/{DATASET}'
    cost = f'{costs[0]}-{costs[-1]}'
    metrics_all.to_csv(f'{path_save}/{METRIC_TEST}_{cost}_2.csv', index=False)
    test_losses_df.to_csv(f'{path_save}/losses_{cost}_2.csv', index=False)
    with open(f'{path_save}/losses_2.pkl', 'wb') as f:
        pickle.dump(losses_df, f)

    ##Process results and graph
    save = True
    pr.process_results(DATASET, METRIC_TEST, costs, save)



if __name__ == '__main__':
    main()