global ROOT_DIR
ROOT_DIR = '/gpfs/commons/groups/gursoy_lab/aelhussein/ot_cost/otcost_fl_rebase'

import pandas as pd
import torch
import torch.nn as nn
import sys
import numpy as np
sys.path.append(f'{ROOT_DIR}/code/helper')
import data_preprocessing as dp
import trainers as tr
import importlib
importlib.reload(dp)
importlib.reload(tr)
import warnings
# Suppress the specific LR warning that is non-issue
warnings.filterwarnings("ignore", "Seems like `optimizer.step()` has been overridden after learning rate scheduler initialization.")


global DEVICE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ARCHITECTURES = ['single', 'joint', 'transfer', 'federated', 'pfedme']

class ModelPipeline:
    def __init__(self, c, loadDataFunc, DATASET, METRIC_TEST, BATCH_SIZE, EPOCHS, DEVICE, RUNS):
        self.ARCHITECTURES = ['single', 'joint', 'transfer', 'federated', 'pfedme']
        self.CATEGORICAL_CLASSES = ['EMNIST', 'CIFAR']
        self.c = c
        self.loadDataFunc = loadDataFunc
        self.RUNS = RUNS
        self.DATASET = DATASET
        self.METRIC_TEST = METRIC_TEST
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCHS = EPOCHS
        self.DEVICE = DEVICE
        self.SINGLE = False
        self.SINGLE_CLASS = None
        self.MSL_CLASS = None

        _, y1 = self.loadDataFunc(1, self.c)
        _, y2 = self.loadDataFunc(2, self.c)
        if self.DATASET in self.CATEGORICAL_CLASSES:
            y1, y2 = self.remap_categoricals(y1, y2)
            self.SINGLE_CLASS = len(set(list(y1)))
            self.MSL_CLASS = len(set(list(y1) + list(y2)))

    def load(self):
        X1, y1 = self.loadDataFunc(1, self.c)
        X2, y2 = self.loadDataFunc(2, self.c)
        if self.DATASET in self.CATEGORICAL_CLASSES:
            y1, y2 = self.remap_categoricals(y1, y2)
            self.SINGLE_CLASS = len(set(list(y1)))
            self.MSL_CLASS = len(set(list(y1) + list(y2)))
        return X1, y1, X2, y2

    def set_functions(self, createModelFunc):
        self.createModelFunc = createModelFunc


    def saveLosses(self, train_type, pipeline):
        self.losses_for_c[train_type]['train_losses'].append(pipeline.train_losses)
        self.losses_for_c[train_type]['val_losses'].append(pipeline.val_losses)
        self.losses_for_c[train_type]['test_losses'].append(pipeline.test_losses)
        return 

    def runModels(self):
        X1, y1, X2, y2 = self.load()
        self.single(X1, y1)
        self.joint(X1, y1, X2, y2)
        self.transfer(X1, y1, X2, y2)
        self.federated(X1, y1, X2, y2)
        self.federated(X1, y1, X2, y2, pfedme = True)
        self.maml(X1, y1, X2, y2)

        metrics = [self.single_test_metrics, self.joint_test_metrics, self.transfer_test_metrics, self.fedavg_test_metrics, self.pfedme_test_metrics]
        metrics_df = pd.DataFrame(metrics, index=ARCHITECTURES).T
        metrics_df['cost'] = self.c
        return metrics_df
        
    def remap_categoricals(self, y1, y2):
        unique_y1 = sorted(set(y1))
        mapping = {label: idx for idx, label in enumerate(unique_y1)}
        X = len(unique_y1)
        unique_y2 = sorted(set(y2) - set(unique_y1))
        mapping.update({label: X + idx for idx, label in enumerate(unique_y2)})
        y1_mapped = np.array([mapping[label] for label in y1])
        y2_mapped = np.array([mapping[label] for label in y2])
        return y1_mapped, y2_mapped
    

    def single(self, X1, y1):
        self.SINGLE = True
        model, criterion, optimizer, lr_scheduler = self.createModelFunc
        dataloader = dp.DataPreprocessor(self.DATASET, self.BATCH_SIZE)
        train_loader, val_loader, test_loader = dataloader.preprocess(X1, y1)
        run_pipeline = tr.ModelTrainer(self.DATASET, model, optimizer, criterion, lr_scheduler, DEVICE)
        
        epoch = 0
        while (not run_pipeline.stopping) & (epoch < self.EPOCHS):
            run_pipeline.fit(train_loader, val_loader)
            epoch += 1

        self.single_test_metrics = run_pipeline.test(test_loader, metric_name=self.METRIC_TEST)
        self.saveLosses('single', run_pipeline)
        return 

    def joint(self, X1, y1, X2, y2):
        self.SINGLE = False
        model, criterion, optimizer, lr_scheduler = self.createModelFunc
        dataloader = dp.DataPreprocessor(self.DATASET, self.BATCH_SIZE)
        train_loader, val_loader, test_loader = dataloader.preprocess_joint(X1, y1, X2, y2)
        run_pipeline = tr.ModelTrainer(self.DATASET, model, optimizer, criterion, lr_scheduler, self.DEVICE)

        epoch = 0
        while (not run_pipeline.stopping) & (epoch < self.EPOCHS):
            run_pipeline.fit(train_loader, val_loader)
            epoch += 1
        self.joint_test_metrics = run_pipeline.test(test_loader, metric_name=self.METRIC_TEST)
        self.saveLosses('joint', run_pipeline)
        return

    def transfer(self, X1, y1, X2, y2):
        self.SINGLE = False
        model, criterion, optimizer, lr_scheduler = self.createModelFunc
        dataloader = dp.DataPreprocessor(self.DATASET, self.BATCH_SIZE)
        train_loader_target, val_loader_target, test_loader_target = dataloader.preprocess(X1, y1)
        train_loader_source, val_loader_source, _ = dataloader.preprocess(X2, y2)
        transfer_run_pipeline = tr.TransferModelTrainer(self.DATASET, model, optimizer, criterion, lr_scheduler, self.DEVICE)
        
        epoch = 0
        while (not transfer_run_pipeline.stopping) & (epoch < self.EPOCHS):
            transfer_run_pipeline.fit(train_loader_target, train_loader_source, val_loader_target, val_loader_source)
            epoch +=1
        self.transfer_test_metrics = transfer_run_pipeline.test(test_loader_target, metric_name=self.METRIC_TEST)
        self.saveLosses('transfer', transfer_run_pipeline)
        return 

    def federated(self, X1, y1, X2, y2, pfedme=False, pfedme_reg=1e-1):
        self.SINGLE = False
        model, criterion, optimizer, lr_scheduler = self.createModelFunc
        dataloader = dp.DataPreprocessor(self.DATASET, self.BATCH_SIZE)
        train_loader_1, val_loader_1, test_loader_1 = dataloader.preprocess(X1, y1)
        train_loader_2, val_loader_2, _ = dataloader.preprocess(X2, y2)
        federated_run_pipeline = tr.FederatedModelTrainer(self.DATASET, model, optimizer, criterion,lr_scheduler, self.DEVICE, train_loader_1, train_loader_2, pfedme, pfedme_reg)

        epoch = 0
        while (not federated_run_pipeline.stopping) & (epoch < self.EPOCHS // tr.FED_EPOCH):
            federated_run_pipeline.fit(train_loader_1, train_loader_2, val_loader_1, val_loader_2)
            epoch +=1
        fed_test_metrics = federated_run_pipeline.test(test_loader_1, metric_name=self.METRIC_TEST)
        if not pfedme:
            self.fedavg_test_metrics = fed_test_metrics 
            self.saveLosses('federated', federated_run_pipeline)
        elif pfedme: 
            self.pfedme_test_metrics = fed_test_metrics 
            self.saveLosses('pfedme', federated_run_pipeline)
        return 
    
    def maml(self, X1, y1, X2, y2):
        self.SINGLE = False
        model, criterion, optimizer, lr_scheduler = self.createModelFunc
        dataloader = dp.DataPreprocessor(self.DATASET, self.BATCH_SIZE)
        train_loader_1, val_loader_1, test_loader_1 = dataloader.preprocess(X1, y1)
        train_loader_2, val_loader_2, _ = dataloader.preprocess(X2, y2)
        maml_run_pipeline = tr.MAMLModelTrainer(self.DATASET, model, optimizer, criterion,lr_scheduler, self.DEVICE, train_loader_1, train_loader_2)
        epoch = 0
        while (not maml_run_pipeline.stopping) & (epoch < self.EPOCHS // tr.FED_EPOCH):
            maml_run_pipeline.fit(train_loader_1, train_loader_2, val_loader_1, val_loader_2)
            epoch +=1
        maml_test_metrics = maml_run_pipeline.test(test_loader_1, metric_name=self.METRIC_TEST)
        self.maml_test_metrics = maml_test_metrics
        self.saveLosses('maml', maml_run_pipeline)
        return


    def run_model_for_cost(self):
        self.losses_for_c = {}
        for architecture in self.ARCHITECTURES:
            self.losses_for_c[architecture] = {'train_losses': [], 'val_losses': [], 'test_losses': []}
        metrics_for_c = pd.DataFrame()
        for _ in range(self.RUNS):
            metrics_run = self.runModels()
            metrics_for_c = pd.concat([metrics_for_c, metrics_run], axis=0)
        return self.c, self.losses_for_c, metrics_for_c


def loss_dictionary_to_dataframe(losses, costs, RUNS):
    ##EPOCH tracking of train and val
    losses_df = {}
    for architecture in ARCHITECTURES:
        losses_df[architecture] = {}
        for c in costs:
            losses_df[architecture][c] = {}
            loss_lists = losses[c][architecture]
            max_length = max(len(loss) for loss in loss_lists)
            padded_losses= [[loss + [np.nan] * (max_length - len(loss)) for loss in loss_lists[key]] for key in loss_lists]
            losses_df[architecture][c]['train'] = pd.DataFrame(padded_losses[0])
            losses_df[architecture][c]['val'] = pd.DataFrame(padded_losses[1])

    ##Per cost tracking of test losses (graphed in the same way as metric performance)
    test_losses_df = {}
    for architecture in ARCHITECTURES:
        test_losses_df[architecture] = []
        for c in costs:
            loss = losses[c][architecture]['test_losses']
            loss = [l[0] for l in loss]
            test_losses_df[architecture].extend(loss)
    test_losses_df = pd.DataFrame.from_dict(test_losses_df, orient = 'index').T
    test_losses_df['cost'] =  [item for item in costs for _ in range(RUNS)]
    return losses_df, test_losses_df
