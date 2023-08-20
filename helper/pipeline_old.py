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
global ARCHITECTURES
ARCHITECTURES = ['single', 'joint', 'transfer', 'federated', 'pfedme']
CATEGORICAL_CLASSES = ['EMNIST', 'CIFAR']


def saveLosses(losses, train_type, pipeline):
    losses[train_type]['train_losses'].append(pipeline.train_losses)
    losses[train_type]['val_losses'].append(pipeline.val_losses)
    losses[train_type]['test_losses'].append(pipeline.test_losses)
    return losses
    

def runModels(loadDataFunc, createModelFunc, c, losses, DATASET, METRIC_TEST, BATCH_SIZE, EPOCHS):
    X1, y1 = loadDataFunc(1, c)
    X2, y2 = loadDataFunc(2, c)
    if DATASET in CATEGORICAL_CLASSES:
        y1, y2 = remap_categoricals(y1, y2)
        SINGLE_CLASS = len(set(list(y1)))
        MSL_CLASS = len(set(list(y1) + list(y2)))

    #Single
    ##Get data and models
    SINGLE = True
    model, criterion, optimizer, lr_scheduler = createModelFunc()
    dataloader = dp.DataPreprocessor(DATASET, BATCH_SIZE)
    train_loader, val_loader, test_loader = dataloader.preprocess(X1, y1)
    run_pipeline = tr.ModelTrainer(DATASET, model, optimizer, criterion, lr_scheduler, DEVICE)
    
    ##Run until epochs or eaerly stopping
    epoch = 0
    while (not run_pipeline.stopping) & (epoch < EPOCHS):
        run_pipeline.fit(train_loader, val_loader)
        epoch += 1
    single_test_metrics = run_pipeline.test(test_loader, metric_name=METRIC_TEST)
    losses = saveLosses(losses, 'single', run_pipeline)

    SINGLE = False
    #Joint
    model, criterion, optimizer, lr_scheduler = createModelFunc()
    dataloader = dp.DataPreprocessor(DATASET, BATCH_SIZE)
    train_loader_1, val_loader_1, test_loader_1 = dataloader.preprocess(X1, y1)
    train_loader_2, val_loader_2, test_loader_2 = dataloader.preprocess(X2, y2)
    #this keeps the test set unique to each dataset
    train_loader, val_loader = dataloader.combine(train_loader_1, train_loader_2, val_loader_1, val_loader_2)
    run_pipeline = tr.ModelTrainer(DATASET, model, optimizer, criterion, lr_scheduler, DEVICE)
    ##Run until epochs or eaerly stopping
    epoch = 0
    while (not run_pipeline.stopping) & (epoch < EPOCHS):
        run_pipeline.fit(train_loader, val_loader)
        epoch += 1
    joint_test_metrics = run_pipeline.test(test_loader, metric_name=METRIC_TEST)
    losses = saveLosses(losses, 'joint', run_pipeline)

    #Transfer
    model, criterion, optimizer, lr_scheduler = createModelFunc()
    dataloader = dp.DataPreprocessor(DATASET, BATCH_SIZE)
    train_loader_target, val_loader_target, test_loader_target = dataloader.preprocess(X1, y1)
    train_loader_source, val_loader_source, test_loader_source = dataloader.preprocess(X2, y2)
    transfer_run_pipeline = tr.TransferModelTrainer(DATASET, model, optimizer, criterion, lr_scheduler, DEVICE)
    
    ##Run until epochs or eaerly stopping
    epoch = 0
    while (not transfer_run_pipeline.stopping) & (epoch < EPOCHS):
        transfer_run_pipeline.fit(train_loader_target, train_loader_source, val_loader_target, val_loader_source)
        epoch +=1
    transfer_test_metrics = transfer_run_pipeline.test(test_loader_target, metric_name=METRIC_TEST)
    losses = saveLosses(losses, 'transfer', transfer_run_pipeline)


    #federated
    model, criterion, optimizer, lr_scheduler = createModelFunc()
    dataloader = dp.DataPreprocessor(DATASET, BATCH_SIZE)
    train_loader_1, val_loader_1, test_loader_1 = dataloader.preprocess(X1, y1)
    train_loader_2, val_loader_2, test_loader_2 = dataloader.preprocess(X2, y2)
    federated_run_pipeline = tr.FederatedModelTrainer(DATASET, model, optimizer, criterion, lr_scheduler, DEVICE, train_loader_1, train_loader_2)

    ##Run until epochs or eaerly stopping
    epoch = 0
    while (not federated_run_pipeline.stopping) & (epoch < EPOCHS // tr.FED_EPOCH):
        federated_run_pipeline.fit(train_loader_1, train_loader_2, val_loader_1, val_loader_2)
        epoch +=1
    fed_test_metrics = federated_run_pipeline.test(test_loader_1, metric_name=METRIC_TEST)
    losses = saveLosses(losses, 'federated', federated_run_pipeline)


    #pfedme
    model, criterion, optimizer, lr_scheduler = createModelFunc()
    dataloader = dp.DataPreprocessor(DATASET, BATCH_SIZE)
    train_loader_1, val_loader_1, test_loader_1 = dataloader.preprocess(X1, y1)
    train_loader_2, val_loader_2, test_loader_2 = dataloader.preprocess(X2, y2)
    federated_run_pipeline = tr.FederatedModelTrainer(DATASET, model, optimizer, criterion,lr_scheduler, DEVICE, train_loader_1, train_loader_2, pfedme = True, pfedme_reg =1e-1)
    ##Run until epochs or eaerly stopping
    epoch = 0
    while (not federated_run_pipeline.stopping) & (epoch < EPOCHS // tr.FED_EPOCH):
        federated_run_pipeline.fit(train_loader_1, train_loader_2, val_loader_1, val_loader_2)
        epoch +=1
    pfedme_test_metrics = federated_run_pipeline.test(test_loader_1, metric_name=METRIC_TEST)
    losses = saveLosses(losses, 'pfedme', federated_run_pipeline)

    metrics = [single_test_metrics, joint_test_metrics, transfer_test_metrics, fed_test_metrics, pfedme_test_metrics]
    metrics_df = pd.DataFrame(metrics, index=['single', 'joint','transfer','federated', 'pfedme']).T
    metrics_df['cost'] = c

    return metrics_df, losses

def remap_categoricals(y1, y2):
    unique_y1 = sorted(set(y1))
    mapping = {label: idx for idx, label in enumerate(unique_y1)}
    X = len(unique_y1)
    unique_y2 = sorted(set(y2) - set(unique_y1))
    mapping.update({label: X + idx for idx, label in enumerate(unique_y2)})
    y1_mapped = np.array([mapping[label] for label in y1])
    y2_mapped = np.array([mapping[label] for label in y2])
    return y1_mapped, y2_mapped

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
    cost_values = []
    for architecture in ARCHITECTURES:
        test_losses_df[architecture] = []
        for c in costs:
            loss = losses[c][architecture]['test_losses']
            loss = [l[0] for l in loss]
            test_losses_df[architecture].extend(loss)
    test_losses_df = pd.DataFrame.from_dict(test_losses_df, orient = 'index').T
    test_losses_df['cost'] =  [item for item in costs for _ in range(RUNS)]
    return losses_df, test_losses_df


def run_model_for_cost(inputs):
    c, loadDataFunc, createModelFunc, DATASET, METRIC_TEST, BATCH_SIZE, EPOCHS, RUNS = inputs
    losses_for_c = {}
    for architecture in ARCHITECTURES:
        losses_for_c[architecture] =  {'train_losses': [], 'val_losses': [], 'test_losses': []}
    
    metrics_for_c = pd.DataFrame()
    for _ in range(RUNS):
        metrics_run, losses_for_c = runModels(loadDataFunc, createModelFunc, c, losses_for_c, DATASET, METRIC_TEST, BATCH_SIZE, EPOCHS)
        metrics_for_c = pd.concat([metrics_for_c, metrics_run], axis=0)
    return c, losses_for_c, metrics_for_c
