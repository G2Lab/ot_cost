import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


global ROOT_DIR
ROOT_DIR = '/gpfs/commons/groups/gursoy_lab/aelhussein/ot_cost/otcost_fl_rebase'
SAVE_DIR = f'{ROOT_DIR}/results'

def grapher(dataset, df, metric, costs, save = False):
    plt.figure()
    if (metric == 'Loss'):
        if dataset == 'Weather':
            df_change = 100 * (np.log(df[['single']].values) - np.log(df[['joint', 'transfer', 'federated', 'pfedme']])) / np.log(df[['single']].values)
        else:
            df_change = 100 * (df[['single']].values - df[['joint', 'transfer', 'federated', 'pfedme']] ) / df[['single']].values
    else:
        df_change = 100 * (df[['joint', 'transfer', 'federated', 'pfedme']] - df[['single']].values) / df[['single']].values
    df_change['cost'] = df['cost']
    sns.lineplot( x= df_change['cost'], y =df_change['joint'], alpha = 0.8, marker = 'o', label = 'Joint')
    sns.lineplot( x= df_change['cost'], y =df_change['transfer'], alpha = 0.8, marker = 'o', label = 'Transfer')
    sns.lineplot( x= df_change['cost'], y =df_change['federated'], alpha = 0.8, marker = 'o', label = 'Federated')
    sns.lineplot( x= df_change['cost'], y =df_change['pfedme'], alpha = 0.8, marker = 'o', label = 'pFedMe')
    plt.axhline(y=0, color='black', linestyle = '--', alpha = 0.5, label = 'Baseline')
    plt.xlabel('OT_cost', fontsize = 14)
    plt.ylabel(f'% Change in {metric}', fontsize = 14)
    plt.legend(fontsize = 14)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    if save == True:
        plt.savefig(f'{SAVE_DIR}/{dataset}/{metric}_{costs}_change.pdf')
    plt.show()
    return


def grapher_nIID(dataset, df, metric, costs, save = False):
    plt.figure()
    if (metric == 'Loss'):
        df_change = 100 * (df[['single']].values - df[['federated', 'personalised_fed']]) / df[['single']].values
    else:
        df_change = 100 * (df[['federated', 'personalised_fed']] - df[['single']].values) / df[['single']].values
    df_change['cost'] = df['cost']
    sns.lineplot( x= df_change['cost'], y =df_change['federated'], alpha = 0.8, marker = 'o', label = 'FedAvg')
    sns.lineplot( x= df_change['cost'], y =df_change['personalised_fed'], alpha = 0.8, marker = 'o', label = 'pFedMe')
    plt.axhline(y=0, color='black', linestyle = '--', alpha = 0.5, label = 'Baseline')
    plt.xlabel('OT_cost', fontsize = 14)
    plt.ylabel(f'% Change in {metric}', fontsize = 14)
    plt.legend(fontsize = 14)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    if save == True:
        plt.savefig(f'{SAVE_DIR}/{dataset}/niid_{metric}_{costs}_change.pdf')
    plt.show()
    return

def grapher_losses(dataset, losses, costs, save=False):
    colors = sns.color_palette('tab10', n_colors=len(costs))
    for arch in losses.keys():
        df = losses[arch]
        plt.figure(figsize=(10, 6))
        for idx, cost in enumerate(costs):
            color = colors[idx]
            for split in ['val']:
                alpha_val = 0.4 if split == 'train' else 0.8
                df_plot = df[cost][split].melt(var_name='epoch', value_name='loss')
                sns.lineplot(x='epoch', y='loss', data=df_plot, color=color, alpha=alpha_val, label=f'{cost}_{split}')
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.title(f'{arch} training', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=14, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        if save:
            plt.savefig(f'{SAVE_DIR}/{dataset}/loss_{arch}.pdf', bbox_inches='tight')
        plt.show()
    return