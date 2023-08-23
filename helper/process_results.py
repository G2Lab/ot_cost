import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import bootstrap

global ROOT_DIR
ROOT_DIR = '/gpfs/commons/groups/gursoy_lab/aelhussein/ot_cost/otcost_fl_rebase'
SAVE_DIR = f'{ROOT_DIR}/results'
ARCHITECTURES = ['single', 'joint', 'transfer', 'federated', 'pfedme']

def load_results(DATASET, name):
    return pd.read_csv(f'{ROOT_DIR}/results/{DATASET}/{name}')

def bootstrap_ci(data, alpha=0.95):
    median = np.mean(data)
    bs_reps = bootstrap(np.array(data).reshape(1,-1), statistic=np.mean, n_resamples=1000)
    ci = bs_reps.confidence_interval[0:2]
    return round(median, 3), round(ci[0],3), round(ci[1],3)

def get_estimates(results, costs):
    results_estimates = {}
    for cost in costs:
        results_estimates[cost] ={}
        for architecture in ARCHITECTURES:
            res = results.loc[results['cost'] == cost, architecture].values
            results_estimates[cost][architecture] = bootstrap_ci(res, alpha=0.95)
    return  pd.DataFrame.from_dict(results_estimates, orient = 'index')

#BOOTSTRAP ESTIMATE DONE DIFFERENTLY
def get_difference_estimates(results, costs):
    results_diff = {}
    n_iterations = 1000
    for cost in costs:
        results_diff[cost] = {}
        for architecture in ARCHITECTURES[1:]:
            single = results.loc[results['cost'] == cost, 'single'].values
            other = results.loc[results['cost'] == cost, architecture].values
            bs_single_samples = bootstrap_samples(single, n_iterations)
            bs_other_samples = bootstrap_samples(other, n_iterations)
            bs_single_means = np.mean(bs_single_samples, axis=1)
            bs_other_means = np.mean(bs_other_samples, axis=1)
            bs_diff = bs_other_means - bs_single_means 
            mean_single = np.mean(single)
            median_diff = 100 * np.percentile(bs_diff, 50) / mean_single
            lower_ci_diff = 100 * np.percentile(bs_diff, 2.5) / mean_single
            upper_ci_diff = 100 * np.percentile(bs_diff, 97.5) / mean_single
            results_diff[cost][architecture] = (np.round(median_diff, 3), np.round(lower_ci_diff, 3), np.round(upper_ci_diff, 3))
    return pd.DataFrame.from_dict(results_diff, orient='index')

def bootstrap_samples(data, n_iterations):
    n = len(data)
    indices = np.random.randint(0, n, (n_iterations, n))
    return data[indices]


def grapher(results, DATASET, metric, costs, save = False):
    results_long = results.reset_index().rename(columns={'index': 'cost'}).melt(id_vars=['cost'], var_name='architecture')
    results_long[['median_diff', 'lower_ci_diff', 'upper_ci_diff']] = pd.DataFrame(results_long['value'].tolist(), index=results_long.index)
    results_long.drop(columns=['value'], inplace=True)
    results_long
    plt.figure()
    for architecture in results_long['architecture'].unique():
        subset = results_long[results_long['architecture'] == architecture]
        sns.lineplot(x='cost', y='median_diff', marker = 'o', data=subset, label=architecture.capitalize())
        plt.fill_between(x=subset['cost'], y1=subset['lower_ci_diff'], y2=subset['upper_ci_diff'], alpha=0.2)
    plt.axhline(y=0, color='black', linestyle = '--', alpha = 0.5, label = 'Baseline')
    plt.xlabel('Dataset Cost', fontsize = 14)
    plt.ylabel(f'% Change in {metric}', fontsize = 14)
    plt.legend(fontsize = 14)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    if save == True:
        plt.savefig(f'{SAVE_DIR}/{DATASET}/{metric}_{costs}_change.pdf', bbox_inches='tight')
    plt.show()
    return 

def process_result(DATASET, metric, costs, save):
    cost_range = f'{costs[0]}-{costs[-1]}'
    name = f'{metric}_{cost_range}.csv'
    results = load_results(DATASET, name)
    estimates = get_estimates(results, costs)
    estimates_diff = get_difference_estimates(results, costs)
    grapher(estimates_diff, DATASET, metric, cost_range, save)
    if save:
        estimates.to_csv(f'{SAVE_DIR}/{DATASET}/{metric}_{costs}_estimates.csv')
        estimates_diff.to_csv(f'{SAVE_DIR}/{DATASET}/{metric}_{costs}_estimates_change.csv')
    return

def process_results(DATASET, metric, costs, save = False):
    process_result(DATASET, metric, costs, save)
    process_result(DATASET, 'losses', costs, save)
    return

def grapher_losses(losses, DATASET, costs, save=False):
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
            plt.savefig(f'{SAVE_DIR}/{DATASET}/loss_{arch}.pdf', bbox_inches='tight')
        plt.show()
    return