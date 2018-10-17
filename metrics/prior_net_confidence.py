from __future__ import print_function, division

import sys
import os
import numpy as np
from numpy import ma
import scipy.stats
from scipy.special import digamma, gammaln
import math
import matplotlib
import argparse
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm
import matplotlib.colors
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches

from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

sns.set(style='whitegrid')

parser = argparse.ArgumentParser(description='Plot useful graphs for evaluation.')
parser.add_argument('models_parent_dir', type=str, help='Path to directory with models')
parser.add_argument('--model_base_name', type=str, default='atm_prior_net_stats')
parser.add_argument('--save_dir', type=str, default='.',
                    help='Path to directory where to save the plots')
parser.add_argument('--unseen_eval_dir', type=str, default='eval_linsk_ALL')
parser.add_argument('--seen_eval_dir', type=str, default='eval4_CDE')
parser.add_argument('--which_single_model', type=int, default=1,
                    help='For the plots that only use a single model, which model to use.')
parser.add_argument('--num_trained_models', type=int, default=10)

matplotlib.rcParams['savefig.dpi'] = 200


class ModelEvaluationStats(object):
    def __init__(self, model_dir_path, eval_dir='eval4_CDE'):
        self.model_dir_path = model_dir_path
        self.eval_dir = eval_dir
        self.eval_dir_path = os.path.join(model_dir_path, eval_dir)

        # Get the evaluation outputs
        self.labels, self.logits, self.preds = get_labels_logits_predictions(self.eval_dir_path)
        self.alphas = np.exp(self.logits)

        # Calculate the measures of uncertainty
        self.diff_entropy = calc_dirich_diff_entropy(self.alphas)
        self.mutual_info = calc_dirich_mutual_info(self.alphas)

        self.size = len(self.labels)

    def __len__(self):
        return self.size


def calc_dirich_diff_entropy(alphas):
    # alpha_1, alpha_2 = np.exp(logits[:, 0]), np.exp(logits[:, 1])
    # alpha_0 = alpha_1 + alpha_2
    # # with np.errstate(divide='ignore', invalid='ignore'):
    # diff_entropy = loggamma(alpha_1) + loggamma(alpha_2) - np.log(gamma(alpha_0)) - (alpha_1 - 1) * (
    #     digamma(alpha_1) - digamma(alpha_0)) - (alpha_2 - 1) * (digamma(alpha_2) - digamma(alpha_0))

    alpha_0 = np.sum(alphas, axis=1, keepdims=True)
    diff_entropy = np.sum(gammaln(alphas), axis=1) - np.squeeze(gammaln(alpha_0)) - np.sum(
        (alphas - 1.) * (digamma(alphas) - digamma(alpha_0)), axis=1)
    return diff_entropy


def calc_dirich_mutual_info(alphas):
    alpha_0 = np.sum(alphas, axis=1, keepdims=True)
    class_probs = alphas / alpha_0
    mutual_info = -1 * np.sum(class_probs * (np.log(class_probs) - digamma(alphas + 1.) + digamma(alpha_0 + 1.)),
                              axis=1)

    return mutual_info


def get_labels_logits_predictions(eval_dir):
    labels = []
    with open(os.path.join(eval_dir, 'labels.txt'), "r") as file:
        for line in file.readlines():
            single_example = line.strip()
            label = float(single_example)
            labels.append(label)

    logits = []
    with open(os.path.join(eval_dir, 'logits.txt'), "r") as file:
        for line in file.readlines():
            single_example = line.strip().split()
            logit = map(lambda x: float(x), single_example)
            logits.append(logit)
    predictions = []
    with open(os.path.join(eval_dir, 'predictions.txt'), "r") as file:
        for line in file.readlines():
            single_example = line.strip()
            prediction = float(single_example)
            predictions.append(prediction)
    labels_array = np.array(labels, dtype=np.float32)
    logits_array = np.array(logits, dtype=np.float32)
    preds_array = np.array(predictions, dtype=np.float32)
    return labels_array, logits_array, preds_array


def plot_auc_vs_percentage_included_single(labels, predictions, sort_by_array, resolution=100, color='black',
                                           sort_by_name='std'):
    """
    Plot the ROC AUC score vs. the percentage of examples included for a single model
    where the examples are sorted by the array
    sort_by_array. This array could for instance represent the spread of the ensemble predictions, and hence the
    curve would show the performance on the subset of the examples given by thresholding the spread.
    :param labels: target label array
    :param predictions: label probabilities as predicted by the model
    :param sort_by_array: array of values to use as keys for sorting
    :param resolution: Number of points to plot
    :return:
    """
    proportions_included, roc_auc_scores, _ = _get_cum_roc_auc_with_sort(labels, predictions, sort_by_array,
                                                                         resolution=resolution)

    plt.plot(proportions_included, roc_auc_scores, color=color)
    plt.xlabel("Percentage examples included as sorted by " + sort_by_name + " of Prior Net output.")
    plt.ylabel("ROC AUC score on the subset examples included")
    return


def _get_cum_roc_auc_with_sort(labels, predictions, sort_by_array, resolution=100):
    num_examples = len(labels)

    sorted_order = np.argsort(sort_by_array)

    labels_sorted = labels[sorted_order]
    predictions_sorted = predictions[sorted_order]

    proportions_included = np.linspace(0, 1, num=resolution)
    roc_auc_scores = np.zeros_like(proportions_included)
    for i in range(resolution):
        proportion = proportions_included[i]
        last_idx = int(math.floor(num_examples * proportion)) + 1
        labels_subset = labels_sorted[:last_idx]
        predictions_subset = predictions_sorted[:last_idx]

        try:
            roc_auc_scores[i] = roc_auc_score(labels_subset, predictions_subset)
        except ValueError:
            roc_auc_scores[i] = np.nan
    return proportions_included, roc_auc_scores, sorted_order


def plot_auc_vs_percentage_included_with_proportions(labels_seen, labels_unseen, predictions_seen, predictions_unseen,
                                    sort_by_array_seen, sort_by_array_unseen, resolution=100, sort_by_name='std',
                                    bg_alpha=0.25):
    """
    Plot the ROC AUC score vs. the percentage of examples included where the examples are sorted by the array
    sort_by_array. This array could for instance represent the spread of the ensemble predictions, and hence the
    curve would show the performance on the subset of the examples given by thresholding the spread.
    """
    labels = np.hstack((labels_seen, labels_unseen))
    predictions = np.hstack((predictions_seen, predictions_unseen))
    sort_by_array = np.hstack((sort_by_array_seen, sort_by_array_unseen))

    num_examples = len(labels)

    # Array with ones if the example is seen-seen and zero otherwise
    is_seen = np.hstack((np.ones_like(labels_seen), np.zeros_like(labels_unseen)))

    proportions_included, roc_auc_scores, sorted_order = _get_cum_roc_auc_with_sort(labels, predictions, sort_by_array,
                                                                         resolution=resolution)

    is_seen_sorted = is_seen[sorted_order]
    percentage_seen = np.cumsum(is_seen_sorted, dtype=np.float32) / np.arange(num_examples, dtype=np.float32)

    # Plot the ROC_AUC vs. proportions included
    plt.plot(proportions_included, roc_auc_scores)

    # Plot the proportions included
    x = np.arange(num_examples, dtype=np.float32) / num_examples
    clrs = sns.color_palette("husl", 2)
    plt.fill_between(x, 0., percentage_seen, facecolor=clrs[0], alpha=bg_alpha)
    plt.fill_between(x, percentage_seen, 1., facecolor=clrs[1], alpha=bg_alpha)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Percentage examples included as sorted by " + sort_by_name + " of Prior Net output.")
    # plt.ylabel("ROC AUC score on the subset examples included")
    return


def main(args):
    # Create save directory if doesn't exist:
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    # Load the data for all the N models trained (for plots where variation across students is shown)
    ensemble_models_paths = [os.path.join(args.models_parent_dir, args.model_base_name + str(i + 1)) for i in
                             range(args.num_trained_models)]
    all_evaluation_stats_seen = [ModelEvaluationStats(model_path, eval_dir=args.seen_eval_dir) for model_path in
                                 ensemble_models_paths]
    all_evaluation_stats_unseen = [ModelEvaluationStats(model_path, eval_dir=args.unseen_eval_dir) for model_path in
                                   ensemble_models_paths]


    # Load the data for a single model of choice (normally the best model on the seen-seen data)
    single_evaluation_stats_seen = all_evaluation_stats_seen[args.which_single_model - 1]
    single_evaluation_stats_unseen = all_evaluation_stats_unseen[args.which_single_model - 1]


    # Make AUC vs. cumulative samples included for N models - DIFF. ENTROPY
    # Seen-seen
    for i in range(args.num_trained_models):
        eval_stats = all_evaluation_stats_seen[i]
        plot_auc_vs_percentage_included_single(eval_stats.labels, eval_stats.preds, eval_stats.diff_entropy,
                                               resolution=200, sort_by_name='diff. entropy')
    plt.savefig(
        os.path.join(args.save_dir, 'auc_vs_cum_samples_incl_diff_entropy_seen_{}.png'.format(args.model_base_name)),
        bbox_inches='tight')
    plt.clf()

    # unseen-unseen
    for i in range(args.num_trained_models):
        eval_stats = all_evaluation_stats_unseen[i]
        plot_auc_vs_percentage_included_single(eval_stats.labels, eval_stats.preds, eval_stats.diff_entropy,
                                               resolution=200, sort_by_name='diff. entropy')
    plt.savefig(
        os.path.join(args.save_dir, 'auc_vs_cum_samples_incl_diff_entropy_unseen_{}.png'.format(args.model_base_name)),
        bbox_inches='tight')
    plt.clf()

    # Make AUC vs. cumulative samples included for N models - MUTUAL INFORMATION
    # Seen-seen
    for i in range(args.num_trained_models):
        eval_stats = all_evaluation_stats_seen[i]
        plot_auc_vs_percentage_included_single(eval_stats.labels, eval_stats.preds, eval_stats.mutual_info,
                                               resolution=200, sort_by_name='mutual information')
    plt.savefig(
        os.path.join(args.save_dir, 'auc_vs_cum_samples_incl_mutual_info_seen_{}.png'.format(args.model_base_name)),
        bbox_inches='tight')
    plt.clf()

    # unseen-unseen
    for i in range(args.num_trained_models):
        eval_stats = all_evaluation_stats_unseen[i]
        plot_auc_vs_percentage_included_single(eval_stats.labels, eval_stats.preds, eval_stats.mutual_info,
                                               resolution=200, sort_by_name='mutual information')
    plt.savefig(
        os.path.join(args.save_dir, 'auc_vs_cum_samples_incl_mutual_info_unseen_{}.png'.format(args.model_base_name)),
        bbox_inches='tight')
    plt.clf()

    mean_diff_entropy_seen = reduce(lambda x, y: x + y, map(lambda x: np.mean(x.diff_entropy),
                                                            all_evaluation_stats_seen)) / args.num_trained_models
    mean_diff_entropy_unseen = reduce(lambda x, y: x + y, map(lambda x: np.mean(x.diff_entropy),
                                                              all_evaluation_stats_unseen)) / args.num_trained_models
    mean_mutual_info_seen = reduce(lambda x, y: x + y, map(lambda x: np.mean(x.mutual_info),
                                                           all_evaluation_stats_seen)) / args.num_trained_models
    mean_mutual_info_unseen = reduce(lambda x, y: x + y, map(lambda x: np.mean(x.mutual_info),
                                                             all_evaluation_stats_unseen)) / args.num_trained_models
    print("Mean Diff. entropy seen: {}, unseen: {}".format(mean_diff_entropy_seen, mean_diff_entropy_unseen))
    print("Mean mutual information seen: {}, unseen: {}".format(mean_mutual_info_seen, mean_mutual_info_unseen))
    # #   AUC seen and unseen combined vs. CUMULATIVE INCLUDED
    # plt.savefig(os.path.join(args.save_dir, 'auc_vs_cumulative_samples_included_diff_entropy_combined.png'),
    #             bbox_inches='tight')



if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
