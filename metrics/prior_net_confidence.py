from __future__ import print_function, division

import sys
import os
import numpy as np
from numpy import ma
import scipy.stats
from scipy.stats import loggamma, gamma, digamma
import math
import matplotlib
import argparse
import matplotlib
import seaborn as sns

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm
import matplotlib.colors
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches

from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

parser = argparse.ArgumentParser(description='Plot useful graphs for evaluation.')
parser.add_argument('model_parent_dir', type=str, help='Path to model directory')
parser.add_argument('--save_dir', type=str, default='.',
                    help='Path to directory where to save the plots')
parser.add_argument('--unseen_eval_dir', type=str, default='eval_linsk_ALL')
parser.add_argument('--seen_eval_dir', type=str, default='eval4_CDE')

matplotlib.rcParams['savefig.dpi'] = 200


def calc_diff_entropy(logits):
    alpha_1, alpha_2 = np.exp(logits[:, 0]), np.exp(logits[:, 1])
    alpha_0 = alpha_1 + alpha_2
    # with np.errstate(divide='ignore', invalid='ignore'):
    diff_entropy = loggamma(alpha_1) + loggamma(alpha_2) - np.log(gamma(alpha_0)) - (alpha_1 - 1) * (
        digamma(alpha_1) - digamma(alpha_0)) - (alpha_2 - 1) * (digamma(alpha_2) - digamma(alpha_0))
        # diff_entropy[~ np.isfinite(diff_entropy)] = 0.

    return diff_entropy


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


def plot_auc_vs_percentage_included(labels, predictions, sort_by_array, resolution=100, sort_by_name='std'):
    """
    Plot the ROC AUC score vs. the percentage of examples included where the examples are sorted by the array
    sort_by_array. This array could for instance represent the spread of the ensemble predictions, and hence the
    curve would show the performance on the subset of the examples given by thresholding the spread.
    :param labels: target label array
    :param predictions: label probabilities as predicted by the model
    :param sort_by_array: array of values to use as keys for sorting
    :param resolution: Number of points to plot
    :return:
    """
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

        # print(len(labels_subset), len(predictions_subset))
        # print(labels_subset[max(0, last_idx-5): last_idx])
        try:
            roc_auc_scores[i] = roc_auc_score(labels_subset, predictions_subset)
        except ValueError:
            roc_auc_scores[i] = np.nan

    plt.plot(proportions_included, roc_auc_scores, color=(.2, .2, .6))
    plt.xlabel("Percentage examples included as sorted by " + sort_by_name + " of Prior Net output.")
    plt.ylabel("ROC AUC score on the subset examples included")
    return


def main(args):
    labels_seen, logits_seen, preds_seen = get_labels_logits_predictions(os.path.join(args.model_parent_dir, args.seen_eval_dir))
    labels_unseen, logits_unseen, preds_unseen = get_labels_logits_predictions(os.path.join(args.model_parent_dir, args.unseen_eval_dir))

    diff_entropy_seen = calc_diff_entropy(logits_seen)
    diff_entropy_unseen = calc_diff_entropy(logits_unseen)


    #   AUC vs. CUMULATIVE INCLUDED
    # Make AUC vs. cumulative samples included
    plot_auc_vs_percentage_included(labels_seen, preds_seen, diff_entropy_seen, resolution=200, sort_by_name='diff. entropy')
    plt.savefig(os.path.join(args.save_dir, 'auc_vs_cumulative_samples_included_diff_entropy_seen.png'), bbox_inches='tight')
    plt.clf()

    plot_auc_vs_percentage_included(labels_unseen, preds_unseen, diff_entropy_unseen, resolution=200, sort_by_name='diff. entropy')
    plt.savefig(os.path.join(args.save_dir, 'auc_vs_cumulative_samples_included_diff_entropy_unseen.png'), bbox_inches='tight')
    plt.clf()

    print("Mean Diff. entropy seen: {}, unseen: {}".format(np.mean(diff_entropy_seen), np.mean(diff_entropy_unseen)))

    # mean_target_deviation = np.abs(labels - avg_predictions)
    # # print("mean_target_deviation\n", mean_target_deviation.shape, "\n", mean_target_deviation[:5])
    #
    # correct = mean_target_deviation < 0.5
    # incorrect = np.invert(correct)
    #
    # # Calculate the true_positives, true_negatives .. e.t.c.
    # # Define POSITIVE as OFF TOPIC
    # tp = np.logical_and(correct, avg_predictions < 0.5)
    # tn = np.logical_and(correct, avg_predictions >= 0.5)
    # fp = np.logical_and(incorrect, avg_predictions < 0.5)
    # fn = np.logical_and(incorrect, avg_predictions >= 0.5)
    #
    # print("Metrics calculated")
    #
    # # Make the plots:
    # savedir = args.savedir
    #
    # #    RATIOS PLOTS
    # # Make the std ratios plots
    # plot_ratio_bar_chart(correct, incorrect, std_spread, n_bins=40, y_lim=[0.0, 1.0])
    # plt.xlabel("Spread (std of ensemble predictions)")
    # plt.ylabel("Ratio correct to incorrect predictions (thresh = 0.5)")
    # plt.savefig(savedir + '/ratios_std_spread_histogram.png', bbox_inches='tight')
    # plt.clf()



if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
