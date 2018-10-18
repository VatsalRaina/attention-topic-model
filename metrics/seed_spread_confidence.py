#! /usr/bin/env python
"""
Just a one-off script for comparing the seed spread versus confidence
"""
from __future__ import print_function, division
import matplotlib
matplotlib.use('Agg')
import sys
import os
import numpy as np
from numpy import ma
import scipy.stats
import math
import argparse


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
parser.add_argument('models_parent_dir', type=str, help='Path to ensemble directory')
parser.add_argument('--savedir', type=str, default='./',
                    help='Path to directory where to save the plots')
parser.add_argument('--rel_labels_path', type=str, default='eval4_naive/labels-probs.txt')

matplotlib.rcParams['savefig.dpi'] = 200
import seaborn as sns
sns.set()

# Specify the colours
green = (0.3, 0.9, 0.3)
dark_green = (0.1, 0.7, 0.3)
red = (0.9, 0.3, 0.3)
dark_orange = (0.9, 0.5, 0.2)
dark_blue = (0.1, 0.15, 0.27)


def get_ensemble_predictions(model_dirs, rel_labels_filepath='eval4_naive/labels-probs.txt'):
    """
    Get the target labels and model predictions from a txt file from all the models pointed to by list model_dirs.
    :param model_dirs: list of paths to model directories
    :param rel_labels_filepath: path to where the labels/predictions file is located within each model directory
    :return: ndarray of target labels and ndarray predictions of each model with shape [num_examples, num_models]
    """
    labels_files = map(lambda model_dir: os.path.join(model_dir, rel_labels_filepath), model_dirs)

    # Get the target labels:
    labels, _ = get_label_predictions(list(labels_files)[0])

    # List to store predictions from all the models considered
    all_predictions = []
    for labels_filepath in labels_files:
        # Get the predictions from each of the models
        _, model_predictions = get_label_predictions(labels_filepath)
        all_predictions.append(model_predictions)
    # print("shape of 1 pred", all_predictions[0].shape)  # todo:
    all_predictions = np.stack(all_predictions, axis=1)
    return labels, all_predictions


def get_label_predictions(labels_filepath):
    labels = []
    predictions = []
    with open(labels_filepath, "r") as file:
        for line in file.readlines():
            single_example = line.strip().split()
            label, prediction = map(lambda x: float(x), single_example)

            labels.append(label)
            predictions.append(prediction)
    labels_array = np.array(labels, dtype=np.float32)
    predictions_array = np.array(predictions, dtype=np.float32)
    return labels_array, predictions_array


def calc_avg_predictions(ensemble_predictions):
    avg_predictions = np.sum(ensemble_predictions, axis=1)
    avg_predictions /= ensemble_predictions.shape[1]
    return avg_predictions


def calc_entropy(predictions):
    entropy = -(predictions * np.log(predictions) + (1 - predictions) * np.log(1 - predictions))
    return entropy


def calc_mutual_information(ensemble_predictions):
    avg_predictions = calc_avg_predictions(ensemble_predictions)
    # Calculate entropy of expected distribution (also the entropy of the overall ensemble predictions)
    entropy_of_expected = calc_entropy(avg_predictions)
    # Calculate the entropy of each model
    entropy = calc_entropy(ensemble_predictions)
    # Calculate the expected entropy of each distribution
    expected_entropy = np.mean(entropy, axis=1)
    # Mutual information can be expressed as the difference between the two
    mutual_information = entropy_of_expected - expected_entropy
    return mutual_information, entropy_of_expected


def plot_spread_histogram(correct, incorrect, spread, n_bins=20, ax=None, spread_name='std'):
    if ax is None:
        ax = plt
    spread_correct = np.extract(correct, spread)
    spread_incorrect = np.extract(incorrect, spread)
    ax.hist((spread_correct, spread_incorrect), n_bins, density=True,
            histtype='bar', stacked=True, alpha=0.5)
    #ax.hist((spread_correct, spread_incorrect), n_bins, density=True, fill=False,
    #        histtype='step', stacked=True, color=['white', 'white'])
    plt.xlabel("Spread (" + spread_name + " of ensemble predictions)")
    plt.ylabel("Example Count")
    plt.legend(['On-Topic','Off-Topic'])
    return ax


def plot_ratio_bar_chart(correct, incorrect, spread, n_bins=20, ax=None, y_lim=(0., 1.0)):
    """
    Plots the ratio of correct to incorrect for each bin in the histogram, where the bins are taken over spread.
    :param correct: ndarray of bool representing which examples are correct
    :param incorrect: invertion of correct
    :param spread: spread of ensemble for each example
    :param n_bins: int
    :param ax: either an plt.ax object or None. If None, default figure will be used.
    :param y_lim: range of y values on the plot
    """
    if ax is None:
        # If ax not given, just plot on the current figure
        ax = plt
    min_x = 0.
    max_x = np.max(spread)
    assert max_x > min_x
    spread_correct = np.extract(correct, spread)
    spread_incorrect = np.extract(incorrect, spread)
    correct_binned, edges_correct = np.histogram(spread_correct, bins=n_bins, range=(min_x, max_x), density=False,
                                                 normed=False)
    incorrect_binned, edges_incorrect = np.histogram(spread_incorrect, bins=n_bins, range=(min_x, max_x), density=False,
                                                     normed=False)

    assert np.all(edges_correct == edges_incorrect)
    edges = edges_correct

    total_binned = correct_binned + incorrect_binned
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio_correct = np.divide(correct_binned.astype(np.float32), total_binned.astype(np.float32))
        ratio_correct[~ np.isfinite(ratio_correct)] = 0.

    # Construct plot points:
    plot_point_x = np.empty(shape=[len(ratio_correct) * 2])
    plot_point_x[0] = edges[0]
    plot_point_x[-1] = edges[-1]
    plot_point_x[1:-1] = np.repeat(edges[1:-1], repeats=2)

    plot_point_y = np.repeat(ratio_correct, repeats=2)

    # Fill the colours:
    for i in range(n_bins):
        if total_binned[i] != 0:
            ax.fill_between(edges[i:i + 2], ratio_correct[i], 1., color=red)
            ax.fill_between(edges[i:i + 2], 0., ratio_correct[i], color=green)

    ax.plot(plot_point_x, plot_point_y, color='white')

    ax.xlim([edges[0], edges[-1]])
    ax.ylim(y_lim)
    return ax


def plot_confusion_matrix_ratio_chart(tp, fp, tn, fn, spread, n_bins=20, ax=None, y_lim=(0., 1.0)):
    """
    :param tp: true positives np.ndarray of bool values (same shape as spread)
    :param fp: false positives
    :param tn: true negatives
    :param fn: false negatives
    :param spread: spread of ensemble for each example
    :param n_bins: int
    :param ax: either an plt.ax object or None. If None, default figure will be used.
    :param y_lim: range of y values on the plot
    :return:
    """
    if ax is None:
        # If ax not given, just plot on the current figure
        ax = plt
    min_x = 0.
    max_x = np.max(spread)
    # Extract the spreads of true positives, true negatives, ...
    spread_tp, spread_fp, spread_tn, spread_fn = map(lambda x: np.extract(x, spread), [tp, fp, tn, fn])

    # Count the number of true positives e.t.c. in each spread bin
    binned_tp, edges_tp = np.histogram(spread_tp, bins=n_bins, range=(min_x, max_x), density=False, normed=False)
    binned_fp, _ = np.histogram(spread_fp, bins=n_bins, range=(min_x, max_x), density=False, normed=False)
    binned_tn, _ = np.histogram(spread_tn, bins=n_bins, range=(min_x, max_x), density=False, normed=False)
    binned_fn, _ = np.histogram(spread_fn, bins=n_bins, range=(min_x, max_x), density=False, normed=False)

    edges = edges_tp

    total_binned = binned_tp + binned_fp + binned_tn + binned_fn

    # Get the ratios
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio_tp, ratio_fp, ratio_tn, ratio_fn = map(
            lambda binned_count: np.divide(binned_count.astype(np.float32), total_binned.astype(np.float32)),
            [binned_tp, binned_fp, binned_tn, binned_fn])
        for ratio in [ratio_tp, ratio_fp, ratio_tn, ratio_fn]:
            ratio[~ np.isfinite(ratio)] = 0.

    # Construct plot points:
    plot_point_x = np.empty(shape=[len(ratio_tp) * 2])
    plot_point_x[0] = edges[0]
    plot_point_x[-1] = edges[-1]
    plot_point_x[1:-1] = np.repeat(edges[1:-1], repeats=2)

    plot_point_y_tp = np.repeat(ratio_tp, repeats=2)
    plot_point_y_tn = np.repeat(ratio_tn, repeats=2) + plot_point_y_tp
    plot_point_y_fn = np.repeat(ratio_fn, repeats=2) + plot_point_y_tn

    # Fill the colours:
    for i in range(n_bins):
        if total_binned[i] != 0:
            ax.fill_between(plot_point_x[i*2:i*2 + 2], 0., plot_point_y_tp[i*2:i*2 + 2], color=green)
            ax.fill_between(plot_point_x[i*2:i*2 + 2], plot_point_y_tp[i*2:i*2 + 2], plot_point_y_tn[i*2:i*2 + 2], color=dark_green)
            ax.fill_between(plot_point_x[i*2:i*2 + 2], plot_point_y_tn[i*2:i*2 + 2], plot_point_y_fn[i*2:i*2 + 2], color=dark_orange)
            ax.fill_between(plot_point_x[i*2:i*2 + 2], plot_point_y_fn[i*2:i*2 + 2], 1., color=red)

    # Plot the white contour
    for plot_point_y in [plot_point_y_tp, plot_point_y_tn, plot_point_y_fn]:
        ax.plot(plot_point_x, plot_point_y, color='white', linewidth=0.9)

    ax.xlim([edges[0], edges[-1]])
    ax.ylim(y_lim)

    # Create the legend
    tp_patch = mpatches.Patch(color=green, label='True Positives')
    tn_patch = mpatches.Patch(color=dark_green, label='True Negatives')
    fn_patch = mpatches.Patch(color=dark_orange, label='False Negatives')
    fp_patch = mpatches.Patch(color=red, label='False Positives')
    plt.legend(handles=[tp_patch, tn_patch, fp_patch, fn_patch], bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
    return ax


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
    plt.xlabel("Percentage examples included as sorted by " + sort_by_name + " of ensemble predictions.")
    plt.ylabel("ROC AUC score on the subset examples included")
    return


def plot_pr_spread_mesh(labels, predictions, sort_by_array, pos_labels=1, resolution=40, spread_name='std'):
    assert pos_labels == 1 or pos_labels == 0

    if pos_labels == 0:
        labels = 1 - labels
        predictions = 1 - predictions

    num_examples = len(labels)

    sorted_order = np.argsort(sort_by_array)

    labels_sorted = labels[sorted_order]
    predictions_sorted = predictions[sorted_order]

    proportions_included = np.linspace(0, 1, num=resolution)
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    for i in range(resolution):
        proportion = proportions_included[i]
        last_idx = int(math.floor(num_examples * proportion)) + 1

        # Threshold the predictions based on sorted order:
        predictions_thresh = predictions_sorted.copy()
        predictions_thresh[last_idx:] = 0.

        precision, recall, _ = precision_recall_curve(labels_sorted, predictions_thresh)
        # Get rid of the last value (messes up the plot)
        precision, recall = precision[1:], recall[1:]
        ax.plot(recall, np.ones_like(recall) * proportion, precision, color=dark_blue, linewidth=0.6)

        del predictions_thresh

    ax.set_xlabel('Recall')
    ax.set_ylabel('Proportion classified by ensemble (' + spread_name + ')')
    ax.set_zlabel('Precision')
    return


def plot_family_of_curves_pr(labels, predictions, sort_by_array, proportions_included, pos_labels=1):
    assert pos_labels == 1 or pos_labels == 0

    if pos_labels == 0:
        labels = 1 - labels
        predictions = 1 - predictions

    num_examples = len(labels)
    sorted_order = np.argsort(sort_by_array)

    labels_sorted = labels[sorted_order]
    predictions_sorted = predictions[sorted_order]

    # Set a color map
    viridis = plt.get_cmap('viridis')
    c_norm = matplotlib.colors.Normalize(vmin=0, vmax=len(proportions_included))
    scalar_map = matplotlib.cm.ScalarMappable(norm=c_norm, cmap=viridis)

    for i in range(len(proportions_included)):
        proportion = proportions_included[i]

        last_idx = int(math.floor(num_examples * proportion)) + 1

        # Threshold the predictions based on sorted order:
        predictions_thresh = predictions_sorted.copy()
        predictions_thresh[last_idx:] = 0.

        precision, recall, _ = precision_recall_curve(labels_sorted, predictions_thresh)
        # Get rid of the first value (messes up the plot)
        precision, recall = precision[1:], recall[1:]

        color_val = scalar_map.to_rgba(i)
        plt.plot(recall, precision, color=color_val, linewidth=1.0, label="{0:.2f}".format(proportion), alpha=0.8)

        del predictions_thresh

    plt.legend(loc="lower left", ncol=2, prop={'size': 7}, title="Proportion Thresholded")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim(0.0, 1.0)
    return


def test_ratio_bar_chart(savedir=None):
    num_examples = 10000
    num_correct = int(0.7 * num_examples)
    num_incorrect = num_examples - num_correct

    correct_spread = np.random.randn(num_correct)
    incorrect_spread = np.random.uniform(-3, 3, size=num_incorrect)

    correct = np.empty(num_examples, dtype=np.bool)
    correct[:num_correct] = True
    correct[num_correct:] = False
    incorrect = np.invert(correct)
    spread = np.hstack((correct_spread, incorrect_spread))

    plt.figure(0)
    plot_ratio_bar_chart(correct, incorrect, spread, n_bins=40)
    if savedir:
        plt.savefig(savedir + '/test_ratio_bar_chart__ratio.png')
    plt.figure(1)
    plot_spread_histogram(correct, incorrect, spread, n_bins=40)
    if savedir:
        plt.savefig(savedir + '/test_ratio_bar_chart__histogram.png')
    plt.show()
    return


def test_plot_auc_vs_percentage_included():
    num_examples = 10000
    labels = np.random.randint(0, 2, size=num_examples)
    print(labels[:10])
    predictions = np.random.uniform(0, 1, size=num_examples)
    print(predictions[:10])

    spread = np.random.randn(num_examples)
    print(spread[:10])

    plot_auc_vs_percentage_included(labels, predictions, spread, resolution=100)
    plt.show()
    return


def main():
    args = parser.parse_args()

    model_dirs = [os.path.join(args.models_parent_dir, "atm_seed_{}".format(int(i))) for i in range(1, 11)]

    labels, ensemble_predictions = get_ensemble_predictions(model_dirs, rel_labels_filepath=args.rel_labels_path)

    avg_predictions = calc_avg_predictions(ensemble_predictions)

    std_spread = np.std(ensemble_predictions, axis=1)
    range_spread = np.ptp(ensemble_predictions, axis=1)
    iqr_spread = scipy.stats.iqr(ensemble_predictions, axis=1)  # interquartile range (IQR)
    mutual_information, entropy_of_avg = calc_mutual_information(ensemble_predictions)

    assert np.all(mutual_information >= 0.)

    mean_target_deviation = np.abs(labels - avg_predictions)
    # print("mean_target_deviation\n", mean_target_deviation.shape, "\n", mean_target_deviation[:5])

    correct = mean_target_deviation < 0.5
    incorrect = np.invert(correct)

    # Calculate the true_positives, true_negatives .. e.t.c.
    # Define POSITIVE as OFF TOPIC
    tp = np.logical_and(correct, avg_predictions < 0.5)
    tn = np.logical_and(correct, avg_predictions >= 0.5)
    fp = np.logical_and(incorrect, avg_predictions < 0.5)
    fn = np.logical_and(incorrect, avg_predictions >= 0.5)

    print("Metrics calculated")

    # Make the plots:
    savedir = args.savedir

    #    RATIOS PLOTS
    # Make the std ratios plots
    #plot_ratio_bar_chart(correct, incorrect, std_spread, n_bins=40, y_lim=[0.0, 1.0])
    #plt.xlabel("Spread (std of ensemble predictions)")
    #plt.ylabel("Ratio correct to incorrect predictions (thresh = 0.5)")
    #plt.savefig(savedir + '/ratios_std_spread_histogram.png', bbox_inches='tight')
    #plt.clf()

    # Make the range ratios plots
    #plot_ratio_bar_chart(correct, incorrect, range_spread, n_bins=40, y_lim=[0.0, 1.0])
    #plt.xlabel("Spread (range of ensemble predictions)")
    #plt.ylabel("Ratio correct to incorrect predictions (thresh = 0.5)")
    #plt.savefig(savedir + '/ratios_range_spread_histogram.png', bbox_inches='tight')
    #plt.clf()

    # Make the mutual information ratios plots
    plot_ratio_bar_chart(correct, incorrect, mutual_information, n_bins=40, y_lim=[0.0, 1.0])
    plt.xlabel("Mutual Information")
    plt.ylabel("Ratio correct to incorrect predictions (thresh = 0.5)")
    plt.savefig(savedir + '/ratios_mutual_info_histogram.png', bbox_inches='tight')
    plt.clf()

    # Make the mutual information ratios confusion matrix plot:
    plot_confusion_matrix_ratio_chart(tp, fp, tn, fn, mutual_information, n_bins=40)
    plt.xlabel("Mutual Information")
    plt.ylabel("Ratios in each bin")
    plt.savefig(savedir + '/conf_mat_ratios_mutual_info_histogram.png', bbox_inches='tight')
    plt.clf()

    #     HISTOGRAM PLOTS
    # Make std_spread histogram plot:
    #plot_spread_histogram(correct, incorrect, std_spread, n_bins=40, spread_name='std')
    #plt.savefig(savedir + '/std_spread_histogram.png', bbox_inches='tight')
    #plt.clf()

    # # Make range_spread histogram plot:
    # plot_spread_histogram(correct, incorrect, range_spread, n_bins=40, spread_name='range')
    # plt.savefig(savedir + '/range_spread_histogram.png', bbox_inches='tight')
    # plt.clf()

    # Make mutual_infromation histogram plot:
    plot_spread_histogram(correct, incorrect, mutual_information, n_bins=40, spread_name='mutual information')
    plt.savefig(savedir + '/mutual_info_histogram.png', bbox_inches='tight')
    plt.clf()

    #   SCATTER Plots
    # Make std_spread vs mean deviation plot
    marker_size = 0.01
    # Positive examples
    sns.jointplot(mutual_information, mean_target_deviation, kind='kde')
    plt.xlabel("Spread (std of ensemble predictions)")
    plt.ylabel("Deviation of average ensemble prediction from label")
    plt.savefig(savedir + '/mutual_information_vs_mean_density.png', bbox_inches='tight')
    plt.clf()

    plt.scatter(np.extract(labels.astype(np.bool), mutual_information),
                np.extract(labels.astype(np.bool), mean_target_deviation), alpha=0.5, marker='o',
                s=marker_size)
    # Negative examples
    plt.scatter(np.extract(np.invert(labels.astype(np.bool)), mutual_information),
                np.extract(np.invert(labels.astype(np.bool)), mean_target_deviation), alpha=0.5,  marker='x',
                s=marker_size)
    plt.xlabel("Spread (std of ensemble predictions)")
    plt.ylabel("Deviation of average ensemble prediction from label")
    plt.savefig(savedir + '/mutual_information_vs_mean_chart.png', bbox_inches='tight')
    plt.clf()

    # Split positive and negative examples into separate plots as well:
    # Positive examples
    #plt.scatter(np.extract(labels.astype(np.bool), std_spread),
    #            1 - np.extract(labels.astype(np.bool), avg_predictions), alpha=0.15, color=green, marker='o',
    #            s=marker_size)
    #plt.xlabel("Spread (std of ensemble predictions)")
    #plt.ylabel("Ensemble Prediction (1 = off-topic)")
    #plt.savefig(savedir + '/std_spread_vs_prediction_on_topic.png', bbox_inches='tight')
    #plt.clf()
    # Negative examples
    #plt.scatter(np.extract(np.invert(labels.astype(np.bool)), std_spread),
    #            1 - np.extract(np.invert(labels.astype(np.bool)), avg_predictions), alpha=0.15, color=red, marker='x',
    #            s=marker_size)
    #plt.xlabel("Spread (std of ensemble predictions)")
    #plt.ylabel("Ensemble Prediction (1 = off-topic)")
    #plt.savefig(savedir + '/std_spread_vs_prediction_off_topic.png', bbox_inches='tight')
    #plt.clf()

    # Make range vs mean deviation plot
    # Positive examples
    #plt.scatter(np.extract(labels.astype(np.bool), std_spread),
    #            np.extract(labels.astype(np.bool), mean_target_deviation), alpha=0.15, color=green, marker='o',
    #            s=marker_size)
    # Negative examples
    #plt.scatter(np.extract(np.invert(labels.astype(np.bool)), std_spread),
    #            np.extract(np.invert(labels.astype(np.bool)), mean_target_deviation), alpha=0.15, color=red, marker='x',
    #            s=marker_size)
    #plt.xlabel("Spread (range of ensemble predictions)")
    #plt.ylabel("Deviation of average ensemble prediction from label")
    #plt.savefig(savedir + '/range_spread_vs_mean_chart.png', bbox_inches='tight')
    #plt.clf()

    #   AUC vs. CUMULATIVE INCLUDED
    # Make AUC vs. cumulative samples included by std spread
    #plot_auc_vs_percentage_included(labels, avg_predictions, std_spread, resolution=200, sort_by_name='std')
    #plt.savefig(savedir + '/auc_vs_cumulative_samples_included_std.png', bbox_inches='tight')
    #plt.clf()

    # # Make AUC vs. cumulative samples included by range spread
    # plot_auc_vs_percentage_included(labels, avg_predictions, range_spread, resolution=200, sort_by_name='range')
    # plt.savefig(savedir + '/auc_vs_cumulative_samples_included_range.png', bbox_inches='tight')
    # plt.clf()

    # Make AUC vs. cumulative samples included by range spread
    plot_auc_vs_percentage_included(labels, avg_predictions, mutual_information, resolution=200,
                                    sort_by_name='mutual information')
    plt.savefig(savedir + '/auc_vs_cumulative_samples_included_mutual_info.png', bbox_inches='tight')
    plt.clf()

    # Make precision recall curve for average of predictions
    for i in range(10):
        precision, recall, _ = precision_recall_curve(1 - labels, 1 - ensemble_predictions[:, i])
        plt.plot(recall, precision, color='black', alpha=0.1, linewidth=0.6)
    precision, recall, thresholds = precision_recall_curve(1 - labels, 1 - avg_predictions)
    plt.plot(recall, precision, color=dark_blue, alpha=1.0)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim(0.0, 1.0)
    plt.xlim(0.0, 1.0)
    plt.savefig(savedir + '/ensemble_pr_curve.png', bbox_inches='tight')
    plt.clf()

    # PR 3D PLOTS
    # Make precision recall 3D plot vs std
    #plot_pr_spread_mesh(labels, avg_predictions, std_spread, pos_labels=0)
    #plt.savefig(savedir + '/ensemble_3d_pr_curve_std.png')
    #plt.clf()

    # Make precision recall 3D plot vs mutual information
    plot_pr_spread_mesh(labels, avg_predictions, mutual_information, pos_labels=0)
    plt.savefig(savedir + '/ensemble_3d_pr_curve_mutual_info.png')
    plt.clf()

    # Make family of curves precision recall plot vs. std
    #plot_family_of_curves_pr(labels, avg_predictions, std_spread, proportions_included=[0.05, 0.1, 0.2, 0.4, 0.6, 1.0],
    #                         pos_labels=0)
    #plt.savefig(savedir + '/ensemble_family_pr_curve_std.png', bbox_inches='tight')
    #plt.clf()

    # Make family of curves precision recall plot vs. std
    plot_family_of_curves_pr(labels, avg_predictions, mutual_information,
                             proportions_included=[0.05, 0.1, 0.2, 0.4, 0.6, 1.0],
                             pos_labels=0)
    plt.savefig(savedir + '/ensemble_family_pr_curve_mutual_info.png', bbox_inches='tight')
    plt.clf()

    test_ratio_bar_chart(savedir=args.savedir)
    return


if __name__ == "__main__":
    main()
    # test_plot_auc_vs_percentage_included()
