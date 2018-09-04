"""
Just a one-off script for comparing the seed spread versus confidence
"""
from __future__ import print_function, division

import sys
import os
import numpy as np
import scipy.stats
import math
import matplotlib
import argparse
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm
import matplotlib.colors
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

parser = argparse.ArgumentParser(description='Plot useful graphs for evaluation.')
parser.add_argument('--savedir', type=str, default='./',
                               help='Path to directory where to save the plots')
parser.add_argument('--rel_labels_path', type=str, default='eval4_naive/labels-probs.txt')


matplotlib.rcParams['savefig.dpi'] = 200
# Specify the colours
green = (0.3, 0.9, 0.3)
red = (0.9, 0.3, 0.3)
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


def plot_spread_histogram(correct, incorrect, spread, n_bins=20, ax=None):
    if ax is None:
        ax = plt
    spread_correct = np.extract(correct, spread)
    spread_incorrect = np.extract(incorrect, spread)
    ax.hist((spread_correct, spread_incorrect), n_bins, density=True,
            histtype='bar', stacked=True, color=['green', 'red'])
    ax.hist((spread_correct, spread_incorrect), n_bins, density=True, fill=False,
            histtype='step', stacked=True, color=['white', 'white'])
    return ax


def plot_ratio_bar_chart(correct, incorrect, spread, n_bins=20, ax=None, y_lim=[0., 1.0]):
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
    spread_correct = np.extract(correct, spread)
    spread_incorrect = np.extract(incorrect, spread)
    correct_binned, edges_correct = np.histogram(spread_correct, bins=n_bins, range=(min_x, max_x), density=False, normed=False)
    incorrect_binned, edges_incorrect = np.histogram(spread_incorrect, bins=n_bins, range=(min_x, max_x), density=False, normed=False)

    assert np.all(edges_correct == edges_incorrect)
    edges = edges_correct

    total_binned = correct_binned + incorrect_binned
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio_correct = np.divide(correct_binned.astype(np.float32), total_binned.astype(np.float32))
        ratio_correct[ ~ np.isfinite(ratio_correct)] = 0.

    # Construct plot points:
    plot_point_x = np.empty(shape=[len(ratio_correct) * 2])
    plot_point_x[0] = edges[0]
    plot_point_x[-1] = edges[-1]
    plot_point_x[1:-1] = np.repeat(edges[1:-1], repeats=2)

    plot_point_y = np.repeat(ratio_correct, repeats=2)

    # Fill the colours:
    for i in range(n_bins):
        # ax.fill_between(edges[i:i+2], plot_point_y[2*i: 2*i + 2], 1., color=green)
        # ax.fill_between(edges[i:i+2], 0., plot_point_y[2])
        if total_binned[i] != 0:
            ax.fill_between(edges[i:i + 2], ratio_correct[i], 1., color=red)
            ax.fill_between(edges[i:i + 2], 0., ratio_correct[i], color=green)

    ax.plot(plot_point_x, plot_point_y, color='white')

    ax.xlim([edges[0], edges[-1]])
    ax.ylim(y_lim)
    return ax


def plot_auc_vs_percentage_included(labels, predictions, sort_by_array, resolution=100):
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


def plot_family_of_curves_pr(labels, predictions, sort_by_array, proportions_included, pos_labels=1, spread_name='std'):
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
    c_norm  = matplotlib.colors.Normalize(vmin=0, vmax=len(proportions_included))
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

    models_parent_dir = '/home/miproj/urop.2018/bkm28/seed_experiments'
    model_dirs = [os.path.join(models_parent_dir, "atm_seed_{}".format(int(i))) for i in range(1, 11)]

    labels, ensemble_predictions = get_ensemble_predictions(model_dirs, rel_labels_filepath=args.rel_labels_path)
    # print(ensemble_predictions[:5, :])  # todo: remove
    # print(ensemble_predictions.shape)
    # print("Predictions retrieved")

    avg_predictions = calc_avg_predictions(ensemble_predictions)
    # print("avg_predictions\n", avg_predictions.shape, "\n", avg_predictions[:5])  # todo: remove

    std_spread = np.std(ensemble_predictions, axis=1)
    range_spread = np.ptp(ensemble_predictions, axis=1)
    iqr_spread = scipy.stats.iqr(ensemble_predictions, axis=1)  # interquartile range (IQR)

    mean_target_deviation = np.abs(labels - avg_predictions)
    # print("mean_target_deviation\n", mean_target_deviation.shape, "\n", mean_target_deviation[:5])

    correct = mean_target_deviation < 0.5
    incorrect = np.invert(correct)
    print("Metrics calculated")

    # Make the plots:
    savedir = args.savedir

    # Make the std ratios plots
    plot_ratio_bar_chart(correct, incorrect, std_spread, n_bins=40, y_lim=[0.0, 1.0])
    plt.xlabel("Spread (std of ensemble predictions)")
    plt.ylabel("Ratio correct to incorrect predictions (thresh = 0.5)")
    plt.savefig(savedir + '/ratios_std_spread_histogram.png', bbox_inches='tight')
    plt.clf()

    # Make the range ratios plots
    plot_ratio_bar_chart(correct, incorrect, range_spread, n_bins=40, y_lim=[0.0, 1.0])
    plt.xlabel("Spread (range of ensemble predictions)")
    plt.ylabel("Ratio correct to incorrect predictions (thresh = 0.5)")
    plt.savefig(savedir + '/ratios_range_spread_histogram.png', bbox_inches='tight')
    plt.clf()

    # Make std_spread histogram plot:
    plot_spread_histogram(correct, incorrect, std_spread, n_bins=30)
    plt.xlabel("Spread (std of ensemble predictions)")
    plt.ylabel("Example Count")
    plt.savefig(savedir + '/std_spread_histogram.png', bbox_inches='tight')
    plt.clf()

    # Make range_spread histogram plot:
    plot_spread_histogram(correct, incorrect, range_spread, n_bins=30)
    plt.savefig(savedir + '/range_spread_histogram.png', bbox_inches='tight')
    plt.xlabel("Spread (range of ensemble predictions)")
    plt.ylabel("Example Count")
    plt.clf()
    
    # Make std_spread vs mean deviation plot
    marker_size = 0.01
    # Positive examples
    plt.scatter(np.extract(labels.astype(np.bool), std_spread),
                np.extract(labels.astype(np.bool), mean_target_deviation), alpha=0.15, color=green, marker='o', s=marker_size)
    # Negative examples
    plt.scatter(np.extract(np.invert(labels.astype(np.bool)), std_spread),
                np.extract(np.invert(labels.astype(np.bool)), mean_target_deviation), alpha=0.15, color=red, marker='x',
                s=marker_size)
    plt.xlabel("Spread (std of ensemble predictions)")
    plt.ylabel("Deviation of average ensemble prediction from label")
    plt.savefig(savedir + '/std_spread_vs_mean_chart.png', bbox_inches='tight')
    plt.clf()
    
    # Split positive and negative examples into separate plots as well:
    # Positive examples
    plt.scatter(np.extract(labels.astype(np.bool), std_spread),
                1 - np.extract(labels.astype(np.bool), avg_predictions), alpha=0.15, color=green, marker='o', s=marker_size)
    plt.xlabel("Spread (std of ensemble predictions)")
    plt.ylabel("Ensemble Prediction (1 = off-topic)")
    plt.savefig(savedir + '/std_spread_vs_prediction_on_topic.png', bbox_inches='tight')
    plt.clf()
    # Negative examples
    plt.scatter(np.extract(np.invert(labels.astype(np.bool)), std_spread),
                1 - np.extract(np.invert(labels.astype(np.bool)), avg_predictions), alpha=0.15, color=red, marker='x',
                s=marker_size)
    plt.xlabel("Spread (std of ensemble predictions)")
    plt.ylabel("Ensemble Prediction (1 = off-topic)")
    plt.savefig(savedir + '/std_spread_vs_prediction_off_topic.png', bbox_inches='tight')
    plt.clf()
    

    # Make range vs mean deviation plot
    # Positive examples
    plt.scatter(np.extract(labels.astype(np.bool), std_spread),
                np.extract(labels.astype(np.bool), mean_target_deviation), alpha=0.15, color=green, marker='o', s=marker_size)
    # Negative examples
    plt.scatter(np.extract(np.invert(labels.astype(np.bool)), std_spread),
                np.extract(np.invert(labels.astype(np.bool)), mean_target_deviation), alpha=0.15, color=red, marker='x',
                s=marker_size)
    plt.xlabel("Spread (range of ensemble predictions)")
    plt.ylabel("Deviation of average ensemble prediction from label")
    plt.savefig(savedir + '/range_spread_vs_mean_chart.png', bbox_inches='tight')
    plt.clf()

    # Make AUC vs. cumulative samples included by std spread
    plot_auc_vs_percentage_included(labels, avg_predictions, std_spread, resolution=200)
    plt.xlabel("Percentage examples included as sorted by std of ensemble predictions (from low to high)")
    plt.ylabel("ROC AUC score on the subset examples included")
    plt.savefig(savedir + '/auc_vs_cumulative_samples_included_std.png', bbox_inches='tight')
    plt.clf()

    # Make AUC vs. cumulative samples included by range spread
    plot_auc_vs_percentage_included(labels, avg_predictions, range_spread, resolution=200)
    plt.xlabel("Percentage examples included as sorted by range of ensemble predictions (from low to high)")
    plt.ylabel("ROC AUC score on the subset examples included")
    plt.savefig(savedir + '/auc_vs_cumulative_samples_included_range.png', bbox_inches='tight')
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

    # Make precision recall 3D plot
    plot_pr_spread_mesh(labels, avg_predictions, std_spread, pos_labels=0)
    plt.savefig(savedir + '/ensemble_3d_pr_curve_std.png')
    plt.clf()

    # Make family of curves precision recall plot
    plot_family_of_curves_pr(labels, avg_predictions, std_spread, proportions_included=[0.05, 0.1, 0.2, 0.4, 0.6, 1.0], pos_labels=0, spread_name='std')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim(0.0, 1.0)
    plt.savefig(savedir + '/ensemble_family_pr_curve_std.png', bbox_inches='tight')
    plt.clf()


    return


if __name__ == "__main__":
    main()
    test_ratio_bar_chart(savedir="/home/alta/WebDir/ind_reports/bkm28")
    # test_plot_auc_vs_percentage_included()
