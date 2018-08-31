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

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score

# Specify the colours
green = (0.3, 0.9, 0.3)
red = (0.9, 0.3, 0.3)


def get_ensemble_predictions(model_dirs, rel_labels_filepath='eval4_naive/labels-probs.txt'):
    """
    Get the target labels and model predictions from a txt file from all the models pointed to by list model_dirs.
    :param model_dirs: list of paths to model directories
    :param rel_labels_filepath: path to where the labels/predictions file is located within each model directory
    :return: ndarray of target labels and ndarray predictions of each model with shape [num_examples, num_models]
    """
    labels_files = map(lambda model_dir: os.path.join(model_dir, rel_labels_filepath), model_dirs)

    # Get the target labels:
    labels, _ = get_label_predictions(labels_files[0])

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
    ax.hist(spread_correct, n_bins, density=True, fill=False,
            histtype='step', stacked=True, color='white')
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
    correct_binned, edges_correct = np.histogram(spread_correct, bins=n_bins, range=(min_x, max_x), density=False)
    incorrect_binned, edges_incorrect = np.histogram(spread_incorrect, bins=n_bins, range=(min_x, max_x), density=False)

    assert np.all(edges_correct == edges_incorrect)
    edges = edges_correct

    print(correct_binned)
    print(incorrect_binned)
    total_binned = correct_binned + incorrect_binned
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio_correct = np.divide(correct_binned, total_binned)
        ratio_correct[~ np.isfinite(ratio_correct)] = 0.
    print("shape bin count : ", ratio_correct.shape)
    print("shape edges ", edges.shape)

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

    ax.plot(plot_point_x, plot_point_y, color='blue')

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
        print(last_idx)
        labels_subset = labels_sorted[:last_idx]
        predictions_subset = predictions_sorted[:last_idx]

        # print(len(labels_subset), len(predictions_subset))
        # print(labels_subset[max(0, last_idx-5): last_idx])
        try:
            roc_auc_scores[i] = roc_auc_score(labels_subset, predictions_subset)
        except ValueError:
            roc_auc_scores[i] = np.nan

    print(roc_auc_scores)
    plt.plot(proportions_included, roc_auc_scores, color=(.2, .2, .6))

    return


def test_ratio_bar_chart():
    num_examples = 10000
    correct = np.random.randint(0, 2, size=num_examples).astype(dtype=np.bool)
    incorrect = np.invert(correct)
    spread = np.random.randn(num_examples)

    plot_ratio_bar_chart(correct, incorrect, spread, n_bins=40)
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
    models_parent_dir = '/home/miproj/urop.2018/bkm28/seed_experiments'
    model_dirs = [os.path.join(models_parent_dir, "atm_seed_{}".format(int(i))) for i in range(1, 11)]

    labels, ensemble_predictions = get_ensemble_predictions(model_dirs)
    # print(ensemble_predictions[:5, :])  # todo: remove
    # print(ensemble_predictions.shape)
    # print("Predictions retrieved")

    avg_predictions = calc_avg_predictions(ensemble_predictions)
    # print("avg_predictions\n", avg_predictions.shape, "\n", avg_predictions[:5])  # todo: remove

    std_spread = np.std(ensemble_predictions, axis=1)
    # print("std_spread:\n", std_spread[:5])  # todo: remove
    range_spread = np.ptp(ensemble_predictions, axis=1)
    iqr_spread = scipy.stats.iqr(ensemble_predictions, axis=1)  # interquartile range (IQR)

    mean_target_deviation = np.abs(labels - avg_predictions)
    # print("mean_target_deviation\n", mean_target_deviation.shape, "\n", mean_target_deviation[:5])

    correct = mean_target_deviation < 0.5
    incorrect = np.invert(correct)
    print("Metrics calculated")

    # Make the plots:
    savedir = "/home/alta/WebDir/ind_reports/bkm28"

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

    # Make the ratios plots
    plot_ratio_bar_chart(correct, incorrect, std_spread, n_bins=30, y_lim=[0.9, 1.0])
    plt.savefig(savedir + '/ratios_std_spread_histogram.png', bbox_inches='tight')
    plt.clf()

    # Make std_spread vs mean deviation plot
    # Positive examples
    plt.scatter(np.extract(labels.astype(np.bool), std_spread),
                np.extract(labels.astype(np.bool), mean_target_deviation), alpha=0.2, color=green, marker='o', s=0.5)
    plt.scatter(np.extract(np.invert(labels.astype(np.bool)), std_spread),
                np.extract(np.invert(labels.astype(np.bool)), mean_target_deviation), alpha=0.2, color=red, marker='x',
                s=0.5)
    plt.savefig(savedir + '/spread_vs_mean_chart.png', bbox_inches='tight')
    plt.clf()

    # Make AUC vs. cumulative samples included by spread
    plot_auc_vs_percentage_included(labels, avg_predictions, std_spread, resolution=200)
    plt.xlabel("Percentage examples included as sorted by std of ensemble predictions (from low to high)")
    plt.ylabel("ROC AUC score on the subset examples included")
    plt.savefig(savedir + '/auc_vs_cumulative_samples_included.png', bbox_inches='tight')
    plt.clf()
    return


if __name__ == "__main__":
    main()
    # test_plot_auc_vs_percentage_included()
