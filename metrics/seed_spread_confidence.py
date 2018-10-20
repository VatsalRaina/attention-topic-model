#! /usr/bin/env python
"""
Just a one-off script for comparing the seed spread versus confidence
"""
from __future__ import print_function, division

import matplotlib

matplotlib.use('Agg')
import os
import numpy as np
import scipy.stats
import math
import argparse

import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.colors
import matplotlib.patches as mpatches

from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

parser = argparse.ArgumentParser(description='Plot useful graphs for evaluation.')
parser.add_argument('models_parent_dir', type=str, help='Path to ensemble directory')
parser.add_argument('--savedir', type=str, default='./',
                    help='Path to directory where to save the plots')
parser.add_argument('--rel_labels_path', type=str, default='eval4_naive/labels-probs.txt')
parser.add_argument('--rel_attention_path', type=str, default='eval4_naive/prompt_attention.txt')
parser.add_argument('--n_levels', type=int, default=20)
parser.add_argument('--hatm', action='store_true', help='Whether to analyse ATM or HATM ensemble')

matplotlib.rcParams['savefig.dpi'] = 200
import seaborn as sns

sns.set()

# Specify the colours
green = (0.3, 0.9, 0.3)
dark_green = (0.1, 0.7, 0.3)
red = (0.9, 0.3, 0.3)
dark_orange = (0.9, 0.5, 0.2)
dark_blue = (0.1, 0.15, 0.27)


def add_off_topic_probability(predictions):
    """

    :param predictions: array of prediction of probability of positive class
    :return: array of shape [batchsize, 2] which probabiltiies of negative class in first index and positive in second
    """

    p_off_topic = 1.0 - predictions
    predictions = np.concatenate((p_off_topic[:, np.newaxis], predictions[:, np.newaxis]), axis=1)

    return predictions


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


def get_ensemble_prompt_entropies(model_dirs, rel_labels_filepath='eval4_naive/prompt_attention.txt'):
    """
    Get measures of uncertainty from prompt attention mechanism of an HATM
    :param model_dirs: list of paths to model directories
    :param rel_labels_filepath: path to where the prompt attention file is located within each model directory
    :return: ndarray arrays of entropies of the mean attention, mean entopies of the attention and mutual information for each input - shape [num_examples]
    """
    attention_files = map(lambda model_dir: os.path.join(model_dir, rel_labels_filepath), model_dirs)

    # List to store predictions from all the models considered
    attention_arrays = []
    for attention_filepath in attention_files:
        # Get the predictions from each of the models
        attention = np.loadtxt(attention_filepath, dtype=np.float64) + 1e-10
        attention_arrays.append(attention)
    attention = np.stack(attention_arrays, axis=2)

    mean_attention = np.mean(attention, axis=2)
    prompt_entropy_mean = - np.sum(mean_attention * np.log(mean_attention), axis=1)
    prompt_entropies = -np.sum(attention * np.log(attention), axis=1)
    prompt_mean_entropy =  np.mean(prompt_entropies, axis=1)
    prompt_mutual_information = prompt_entropy_mean - prompt_mean_entropy

    return prompt_entropy_mean, prompt_mean_entropy, prompt_mutual_information, prompt_entropies


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
    # ax.hist((spread_correct, spread_incorrect), n_bins, density=True, fill=False,
    #        histtype='step', stacked=True, color=['white', 'white'])
    plt.xlabel("Spread (" + spread_name + " of ensemble predictions)")
    plt.ylabel("Example Count")
    plt.legend(['Correct', 'In-Correct'])
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
            ax.fill_between(plot_point_x[i * 2:i * 2 + 2], 0., plot_point_y_tp[i * 2:i * 2 + 2], color=green)
            ax.fill_between(plot_point_x[i * 2:i * 2 + 2], plot_point_y_tp[i * 2:i * 2 + 2],
                            plot_point_y_tn[i * 2:i * 2 + 2], color=dark_green)
            ax.fill_between(plot_point_x[i * 2:i * 2 + 2], plot_point_y_tn[i * 2:i * 2 + 2],
                            plot_point_y_fn[i * 2:i * 2 + 2], color=dark_orange)
            ax.fill_between(plot_point_x[i * 2:i * 2 + 2], plot_point_y_fn[i * 2:i * 2 + 2], 1., color=red)

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


def plot_auc_vs_percentage_included(labels, predictions, sort_by_array, resolution=100, sort_by_name='std',
                                    savedir=None):
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
            roc_auc_scores[i] = 1.0

    plt.plot(proportions_included, roc_auc_scores)
    plt.xlabel("Percentage examples included")
    plt.ylabel("ROC AUC score on the subset examples included")
    plt.xlim(0.0, 1.0)
    plt.ylim(np.min(roc_auc_scores), 1.0)
    with open(os.path.join(savedir, 'ensemble_auc.txt'), 'a') as f:
        f.write('ROC AUC of Ensemble is: ' + str(roc_auc_scores[-1]) + '\n')
    return


def plot_auc_vs_percentage_included_ensemble(labels, predictions, sort_by_array, resolution=100, sort_by_name='std',
                                    savedir=None):
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

    proportions_included = np.linspace(0, 1, num=resolution)
    roc_auc_scores = np.zeros(shape=[proportions_included.shape[0], predictions.shape[-1]], dtype=np.float32)

    for fold in range(predictions.shape[-1]):
        sorted_order = np.argsort(sort_by_array[:,fold])

        labels_sorted = labels[sorted_order]
        predictions_sorted = predictions[sorted_order, fold]
        for i in range(resolution):
            proportion = proportions_included[i]
            last_idx = int(math.floor(num_examples * proportion)) + 1
            labels_subset = labels_sorted[:last_idx]
            predictions_subset = predictions_sorted[:last_idx]

            # print(len(labels_subset), len(predictions_subset))
            # print(labels_subset[max(0, last_idx-5): last_idx])
            try:
                roc_auc_scores[i,fold] = roc_auc_score(labels_subset, predictions_subset)
            except ValueError:
                roc_auc_scores[i,fold] = 1.0

    mean_roc = np.mean(roc_auc_scores,axis=1)
    std_roc = np.std(roc_auc_scores, axis=1)
    plt.plot(proportions_included, mean_roc)
    plt.fill_between(proportions_included, mean_roc - std_roc, mean_roc + std_roc, alpha=.2)
    plt.xlabel("Percentage examples included")
    plt.ylabel("ROC AUC score on the subset examples included")
    plt.xlim(0.0, 1.0)
    plt.ylim(np.min(mean_roc-std_roc), 1.0)
    with open(os.path.join(savedir, 'ensemble_auc.txt'), 'a') as f:
        f.write('ROC AUC of Ensemble is: ' + str(mean_roc[-1]) + '\n')
    return

def plot_aupr_vs_percentage_included(labels, predictions, sort_by_array, pos_label=1, resolution=100,
                                    savedir=None):
    """
    Plot the AUPR score vs. the percentage of examples included where the examples are sorted by the array
    sort_by_array. This array could for instance represent the spread of the ensemble predictions, and hence the
    curve would show the performance on the subset of the examples given by thresholding the spread.
    :param labels: target label array
    :param predictions: label probabilities as predicted by the model
    :param pos_label: whether to consider on-topic or off-topic as the positive class
    :param resolution: Number of points to plot
    :return:
    """
    assert pos_label == 0 or pos_label ==1
    num_examples = len(labels)

    sorted_order = np.argsort(sort_by_array)

    labels_sorted = labels[sorted_order]
    predictions_sorted = predictions[sorted_order]

    if pos_label==0:
        predictions_sorted=1.0-predictions_sorted

    proportions_included = np.linspace(0, 1, num=resolution)

    aupr_scores = np.zeros_like(proportions_included)
    for i in range(resolution):
        proportion = proportions_included[i]
        last_idx = int(math.floor(num_examples * proportion)) + 1
        labels_subset = labels_sorted[:last_idx]
        predictions_subset = predictions_sorted[:last_idx]

        try:
            precision, recall, _ = precision_recall_curve(labels_subset, predictions_subset, pos_label=pos_label)
            aupr_scores[i] = auc(recall, precision)
        except ValueError:
            aupr_scores[i] = 1.0

    plt.plot(proportions_included, aupr_scores)
    plt.xlabel("Percentage examples included")
    plt.ylabel("AUPR score on the subset examples included")
    plt.xlim(0.0, 1.0)
    plt.ylim(np.min(aupr_scores), 1.0)
    with open(os.path.join(savedir, 'ensemble_auc.txt'), 'a') as f:
        f.write('AUPR with pos label of '+str(pos_label)+ 'of Ensemble is: ' + str(aupr_scores[-1]) + '\n')
    return

def plot_aupr_vs_percentage_included_ensemble(labels, predictions, sort_by_array, pos_label=1, resolution=100,
                                    savedir=None):
    """
    Plot the AUPR score vs. the percentage of examples included where the examples are sorted by the array
    sort_by_array. This array could for instance represent the spread of the ensemble predictions, and hence the
    curve would show the performance on the subset of the examples given by thresholding the spread.
    :param labels: target label array
    :param predictions: label probabilities as predicted by the model
    :param pos_label: whether to consider on-topic or off-topic as the positive class
    :param resolution: Number of points to plot
    :return:
    """
    assert pos_label == 0 or pos_label ==1
    num_examples = len(labels)
    proportions_included = np.linspace(0, 1, num=resolution)

    print(predictions.shape)
    aupr_scores = np.zeros(shape=[proportions_included.shape[0], predictions.shape[-1]], dtype=np.float32)

    print(sort_by_array[:,0])
    for fold in range(predictions.shape[-1]):
        sorted_order = np.argsort(sort_by_array[:,fold])

        labels_sorted = labels[sorted_order]
        if pos_label == 0:
            predictions_sorted = 1.0-predictions[sorted_order, fold]
        else:
            predictions_sorted = predictions[sorted_order, fold]

        for i in range(resolution):
            proportion = proportions_included[i]
            last_idx = int(math.floor(num_examples * proportion)) + 1
            labels_subset = labels_sorted[:last_idx]
            predictions_subset = predictions_sorted[:last_idx]

            try:
                precision, recall, _ = precision_recall_curve(labels_subset, predictions_subset, pos_label=pos_label)
                aupr_scores[i,fold] = auc(recall, precision)
            except ValueError:
                aupr_scores[i,fold] =  1.0

    print(mean_roc)
    mean_roc = np.mean(aupr_scores,axis=1)
    std_roc = np.std(aupr_scores, axis=1)

    plt.plot(proportions_included, mean_roc)
    plt.fill_between(proportions_included, mean_roc - std_roc, mean_roc + std_roc, alpha=.2)
    plt.xlabel("Percentage examples included")
    plt.ylabel("AUPR score on the subset examples included")
    plt.xlim(0.0, 1.0)
    plt.ylim(np.min(mean_roc-std_roc), 1.0)
    with open(os.path.join(savedir, 'ensemble_auc.txt'), 'a') as f:
        f.write('AUPR with pos label of '+str(pos_label)+ 'of Ensemble is: ' + str(mean_roc[-1]) + '+\-' + str(std_roc[-1])+'\n')
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


def run_misclassification_detection(misclassification_labels, uncertainty):

    precision, recall, thresholds = precision_recall_curve(misclassification_labels, uncertainty, pos_label=1)
    aupr_pos = auc(recall, precision)

    precision, recall, thresholds = precision_recall_curve(misclassification_labels, -uncertainty, pos_label=0)
    aupr_neg = auc(recall, precision)

    roc_auc = roc_auc_score(misclassification_labels, uncertainty)

    return [roc_auc, aupr_pos, aupr_neg]

def run_misclassification_detection_over_ensemble(labels, predictions, prompt_entopies=None, savedir=None):
    """
    This functions runs a uncertainty based miclassification detection experiment

    :param labels: Labels on-topic / off-topic
    :param predictions: array of predictions (probabilities) of [batch_size, size_ensemble]
    :param prompt_entopies: array of entropies of prompt attention mechanism [batch_size, size_ensemble]
    :return: None. Saves stuff
    """

    off_topic_probabilities = 1.0 - predictions

    on_topic_probabilities = np.reshape(predictions, newshape=[predictions.shape[0], 1, predictions.shape[-1]])
    off_topic_probabilities = np.reshape(off_topic_probabilities, newshape=on_topic_probabilities.shape)
    probabilities = np.concatenate([off_topic_probabilities, on_topic_probabilities], axis=1)

    entropies = - np.sum((probabilities)*np.log(probabilities+1e-10),axis=1)


    predictions = np.argmax(probabilities, axis=1)

    print(labels.shape)
    misclassification = np.asarray(labels[:,np.newaxis] != predictions, dtype= np.int32)
    correct = np.asarray(labels[:,np.newaxis] == predictions, dtype= np.float32)
    print(correct.shape)
    accuracies= np.mean(correct,axis=0)
    m_accuracy = np.mean(accuracies)
    std_accuract = np.std(accuracies)

    auc_array_entropy=[]
    for i in range(predictions.shape[-1]):
        auc = run_misclassification_detection(misclassification[:,i], entropies[:,i])
        auc_array_entropy.append(auc)

    auc_entropy= np.stack(auc_array_entropy, axis=0)
    auc_entropy_mean, auc_entropy_std = np.mean(auc_entropy, axis=0), np.std(auc_entropy, axis=0)

    if prompt_entopies is not None:
        auc_array_pentropy = []
        for i in range(predictions.shape[-1]):
            auc = run_misclassification_detection(misclassification[:, i],prompt_entopies[:, i])
            auc_array_pentropy.append(auc)
            auc_pentropy= np.stack(auc_array_pentropy, axis=0)
            auc_pentropy_mean, auc_pentropy_std = np.mean(auc_pentropy, axis=0), np.std(auc_pentropy, axis=0)

    if savedir:
        with open(os.path.join(savedir, 'misclassification_detect_individual.txt'), 'w') as f:
            f.write('Mean Accuracy = ' +str(m_accuracy) +'+/-' + str(std_accuract)+'\n')
            f.write('entropy ROC AUC: '+str(auc_entropy_mean[0])+ ' +/- ' + str(auc_entropy_std[0])+ '\n')
            f.write('entropy AUPR POS: '+str(auc_entropy_mean[1])+ ' +/- ' + str(auc_entropy_std[1])+ '\n')
            f.write('entropy AUPR NEG: '+str(auc_entropy_mean[2])+ ' +/- ' + str(auc_entropy_std[2])+ '\n')

            if prompt_entopies is not None:
                f.write('prompt entropy ROC AUC: ' + str(auc_pentropy_mean[0]) + ' +/- ' + str(auc_pentropy_std[0]) + '\n')
                f.write('prompt entropy AUPR POS: ' + str(auc_pentropy_mean[1]) + ' +/ -' + str(auc_pentropy_std[1]) + '\n')
                f.write('prompt entropy AUPR NEG: ' + str(auc_pentropy_mean[2]) + ' +/ -' + str(auc_pentropy_std[2]) + '\n')

    return entropies

def main():
    args = parser.parse_args()

    if args.hatm:
        model_dirs = [os.path.join(args.models_parent_dir, "hatm_seed_{}".format(int(i))) for i in range(1, 11)]
    else:
        model_dirs = [os.path.join(args.models_parent_dir, "atm_seed_{}".format(int(i))) for i in range(1, 11)]

    labels, ensemble_predictions = get_ensemble_predictions(model_dirs, rel_labels_filepath=args.rel_labels_path)


    if args.hatm:
        prompt_entropy_mean, \
        prompt_mean_entropy, \
        prompt_mutual_information, \
        prompt_entropies = get_ensemble_prompt_entropies(model_dirs, rel_labels_filepath=args.rel_labels_path)

    if args.hatm:
        entropies=run_misclassification_detection_over_ensemble(labels, ensemble_predictions, prompt_entropies, savedir=args.savedir)
    else:
        entropies=run_misclassification_detection_over_ensemble(labels, ensemble_predictions, savedir=args.savedir)

    avg_predictions = calc_avg_predictions(ensemble_predictions)

    mutual_information, entropy_of_avg = calc_mutual_information(ensemble_predictions)

    assert np.all(mutual_information >= 0.)

    probilities = add_off_topic_probability(avg_predictions)

    predicted_labels = np.argmax(probilities, axis=1)

    misclassification = np.asarray(labels != predicted_labels, dtype= np.int32)
    correct = np.asarray(labels == predicted_labels, dtype= np.float32)

    accuracy = np.mean(correct)
    aucs_entropy = run_misclassification_detection(misclassification, entropy_of_avg)
    aucs_mi = run_misclassification_detection(misclassification, mutual_information)
    with open(os.path.join(args.savedir, 'misclassification_detect_ensemble.txt'), 'w') as f:
        f.write('Mean Accuracy = ' + str(accuracy) +'\n')
        f.write('entropy ROC AUC: ' + str(aucs_entropy[0]) + '\n')
        f.write('entropy AUPR POS: ' + str(aucs_entropy[1]) + '\n')
        f.write('entropy AUPR NEG: ' + str(aucs_entropy[2]) + '\n')

        f.write('mutual information ROC AUC: ' + str(aucs_mi[0]) + '\n')
        f.write('mutual information AUPR POS: ' + str(aucs_mi[1]) + '\n')
        f.write('mutual information AUPR NEG: ' + str(aucs_mi[2]) + '\n')

        if args.hatm:
            aucs_pentropy = run_misclassification_detection(misclassification, prompt_entropy_mean)
            aucs_pmi = run_misclassification_detection(misclassification, prompt_mutual_information)

            f.write('prompt entropy ROC AUC: ' + str(aucs_pentropy[0]) + '\n')
            f.write('prompt entropy AUPR POS: ' + str(aucs_pentropy[1]) + '\n')
            f.write('prompt entropy AUPR NEG: ' + str(aucs_pentropy[2]) + '\n')

            f.write('prompt mutual information ROC AUC: ' + str(aucs_pmi[0]) + '\n')
            f.write('prompt mutual information AUPR POS: ' + str(aucs_pmi[1]) + '\n')
            f.write('prompt mutual information AUPR NEG: ' + str(aucs_pmi[2]) + '\n')




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
    # Make mutual_infromation histogram plot:
    plot_spread_histogram(correct, incorrect, mutual_information, n_bins=40, spread_name='mutual information')
    plt.savefig(savedir + '/mutual_info_histogram.png', bbox_inches='tight')
    plt.clf()

    plot_spread_histogram(correct, incorrect, entropy_of_avg, n_bins=40, spread_name='entropy')
    plt.savefig(savedir + '/entropy_histogram.png', bbox_inches='tight')
    plt.clf()

    # Make mutual_infromation histogram plot:
    if args.hatm:
        plot_spread_histogram(correct, incorrect, prompt_mutual_information, n_bins=40,
                              spread_name='prompt_mutual information')
        plt.savefig(savedir + '/prompt_mutual_info_histogram.png', bbox_inches='tight')
        plt.clf()

        plot_spread_histogram(correct, incorrect, prompt_entropy_mean, n_bins=40,
                              spread_name='prompt_entropy_mean')
        plt.savefig(savedir + '/prompt_entropy_mean.png', bbox_inches='tight')
        plt.clf()

        plot_spread_histogram(correct, incorrect, prompt_mean_entropy, n_bins=40,
                              spread_name='prompt_mean_entropy')
        plt.savefig(savedir + '/prompt_mean_entropy.png', bbox_inches='tight')
        plt.clf()

    # DENSITY Plots
    # All  examples
    sns.kdeplot(mutual_information, mean_target_deviation, cbar=True, n_levels=args.n_levels, cmap='Purples',
                shade_lowest=False, shade=True)
    plt.ylim(0.0, 1.0)
    plt.xlim(0.0, 0.6)
    plt.xlabel("Mutual Information")
    plt.ylabel("Deviation of average ensemble prediction from label")
    plt.savefig(savedir + '/mutual_information_vs_mean_density.png', bbox_inches='tight')
    plt.clf()

    # On-Topic
    sns.kdeplot(np.extract(labels.astype(np.bool), mutual_information),
                np.extract(labels.astype(np.bool), mean_target_deviation), cbar=True, n_levels=args.n_levels,
                shade_lowest=False, cmap='Blues', shade=True)
    plt.ylim(0.0, 1.0)
    plt.xlim(0.0, 0.6)
    plt.xlabel("Mutual Information")
    plt.ylabel("Deviation of average ensemble prediction from label")
    plt.savefig(savedir + '/mutual_information_vs_mean_density_positive.png', bbox_inches='tight')
    plt.clf()

    # Off-Topic
    sns.kdeplot(np.extract(np.invert(labels.astype(np.bool)), mutual_information),
                np.extract(np.invert(labels.astype(np.bool)), mean_target_deviation), cbar=True, n_levels=args.n_levels,
                shade_lowest=False, cmap="Reds", shade=True)
    plt.ylim(0.0, 1.0)
    plt.xlim(0.0, 0.6)
    plt.xlabel("Mutual Information")
    plt.ylabel("Deviation of average ensemble prediction from label")
    plt.savefig(savedir + '/mutual_information_vs_mean_density_negative.png', bbox_inches='tight')
    plt.clf()

    sns.kdeplot(entropy_of_avg, mean_target_deviation, cbar=True, n_levels=args.n_levels, cmap='Purples',
                shade_lowest=False, shade=True)
    plt.ylim(0.0, 1.0)
    plt.xlim(0.0, 1.0)
    plt.xlabel("Entropy of Mean Prediction")
    plt.ylabel("Deviation of average ensemble prediction from label")
    plt.savefig(savedir + '/entropy_vs_mean_density.png', bbox_inches='tight')
    plt.clf()

    # On-Topic
    sns.kdeplot(np.extract(labels.astype(np.bool), entropy_of_avg),
                np.extract(labels.astype(np.bool), mean_target_deviation), cbar=True, n_levels=args.n_levels,
                shade_lowest=False, cmap='Blues', shade=True)
    plt.ylim(0.0, 1.0)
    plt.xlim(0.0, 1.0)
    plt.xlabel("Entropy of Mean Prediction")
    plt.ylabel("Deviation of average ensemble prediction from label")
    plt.savefig(savedir + '/entropy_vs_mean_density_positive.png', bbox_inches='tight')
    plt.clf()

    # Off-Topic
    sns.kdeplot(np.extract(np.invert(labels.astype(np.bool)), entropy_of_avg),
                np.extract(np.invert(labels.astype(np.bool)), mean_target_deviation), cbar=True, n_levels=args.n_levels,
                shade_lowest=False, cmap="Reds", shade=True)
    plt.ylim(0.0, 1.0)
    plt.xlim(0.0, 1.0)
    plt.xlabel("Entropy of Mean Prediction")
    plt.ylabel("Deviation of average ensemble prediction from label")
    plt.savefig(savedir + 'entropy_vs_mean_density_negative.png', bbox_inches='tight')
    plt.clf()

    if args.hatm:
        sns.kdeplot(prompt_entropy_mean, mean_target_deviation, cbar=True, n_levels=args.n_levels, cmap='Purples',
                    shade_lowest=False, shade=True)
        plt.ylim(0.0, 1.0)
        plt.xlim(0.0, 0.6)
        plt.xlabel('Prompt Entropy')
        plt.ylabel("Deviation of average ensemble prediction from label")
        plt.savefig(savedir + '/prompt_entropy_vs_mean_density.png', bbox_inches='tight')
        plt.clf()

        # On-Topic
        sns.kdeplot(np.extract(labels.astype(np.bool), prompt_entropy_mean),
                    np.extract(labels.astype(np.bool), mean_target_deviation), cbar=True, n_levels=args.n_levels,
                    shade_lowest=False, cmap='Blues', shade=True)
        plt.ylim(0.0, 1.0)
        plt.xlim(0.0, 0.6)
        plt.xlabel('Prompt Entropy')
        plt.ylabel("Deviation of average ensemble prediction from label")
        plt.savefig(savedir + '/prompt_entropy_vs_mean_density_positive.png', bbox_inches='tight')
        plt.clf()

        # Off-Topic
        sns.kdeplot(np.extract(np.invert(labels.astype(np.bool)), prompt_entropy_mean),
                    np.extract(np.invert(labels.astype(np.bool)), mean_target_deviation), cbar=True,
                    n_levels=args.n_levels,
                    shade_lowest=False, cmap="Reds", shade=True)
        plt.ylim(0.0, 1.0)
        plt.xlim(0.0, 0.6)
        plt.xlabel('Prompt Entropy')
        plt.ylabel("Deviation of average ensemble prediction from label")
        plt.savefig(savedir + '/prompt_entropy_vs_mean_density_negative.png', bbox_inches='tight')
        plt.clf()

        sns.kdeplot(prompt_mutual_information, mean_target_deviation, cbar=True, n_levels=args.n_levels, cmap='Purples',
                    shade_lowest=False, shade=True)
        plt.ylim(0.0, 1.0)
        plt.xlim(0.0, 0.2)
        plt.xlabel('Prompt Mutual Information')
        plt.ylabel("Deviation of average ensemble prediction from label")
        plt.savefig(savedir + '/prompt_mutual_information_vs_mean_density.png', bbox_inches='tight')
        plt.clf()

        # On-Topic
        sns.kdeplot(np.extract(labels.astype(np.bool), prompt_mutual_information),
                    np.extract(labels.astype(np.bool), mean_target_deviation), cbar=True, n_levels=args.n_levels,
                    shade_lowest=False, cmap='Blues', shade=True)
        plt.ylim(0.0, 1.0)
        plt.xlim(0.0, 0.2)
        plt.xlabel('Prompt Mutual Information')
        plt.ylabel("Deviation of average ensemble prediction from label")
        plt.savefig(savedir + '/prompt_mutual_information_vs_mean_density_positive.png', bbox_inches='tight')
        plt.clf()

        # Off-Topic
        sns.kdeplot(np.extract(np.invert(labels.astype(np.bool)), prompt_mutual_information),
                    np.extract(np.invert(labels.astype(np.bool)), mean_target_deviation), cbar=True,
                    n_levels=args.n_levels,
                    shade_lowest=False, cmap="Reds", shade=True)
        plt.ylim(0.0, 1.0)
        plt.xlim(0.0, 0.2)
        plt.xlabel('Prompt Mutual Information')
        plt.ylabel("Deviation of average ensemble prediction from label")
        plt.savefig(savedir + '/prompt_mutual_information_vs_mean_density_negative.png', bbox_inches='tight')
        plt.clf()



    #   AUC vs. CUMULATIVE INCLUDED

    # Make AUC vs. cumulative samples included by range spread
    plot_auc_vs_percentage_included(labels, avg_predictions, mutual_information, resolution=200,
                                    sort_by_name='mutual information', savedir=args.savedir)
    plot_auc_vs_percentage_included(labels, avg_predictions, entropy_of_avg, resolution=200,
                                    sort_by_name='entropy', savedir=args.savedir)
    if args.hatm:
        plot_auc_vs_percentage_included(labels, avg_predictions, prompt_entropy_mean, resolution=200,
                                        sort_by_name='prompt_entropy', savedir=args.savedir)


    if args.hatm:
        plt.legend(['Mutual Information', 'Entropy', 'Prompt Entropy'])
    else:
        plt.legend(['Mutual Information', 'Entropy'])
    plt.savefig(savedir + '/auc_vs_cumulative_samples.png', bbox_inches='tight')
    plt.clf()

    if args.hatm:
        plot_auc_vs_percentage_included_ensemble(labels, ensemble_predictions, entropies, resolution=100,
                                                 sort_by_name='entropy', savedir=args.savedir)
        plot_auc_vs_percentage_included_ensemble(labels, ensemble_predictions, prompt_entropies, resolution=100,
                                                 sort_by_name='prompt_entropy', savedir=args.savedir)
        plt.legend(['Entropy', 'Prompt Entropy'])
        plt.savefig(savedir + '/auc_vs_cumulative_samples_ensemble.png', bbox_inches='tight')
        plt.clf()

        for i in range(2):
            plot_aupr_vs_percentage_included_ensemble(labels, ensemble_predictions, entropies, resolution=100,
                                                     pos_label=i, savedir=args.savedir)
            plot_aupr_vs_percentage_included_ensemble(labels, ensemble_predictions, prompt_entropies, resolution=100,
                                                    pos_label=i, savedir=args.savedir)
            plt.legend(['Entropy', 'Prompt Entropy'])
            plt.savefig(savedir + '/aupr_vs_cumulative_samples_ensemble_pos_label'+str(i)+'.png', bbox_inches='tight')
            plt.clf()
    else:
        plot_auc_vs_percentage_included_ensemble(labels, ensemble_predictions, entropies, resolution=100,
                                                 sort_by_name='entropy', savedir=args.savedir)
        plt.legend(['Entropy'])
        plt.savefig(savedir + '/auc_vs_cumulative_samples_ensemble.png', bbox_inches='tight')
        plt.clf()
        for i in range(2):
            plot_aupr_vs_percentage_included_ensemble(labels, ensemble_predictions, entropies, resolution=100,
                                                      pos_label=i, savedir=args.savedir)
            plt.legend(['Entropy'])
            plt.savefig(savedir + '/aupr_vs_cumulative_samples_ensemble_pos_label' + str(i) + '.png', bbox_inches='tight')
            plt.clf()


    # Plot AUC of PR curves
    for pos_label in range(2):
        plot_aupr_vs_percentage_included(labels, avg_predictions, mutual_information, resolution=200, pos_label=pos_label,savedir=args.savedir)
        plot_aupr_vs_percentage_included(labels, avg_predictions, entropy_of_avg, resolution=200, pos_label=pos_label, savedir=args.savedir)
        if args.hatm:
            plot_aupr_vs_percentage_included(labels, avg_predictions, prompt_entropy_mean, resolution=200, pos_label=pos_label, savedir=args.savedir)
        if args.hatm:
            plt.legend(['Mutual Information', 'Entropy', 'Prompt Entropy'])
        else:
            plt.legend(['Mutual Information', 'Entropy'])
        plt.savefig(savedir + '/aupr_vs_cumulative_samples_pos_label'+str(pos_label)+'.png', bbox_inches='tight')
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
    # Make precision recall 3D plot vs mutual information
    plot_pr_spread_mesh(labels, avg_predictions, mutual_information, pos_labels=0)
    plt.savefig(savedir + '/ensemble_3d_pr_curve_mutual_info.png')
    plt.clf()

    # Make family of curves precision recall plot vs. mutual information
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
