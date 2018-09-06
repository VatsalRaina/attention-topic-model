"""
Just a one-off script for comparing the seed spread versus confidence
"""
from __future__ import print_function, division

import sys
import os
import numpy as np
import scipy.stats
import math
import time
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
import matplotlib.patches as mpatches

from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

parser = argparse.ArgumentParser(description='Plot useful graphs for evaluation.')
parser.add_argument('--savedir', type=str, default='./',
                    help='Path to directory where to save the plots')
parser.add_argument('--rel_labels_path_seen',  type=str, default='eval4_naive/labels-probs.txt')
parser.add_argument('--rel_labels_path_unseen', type=str, default='linsk_eval03/labels-probs.txt')

matplotlib.rcParams['savefig.dpi'] = 200

# Specify the colours
green = (0.3, 0.9, 0.3)
dark_green = (0.1, 0.7, 0.3)
red = (0.9, 0.3, 0.3)
dark_orange = (0.9, 0.5, 0.2)
dark_blue = (0.14, 0.29, 0.36)


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
    return mutual_information, entropy_of_expected, expected_entropy


def calc_confusion_matrix(y_true, y_pred, thresh):
    y_class_pred = (y_pred >= thresh)
    y_true = y_true.astype(np.bool)
    y_class_pred_inv = np.logical_not(y_class_pred)
    y_true_inv = np.logical_not(y_true)
    tp = np.sum(np.logical_and(y_class_pred, y_true))
    fp = np.sum(np.logical_and(y_class_pred, y_true_inv))
    fn = np.sum(np.logical_and(y_class_pred_inv, y_true))
    tn = np.sum(np.logical_and(y_class_pred_inv, y_true_inv))
    conf_mat = np.array([[tp, fp], [fn, tn]])
    return conf_mat


def calc_recall(conf_mat):
    tp = conf_mat[0, 0]
    fn = conf_mat[1, 0]
    if tp + fn == 0:
        return np.nan
    else:
        return tp / (tp + fn)


def calc_precision(conf_mat):
    tp = conf_mat[0, 0]
    fp = conf_mat[0, 1]
    if tp + fp == 0:
        return np.nan
    else:
        return tp / (tp + fp)


def plot_datasets_histogram(spread_seen, spread_unseen, save_dir, n_bins=60, spread_name='mutual information'):
    color_seen, color_unseen = (0.4, 0.6, 0.8), (0.4, 0.8, 0.6)
    plt.hist((spread_seen, spread_unseen), n_bins, density=False,
             histtype='bar', stacked=True, color=[color_seen, color_unseen])
    plt.xlabel("Spread (" + spread_name + " of ensemble predictions)")
    plt.ylabel("Example Count")
    seen_patch = mpatches.Patch(color=color_seen, label='Seen - seen')
    unseen_patch = mpatches.Patch(color=color_unseen, label='Unseen - unseen')
    plt.legend(handles=[seen_patch, unseen_patch], loc="upper right")
    plt.savefig(os.path.join(save_dir, "datasets_spread_histogram.png"), bbox_inches='tight')
    plt.close()
    return


def plot_precision_recall_balance_legacy(labels_seen, predictions_seen, labels_unseen, predictions_unseen, save_dir,
                                  resolution=100):
    # Plot the normal, joint PR curve
    plt.figure(1)
    precision, recall, thresholds = precision_recall_curve(np.hstack((labels_seen, labels_unseen)),
                                                           np.hstack((predictions_seen, predictions_unseen)))
    # Get rid of the last values which are set manually by the sklrean algorithm
    precision = precision[:-1]
    recall = recall[:-1]
    plt.plot(recall, precision, color=dark_blue)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim(0, 1)
    plt.savefig(os.path.join(save_dir, "total_pr.png"), bbox_inches='tight')
    print("Made first plot")

    # Optimise the number of thresholds to plot
    # First sort by recall value
    sort_idx = np.argsort(recall)
    recall = recall[sort_idx]
    thresholds = thresholds[sort_idx]
    x = np.linspace(np.min(recall), np.max(recall), resolution, endpoint=False)
    select = np.empty(recall.shape, dtype=np.bool)
    select[:] = False
    select[-1] = True
    for i in range(len(x)):
        # Get only the recall values that approximate a grid with a particular resolution
        next_select_idx = np.argmax(recall >= x[i])
        if next_select_idx >= len(select):
            break
        select[next_select_idx] = True
    recall_opt = np.extract(select, recall)
    thresholds_opt = np.extract(select, thresholds)
    print("Number thresholds to consider optimised.")

    # Plot the recall on dataset vs. overall recall plot.
    plt.figure(2)
    recall_seen = np.empty_like(recall_opt)
    recall_unseen = np.empty_like(recall_opt)
    precision_seen = np.empty_like(recall_opt)
    precision_unseen = np.empty_like(recall_opt)

    for i in range(len(thresholds_opt)):
        conf_mat_seen = calc_confusion_matrix(labels_seen, predictions_seen, thresholds_opt[i])
        conf_mat_unseen = calc_confusion_matrix(labels_unseen, predictions_unseen, thresholds_opt[i])
        recall_seen[i] = calc_recall(conf_mat_seen)
        recall_unseen[i] = calc_recall(conf_mat_unseen)
        precision_seen[i] = calc_precision(conf_mat_seen)
        precision_unseen[i] = calc_precision(conf_mat_unseen)
    marker_size = 1.1
    plt.plot([0, 1], [0, 1], linewidth=0.8, alpha=0.5, color='black')
    plt.plot(recall_opt, recall_seen, color=dark_orange, label='Seen - Seen')
    plt.plot(recall_opt, recall_unseen, color=dark_blue, label='Unseen - Unseen')
    plt.scatter(recall_opt, recall_seen, color=red, marker='o', s=marker_size, alpha=0.7)
    plt.scatter(recall_opt, recall_unseen, color='black', marker='o', s=marker_size, alpha=0.7)
    plt.xlabel("Total Recall")
    plt.ylabel("Subset Recall")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(title="Dataset:", loc='lower right')

    plt.savefig(os.path.join(save_dir, "subset_recall_v_total_recall.png"), bbox_inches='tight')

    print("Made second plot.")
    # Plot the PR curve for each individual dataset
    plt.figure(3)
    plt.plot(recall_opt, precision_seen, color=dark_orange, label='Seen - Seen')
    plt.plot(recall_opt, precision_unseen, color=dark_blue, label='Unseen - Unseen')
    plt.scatter(recall_opt, precision_seen, color=red, marker='o', s=marker_size, alpha=0.7)
    plt.scatter(recall_opt, precision_unseen, color='black', marker='o', s=marker_size, alpha=0.7)
    plt.xlabel("Total Recall")
    plt.ylabel("Subset Precision")
    plt.xlim(0, 1)
    plt.legend(title="Dataset:", loc='lower left')

    plt.savefig(os.path.join(save_dir, "subset_pr.png"), bbox_inches='tight')
    plt.close()
    return


def plot_precision_recall_balance(labels_seen, predictions_seen, labels_unseen, predictions_unseen, save_dir,
                                  num_thresh=500):

    labels_total = np.hstack(labels_seen, labels_unseen)
    predictions_total = np.hstack(predictions_seen, predictions_unseen)

    # Plot the normal, joint PR curve
    plt.figure(1)
    thresholds = np.linspace(0, 1, num=num_thresh)

    recall_seen = np.empty_like(thresholds)
    recall_unseen = np.empty_like(thresholds)
    recall_total = np.empty_like(thresholds)
    precision_seen = np.empty_like(thresholds)
    precision_unseen = np.empty_like(thresholds)
    precision_total = np.empty_like(thresholds)
    for i in range(len(thresholds)):
        conf_mat_seen = calc_confusion_matrix(labels_seen, predictions_seen, thresholds[i])
        conf_mat_unseen = calc_confusion_matrix(labels_unseen, predictions_unseen, thresholds[i])
        conf_mat_total = calc_confusion_matrix(labels_total, predictions_total, thresholds[i])
        recall_seen[i] = calc_recall(conf_mat_seen)
        recall_unseen[i] = calc_recall(conf_mat_unseen)
        recall_total[i] = calc_recall(conf_mat_total)
        precision_seen[i] = calc_precision(conf_mat_seen)
        precision_unseen[i] = calc_precision(conf_mat_unseen)
        precision_total[i] = calc_precision(conf_mat_total)

    plt.plot(recall_total, precision_total, color=dark_blue)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim(0, 1)
    plt.savefig(os.path.join(save_dir, "total_pr.png"), bbox_inches='tight')
    print("Made first plot")

    # Plot the recall on dataset vs. overall recall plot.
    plt.figure(2)

    marker_size = 1.1
    plt.plot([0, 1], [0, 1], linewidth=0.8, alpha=0.5, color='black')
    plt.plot(recall_total, recall_seen, color=dark_orange, label='Seen - Seen')
    plt.plot(recall_total, recall_unseen, color=dark_blue, label='Unseen - Unseen')
    plt.scatter(recall_total, recall_seen, color=red, marker='o', s=marker_size, alpha=0.7)
    plt.scatter(recall_total, recall_unseen, color='black', marker='o', s=marker_size, alpha=0.7)
    plt.xlabel("Total Recall")
    plt.ylabel("Subset Recall")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(title="Dataset:", loc='lower right')

    plt.savefig(os.path.join(save_dir, "subset_recall_v_total_recall.png"), bbox_inches='tight')

    print("Made second plot.")
    # Plot the PR curve for each individual dataset
    plt.figure(3)
    # plt.plot(recall_total, precision_seen, color=dark_orange, label='Seen - Seen')
    # plt.plot(recall_total, precision_unseen, color=dark_blue, label='Unseen - Unseen')
    plt.scatter(recall_total, precision_seen, color=red, marker='o', s=marker_size, alpha=0.7)
    plt.scatter(recall_total, precision_unseen, color='black', marker='o', s=marker_size, alpha=0.7)
    plt.xlabel("Total Recall")
    plt.ylabel("Subset Precision")
    plt.xlim(0, 1)
    plt.legend(title="Dataset:", loc='lower left')

    plt.savefig(os.path.join(save_dir, "subset_pr.png"), bbox_inches='tight')
    plt.close()
    return


def calc_metrics(labels, ensemble_predictions):
    avg_predictions = calc_avg_predictions(ensemble_predictions)

    std_spread = np.std(ensemble_predictions, axis=1)
    range_spread = np.ptp(ensemble_predictions, axis=1)
    iqr_spread = scipy.stats.iqr(ensemble_predictions, axis=1)  # interquartile range (IQR)
    mutual_information, entropy_of_avg, avg_entropy = calc_mutual_information(ensemble_predictions)

    mean_target_deviation = np.abs(labels - avg_predictions)

    correct = mean_target_deviation < 0.5
    incorrect = np.invert(correct)

    metrics = {"avg_predictions": avg_predictions,
               "std_spread": std_spread,
               "range_spread": range_spread,
               "iqr_spread": iqr_spread,
               "mutual_information": mutual_information,
               "entropy_of_avg": entropy_of_avg,
               "avg_entropy": avg_entropy,
               "mean_target_deviation": mean_target_deviation,
               "correct": correct,
               "incorrect": incorrect}
    return metrics


def main():
    args = parser.parse_args()

    models_parent_dir = '/home/miproj/urop.2018/bkm28/seed_experiments'
    model_dirs = [os.path.join(models_parent_dir, "atm_seed_{}".format(int(i))) for i in range(1, 11)]

    # Below seen refers to 'seen-seen' and unseen refers to 'unseen-unseen' examples.
    labels_seen, ensemble_pred_seen = get_ensemble_predictions(model_dirs,
                                                               rel_labels_filepath=args.rel_labels_path_seen)
    labels_unseen, ensemble_pred_unseen = get_ensemble_predictions(model_dirs,
                                                                   rel_labels_filepath=args.rel_labels_path_unseen)
    print("Label Data Loaded.")

    # Invert everything so that off-topic is 1:
    labels_seen, ensemble_pred_seen, labels_unseen, ensemble_pred_unseen = map(lambda x: 1 - x,
                                                                               [labels_seen, ensemble_pred_seen,
                                                                                labels_unseen, ensemble_pred_unseen])
    metrics_seen = calc_metrics(labels_seen, ensemble_pred_seen)
    metrics_unseen = calc_metrics(labels_unseen, ensemble_pred_unseen)

    print("Metrics calculated")
    print("Ratio seen-seen dataset to unseen-unseen dataset:", float(len(labels_seen)) / float(len(labels_unseen) + len(labels_seen)))

    # Make the plots:
    save_dir = args.savedir
    start_time = time.time()
    plot_precision_recall_balance(labels_seen, metrics_seen['avg_predictions'], labels_unseen,
                                  metrics_unseen['avg_predictions'], save_dir)
    print("Made triple PR plot with subset split. Time taken: ", time.time() - start_time)
    plot_precision_recall_balance_legacy(labels_seen, metrics_seen['avg_predictions'], labels_unseen,
                                  metrics_unseen['avg_predictions'], save_dir + '/legacy')
    print("Made triple PR plot with subset split legacy. Time taken: ", time.time() - start_time)
    return


if __name__ == "__main__":
    main()
