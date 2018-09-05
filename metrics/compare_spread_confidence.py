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
    return mutual_information, entropy_of_expected, expected_entropy


def plot_precision_recall_balance(labels_seen, predictions_seen, labels_unseen, predictions_unseen, save_dir):
    # Plot the normal, joint PR curve
    plt.figure(1)
    precision, recall, thresholds = precision_recall_curve(np.hstack((labels_seen, labels_unseen)),
                                                           np.hstack((predictions_seen, predictions_unseen)))
    plt.plot(recall, precision, color=dark_blue)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim(0, 1)
    plt.savefig(os.path.join(save_dir, "total_pr.png"), bbox_inches='tight')

    # Plot the recall on dataset vs. overall recall plot.
    plt.figure(2)
    recall_seen = np.empty_like(recall)
    recall_unseen = np.empty_like(recall)
    precision_seen = np.empty_like(recall)
    precision_unseen = np.empty_like(recall)

    for i in range(len(thresholds)):
        class_predicted_seen = (predictions_seen > thresholds[i]).astype(np.int16)
        class_predicted_unseen = (predictions_unseen > thresholds[i]).astype(np.int16)
        recall_seen[i] = recall_score(labels_seen, class_predicted_seen, average='binary')
        recall_unseen[i] = recall_score(labels_unseen, class_predicted_unseen, average='binary')
        precision_seen[i] = precision_score(labels_seen, class_predicted_seen, average='binary')
        precision_unseen[i] = precision_score(labels_unseen, class_predicted_unseen, average='binary')
    plt.plot(recall, recall_seen, color=dark_orange, label='Seen - Seen')
    plt.plot(recall, recall_unseen, color=dark_blue, label='Unseen - Unseen')
    plt.xlabel("Total Recall")
    plt.ylabel("Subset Recall")
    plt.xlim(0, 1)
    plt.legend(title="Dataset:", loc='upper left')

    plt.savefig(os.path.join(save_dir, "subset_recall_v_total_recall.png"), bbox_inches='tight')

    # Plot the PR curve for each individual dataset
    plt.figure(3)
    plt.plot(recall, precision_seen, color=dark_orange, label='Seen - Seen Dataset')
    plt.plot(recall, precision_unseen, color=dark_blue, label='Unseen - Unseen Dataset')
    plt.xlabel("Total Recall")
    plt.ylabel("Subset Precision")
    plt.xlim(0, 1)
    plt.legend(title="Dataset:", loc='bottom left')

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

    # Calculate the true_positives, true_negatives .. e.t.c.
    # Define POSITIVE as OFF TOPIC
    tp = np.logical_and(correct, avg_predictions < 0.5)
    tn = np.logical_and(correct, avg_predictions >= 0.5)
    fp = np.logical_and(incorrect, avg_predictions < 0.5)
    fn = np.logical_and(incorrect, avg_predictions >= 0.5)

    metrics = {"avg_predictions": avg_predictions,
               "std_spread": std_spread,
               "range_spread": range_spread,
               "iqr_spread": iqr_spread,
               "mutual_information": mutual_information,
               "entropy_of_avg": entropy_of_avg,
               "avg_entropy": avg_entropy,
               "mean_target_deviation": mean_target_deviation,
               "correct": correct,
               "incorrect": incorrect,
               "tp": tp, "tn": tn, "fp": fp, "fn": fn}
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
    metrics_seen = calc_metrics(labels_seen, ensemble_pred_seen)
    metrics_unseen = calc_metrics(labels_unseen, ensemble_pred_unseen)

    print("Metrics calculated")

    # Make the plots:
    save_dir = args.savedir
    start_time = time.time()
    plot_precision_recall_balance(labels_seen, metrics_seen['avg_predictions'], labels_unseen,
                                  metrics_unseen['avg_predictions'], save_dir)
    print("Made triple PR plot with subset split. Time taken: ", time.time() - start_time)
    return


if __name__ == "__main__":
    main()
