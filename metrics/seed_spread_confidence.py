"""
Just a one-off script for comparing the seed spread versus confidence
"""

import sys
import os
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from __future__ import print_function, division


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
    all_predictions = np.stack(all_predictions, axis=1)
    return labels, all_predictions


def get_label_predictions(labels_filepath):
    labels = []
    predictions = []
    with open(labels_filepath, "r") as file:
        for line in file.readlines():
            single_example = line.strip().split()
            label = int(single_example[0])
            labels.append(label)
            prediction = float(single_example[1])
            predictions.append(prediction)
    labels_array = np.array(labels, dtype=np.float32)
    predictions_array = np.array(predictions, dtype=np.float32)
    return labels_array, predictions_array


def calc_avg_predictions(ensemble_predictions):
    avg_predictions = np.sum(ensemble_predictions, axis=1)
    avg_predictions /= len(ensemble_predictions)
    return avg_predictions


def main():
    models_parent_dir = '/home/miproj/urop.2018/bkm28/seed_experiments'
    model_dirs = [os.path.join(models_parent_dir, f"atm_seed_{int(i)}") for i in range(1, 11)]

    labels, ensemble_predictions = get_ensemble_predictions(model_dirs)
    avg_predictions = calc_avg_predictions(ensemble_predictions)

    std_spread = np.std(ensemble_predictions, axis=1)
    range_spread = np.ptp(ensemble_predictions, axis=1)
    iqr_spread = scipy.stats.iqr(ensemble_predictions, axis=1)  # interquartile range (IQR)

    mean_target_deviation = np.abs(labels - avg_predictions)

    correct = mean_target_deviation < 0.5
    incorrect = np.invert(correct)


    # Make the plots:
    n_bins = 20
    histogram_data = np.hstack()
    plt.hist((np.where(correct, std_spread), np.where(incorrect, std_spread)), n_bins, density=True,
             histtype='bar', stacked=True, color=['green', 'red'])
    plt.set_title('std bar')
    plt.savefig('~/stdbarchart.png')
    plt.clear()

    plt.scatter(range_spread, mean_target_deviation)
    plt.savefig('~/coollinechart.png')
    return

