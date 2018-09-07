"""
Just a one-off script for comparing the seed spread versus confidence
"""
from __future__ import print_function, division

import sys
import os
import numpy as np
from numpy import ma
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
parser.add_argument('--rel_labels_path_seen', type=str, default='eval4_naive/labels-probs.txt')
parser.add_argument('--rel_labels_path_unseen', type=str, default='linsk_eval03/labels-probs.txt')

matplotlib.rcParams['savefig.dpi'] = 200

# Specify the colours
green = (0.3, 0.9, 0.3)
dark_green = (0.1, 0.7, 0.3)
red = (0.9, 0.3, 0.3)
dark_orange = (0.9, 0.5, 0.2)
dark_blue = (0.14, 0.29, 0.36)


def calc_cum_density(predictions, resolution=200):
    num_examples = len(predictions)
    counts, bin_edges = np.histogram(predictions, bins=resolution, range=(.0, 1.0), density=False)
    prob_mass_function = counts.astype(np.float32) / num_examples
    cum_density = np.cumsum(prob_mass_function)
    cum_density = np.insert(cum_density, 0, 0.0) # Insert the value of 0.0 at x = 0.0
    return cum_density, bin_edges


# def plot_cum_density_signle(predictions, color=resolution=200)


def plot_cum_density_family(predictions, labels, spread, spread_thresholds):

    set_dense_gridlines()
    # Set the color maps
    cool = plt.get_cmap('cool')
    jet = plt.get_cmap('summer')
    c_norm = matplotlib.colors.Normalize(vmin=0, vmax=len(spread_thresholds) + 1)
    colours_on_topic = get_colour_list(len(spread_thresholds) + 1, 'summer')
    colours_off_topic = get_colour_list(len(spread_thresholds) + 1, 'cool')

    # Plot unthresholded data:
    pred_on_topic = np.extract(labels == 0, predictions)
    pred_off_topic = np.extract(labels == 1, predictions)

    cum_dens_on_topic, x = calc_cum_density(pred_on_topic)
    cum_dens_off_topic, _ = calc_cum_density(pred_off_topic)

    plt.plot(x, cum_dens_on_topic, color=colours_on_topic[-1], linewidth=0.6, alpha=0.5,
             label='No-thresh on-topic')
    plt.plot(x, cum_dens_off_topic, color=colours_off_topic[-1], linewidth=0.6, alpha=0.5,
             label='No-thresh off-topic')

    for i in range(len(spread_thresholds)):
        threshold = spread_thresholds[i]
        thresholded_idx = spread <= threshold
        # Extract the predictions and labels by spread threshold
        pred_thresholded, labels_thresholded = predictions[thresholded_idx], labels[thresholded_idx]

        # Extract the ontopic and off-topic predictions
        pred_on_topic = np.extract(labels_thresholded == 0, pred_thresholded)
        pred_off_topic = np.extract(labels_thresholded == 1, pred_thresholded)

        cum_dens_on_topic, x = calc_cum_density(pred_on_topic)
        cum_dens_off_topic, _ = calc_cum_density(pred_off_topic)

        plt.plot(x, cum_dens_on_topic, color=colours_on_topic[i], linewidth=0.6, alpha=0.5,
                 label='{0:.2f} on-topic'.format(threshold))
        plt.plot(x, cum_dens_off_topic, color=colours_off_topic[i], linewidth=0.6, alpha=0.5,
                 label='{0:.2f} off-topic'.format(threshold))

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Expected probability of off-topic as predicted by ensemble.")
    plt.ylabel("Cumulative Density")
    # Sort the legend first by on vs. off topic, than by threshold
    handles, labels = plt.gca().get_legend_handles_labels()
    # labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: _label_order_key(t[0])))
    handles_seen = [handles[i*2] for i in range(int(len(handles) / 2))]
    handles_unseen = [handles[i*2 + 1] for i in range(int(len(handles) / 2))]
    labels_seen = [labels[i*2] for i in range(int(len(handles) / 2))]
    labels_unseen = [labels[i*2 + 1] for i in range(int(len(handles) / 2))]
    handles = handles_seen + handles_unseen
    labels = labels_seen + labels_unseen
    plt.legend(handles, labels, bbox_to_anchor=(1.04, 0), loc="lower left", borderaxespad=0)
    return


def _label_order_key(label):
    label = label.split(label)
    key = (label[1], label[0])
    return ' '.join(key)


def set_dense_gridlines():
    # Don't allow the axis to be on top of your data
    # plt.set_axisbelow(True)

    # Turn on the minor TICKS, which are required for the minor GRID
    plt.minorticks_on()

    # Customize the major grid
    plt.grid(which='major', linestyle='-', linewidth='0.4', color='black', alpha=0.2)
    # Customize the minor grid
    plt.grid(which='minor', linestyle=':', linewidth='0.3', color='black', alpha=0.12)
    return


def get_colour_list(num_colours, cmap_name='viridis'):
    colour_map = plt.get_cmap(cmap_name)
    c_norm = matplotlib.colors.Normalize(vmin=0, vmax=num_colours)
    scalar_map = matplotlib.cm.ScalarMappable(norm=c_norm, cmap=colour_map)
    colour_list = list(map(lambda i: scalar_map.to_rgba(i), range(num_colours)))
    return colour_list


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
    predictions = ma.masked_array(predictions, mask=np.logical_and(predictions == 1., predictions == 0.))
    entropy_masked = -(predictions * np.log(predictions) + (1 - predictions) * np.log(1 - predictions))
    # Convert the values of entropy where np.log gives nan
    entropy = entropy_masked.filled(0.)
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
    plt.hist((spread_seen, spread_unseen), n_bins, density=True,
             histtype='bar', stacked=True, color=[color_seen, color_unseen])
    plt.xlabel("Spread (" + spread_name + " of ensemble predictions)")
    plt.ylabel("Example Count")
    seen_patch = mpatches.Patch(color=color_seen, label='Seen - seen')
    unseen_patch = mpatches.Patch(color=color_unseen, label='Unseen - unseen')
    plt.legend(handles=[seen_patch, unseen_patch], loc="upper right")
    spread_save_name = '_'.join(spread_name.split())
    plt.savefig(os.path.join(save_dir, spread_save_name + "_datasets_hist.png"), bbox_inches='tight')
    plt.clf()
    plt.close()
    return


def plot_precision_recall_balance(labels_seen, predictions_seen, labels_unseen, predictions_unseen, save_dir,
                                  resolution=200, color_seen=dark_blue, color_unseen=dark_green, keep_plots=False):
    # Calculate the precisin recall metrics
    precision_seen, precision_unseen, precision_total, recall_seen, recall_unseen, recall_total = calc_precision_recall_metrics(
        labels_seen, predictions_seen, labels_unseen, predictions_unseen)

    # Plot the normal, joint PR curve
    plt.figure(1)
    plt.plot(recall_total, precision_total, color=color_unseen)  # Doesn't matter which colour actually used
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim(0, 1)
    if not keep_plots:
        plt.savefig(os.path.join(save_dir, "total_pr.png"), bbox_inches='tight')
        plt.clf()

    # Plot the recall on dataset vs. overall recall plot.
    plt.figure(2)
    marker_size = 1.1
    plt.plot([0, 1], [0, 1], linewidth=0.8, alpha=0.5, color='black')
    # plt.plot(recall_opt, recall_seen, color=dark_orange, label='Seen - Seen')
    # plt.plot(recall_opt, recall_unseen, color=dark_blue, label='Unseen - Unseen')
    plt.scatter(recall_total, recall_seen, color=color_seen, marker='o', s=marker_size, alpha=0.7, label='Seen - Seen')
    plt.scatter(recall_total, recall_unseen, color=color_unseen, marker='o', s=marker_size, alpha=0.7, label='Unseen - Unseen')
    plt.xlabel("Total Recall")
    plt.ylabel("Subset Recall")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(title="Dataset:", loc='lower right')

    if not keep_plots:
        plt.savefig(os.path.join(save_dir, "subset_recall_v_total_recall.png"), bbox_inches='tight')
        plt.clf()

    # Plot the PR curve for each individual dataset
    plt.figure(3)
    # plt.plot(recall_opt, precision_seen, color=dark_orange, label='Seen - Seen')
    # plt.plot(recall_opt, precision_unseen, color=dark_blue, label='Unseen - Unseen')
    plt.scatter(recall_total, precision_seen, color=color_seen, marker='o', s=marker_size, alpha=0.7, label='Seen - Seen')
    plt.scatter(recall_total, precision_unseen, color=color_unseen, marker='o', s=marker_size, alpha=0.7,
                label='Unseen - Unseen')
    plt.xlabel("Total Recall")
    plt.ylabel("Subset Precision")
    plt.xlim(0, 1)
    plt.legend(title="Dataset:", loc='lower left')

    if not keep_plots:
        plt.savefig(os.path.join(save_dir, "subset_pr.png"), bbox_inches='tight')
        plt.clf()
        plt.close()
    return


def plot_precision_recall_balance_single(labels_seen, predictions_seen, labels_unseen, predictions_unseen, save_dir,
                                  resolution=200, color_seen=dark_blue, color_unseen=dark_green):
    # Calculate the precisin recall metrics
    precision_seen, precision_unseen, precision_total, recall_seen, recall_unseen, recall_total = calc_precision_recall_metrics(
        labels_seen, predictions_seen, labels_unseen, predictions_unseen)

    # Plot the normal, joint PR curve
    plt.figure(1)
    plt.plot(recall_total, precision_total, color=color_unseen, linewidth=0.8)  # Ignore first value which messes up the plot
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim(0, 1)

    # Plot the recall on dataset vs. overall recall plot.
    plt.figure(2)
    marker_size = 0.5
    plt.plot([0, 1], [0, 1], linewidth=0.8, alpha=0.5, color='black')
    plt.plot(recall_total, recall_seen, color=color_seen, linewidth=0.6, alpha=0.6)
    plt.plot(recall_total, recall_unseen, color=color_unseen, linewidth=0.6, alpha=0.6)
    # plt.scatter(recall_total, recall_seen, color=color_seen, marker='o', s=marker_size, alpha=0.6)
    # plt.scatter(recall_total, recall_unseen, color=color_unseen, marker='o', s=marker_size, alpha=0.6)
    plt.xlabel("Total Recall")
    plt.ylabel("Subset Recall")
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # Plot the PR curve for each individual dataset
    plt.figure(3)
    plt.plot(recall_total, precision_seen, color=color_seen, linewidth=0.6, alpha=0.7)
    plt.plot(recall_total, precision_unseen, color=color_unseen, linewidth=0.6, alpha=0.7)
    # plt.scatter(recall_total, precision_seen, color=color_seen, marker='o', s=marker_size, alpha=0.7)
    # plt.scatter(recall_total, precision_unseen, color=color_unseen, marker='o', s=marker_size, alpha=0.7)
    plt.xlabel("Total Recall")
    plt.ylabel("Subset Precision")
    plt.xlim(0, 1)
    plt.legend(title="Dataset:", loc='lower left')
    return


def _sort_predictions_labels(labels, predictions, sort_by_array):
    num_examples = len(labels)
    sorted_order = np.argsort(sort_by_array)

    labels_sorted = labels[sorted_order]
    predictions_sorted = predictions[sorted_order]
    sort_by_array_sorted = sort_by_array[sorted_order]
    return labels_sorted, predictions_sorted, sort_by_array_sorted


def plot_precision_recall_balance_v_spread(labels_seen, predictions_seen, labels_unseen, predictions_unseen,
                                           spread_seen, spread_unseen, save_dir,
                                           proportions_included=(0.161, 0.4, 0.6, 0.7, 1.0),
                                           resolution=200, spread_name='mutual_info'):

    num_seen = len(labels_seen)
    num_unseen = len(labels_unseen)
    num_total = num_seen + num_unseen
    labels_seen, predictions_seen, spread_seen = _sort_predictions_labels(labels_seen, predictions_seen, spread_seen)
    labels_unseen, predictions_unseen, spread_unseen = _sort_predictions_labels(labels_unseen, predictions_unseen, spread_unseen)
    labels_total, predictions_total, spread_total = _sort_predictions_labels(np.hstack((labels_seen, labels_unseen)),
                                                               np.hstack((predictions_seen, predictions_unseen)),
                                                               np.hstack((spread_seen, spread_unseen)))

    # Set a color map
    viridis = plt.get_cmap('viridis')
    jet = plt.get_cmap('jet')
    c_norm = matplotlib.colors.Normalize(vmin=0, vmax=len(proportions_included))
    scalar_map_seen = matplotlib.cm.ScalarMappable(norm=c_norm, cmap=viridis)
    scalar_map_unseen = matplotlib.cm.ScalarMappable(norm=c_norm, cmap=jet)

    for i in range(len(proportions_included)):
        proportion = proportions_included[i]

        # Extract the last index for each subset
        max_val_idx = int(math.floor(num_total * proportion))
        max_spread = spread_total[min(max_val_idx, num_total - 1)]
        last_idx_seen = np.argmax(spread_seen >= max_spread)
        last_idx_unseen = np.argmax(spread_unseen >= max_spread)

        # Threshold the predictions based on sorted order:
        pred_seen_thresh = predictions_seen.copy()
        pred_unseen_thresh = predictions_unseen.copy()
        if last_idx_seen <= num_seen - 1:
            pred_seen_thresh[last_idx_seen:] = 0.
        if last_idx_unseen <= num_unseen - 1:
            pred_unseen_thresh[last_idx_unseen:] = 0.

        # Get the appriopriate colors
        color_val_seen = scalar_map_seen.to_rgba(i)
        color_val_unseen = scalar_map_unseen.to_rgba(i)

        # Add lines to plot
        plot_precision_recall_balance_single(labels_seen, pred_seen_thresh, labels_unseen, pred_unseen_thresh, save_dir,
                                             color_seen=color_val_seen, color_unseen=color_val_unseen)
        del pred_seen_thresh, pred_unseen_thresh

    # Plot the graphs
    # Create the legend
    seen_patches, unseen_patches, general_patches = [], [], []
    for i in range(len(proportions_included)):
        proportion = proportions_included[i]
        seen_patch = mpatches.Patch(color=scalar_map_seen.to_rgba(i), label='{0:.2f} seen-seen'.format(proportion))
        unseen_patch = mpatches.Patch(color=scalar_map_unseen.to_rgba(i), label='{0:.2f} unseen-unseen'.format(proportion))
        general_patch = mpatches.Patch(color=scalar_map_unseen.to_rgba(i), label='{0:.2f}'.format(proportion))
        seen_patches.append(seen_patch)
        unseen_patches.append(unseen_patch)
        general_patches.append(general_patch)

    # Figure 1
    plt.figure(1)
    plt.legend(handles=general_patches, loc='lower left', title='Proportion Thresholded', prop={'size': 7})
    plt.savefig(os.path.join(save_dir, spread_name + '_total_pr_family.png'), bbox_inches='tight')
    plt.clf()

    # Figure 2
    plt.figure(2)
    plt.legend(handles=seen_patches + unseen_patches, ncol=2, loc='lower right', title='Proportion Thresholded', prop={'size': 6})
    plt.savefig(os.path.join(save_dir, spread_name + '_recall_v_total_recall_family.png'), bbox_inches='tight')
    plt.clf()

    # Figure 3
    plt.figure(3)
    plt.legend(handles=seen_patches + unseen_patches, ncol=2, loc='lower left', title='Proportion Thresholded', prop={'size': 6})
    plt.savefig(os.path.join(save_dir, spread_name + 'subset_pr_family.png'), bbox_inches='tight')
    plt.clf()

    plt.close()
    return


def plot_pr_balance_v_spread_thresh(labels_seen, predictions_seen, labels_unseen, predictions_unseen,
                                           spread_seen, spread_unseen, save_dir,
                                           spread_thresh=(0.03, 0.05, 0.1),
                                           resolution=200, spread_name='mutual_info'):

    num_seen = len(labels_seen)
    num_unseen = len(labels_unseen)
    num_total = num_seen + num_unseen

    # Set a color map
    viridis = plt.get_cmap('viridis')
    jet = plt.get_cmap('jet')
    c_norm = matplotlib.colors.Normalize(vmin=0, vmax=len(spread_thresh) + 1)
    scalar_map_seen = matplotlib.cm.ScalarMappable(norm=c_norm, cmap=viridis)
    scalar_map_unseen = matplotlib.cm.ScalarMappable(norm=c_norm, cmap=jet)

    # Add the full plot for comparison
    plot_precision_recall_balance_single(labels_seen, predictions_seen, labels_unseen, predictions_unseen, save_dir,
                                         color_seen='gray', color_unseen='gray')

    # Plot the graphs
    for i in range(len(spread_thresh)):
        thresh = spread_thresh[i]

        # Extract the idxs of examples where spread larger than thresh
        idx_seen = spread_seen >= thresh
        idx_unseen = spread_unseen >= thresh

        # Threshold the predictions
        pred_seen_thresh = predictions_seen.copy()
        pred_unseen_thresh = predictions_unseen.copy()
        pred_seen_thresh[idx_seen] = 0.
        pred_unseen_thresh[idx_unseen] = 0.

        # Get the appriopriate colors
        color_val_seen = scalar_map_seen.to_rgba(i)
        color_val_unseen = scalar_map_unseen.to_rgba(i)

        # Add lines to plot
        plot_precision_recall_balance_single(labels_seen, pred_seen_thresh, labels_unseen, pred_unseen_thresh, save_dir,
                                             color_seen=color_val_seen, color_unseen=color_val_unseen)
        del pred_seen_thresh, pred_unseen_thresh

    # Create the legend
    seen_patches, unseen_patches, general_patches = [], [], []
    for i in range(len(spread_thresh)):
        thresh = spread_thresh[i]
        seen_patch = mpatches.Patch(color=scalar_map_seen.to_rgba(i), label='{0:.2f} seen-seen'.format(thresh))
        unseen_patch = mpatches.Patch(color=scalar_map_unseen.to_rgba(i), label='{0:.2f} unseen-unseen'.format(thresh))
        general_patch = mpatches.Patch(color=scalar_map_unseen.to_rgba(i), label='{0:.2f}'.format(thresh))
        seen_patches.append(seen_patch)
        unseen_patches.append(unseen_patch)
        general_patches.append(general_patch)

    # Figure 1
    plt.figure(1)
    plt.legend(handles=general_patches, loc='lower left', title='Spread Threshold', prop={'size': 7})
    plt.savefig(os.path.join(save_dir, spread_name + '_total_pr_family_thresh.png'), bbox_inches='tight')

    # Figure 2
    plt.figure(2)
    plt.legend(handles=seen_patches + unseen_patches, ncol=2, loc='lower right', title='Spread Threshold', prop={'size': 6})
    plt.savefig(os.path.join(save_dir, spread_name + '_recall_v_tot_recall_family_thresh.png'), bbox_inches='tight')

    # Figure 3
    plt.figure(3)
    plt.legend(handles=seen_patches + unseen_patches, ncol=2, loc='lower left', title='Spread Threshold', prop={'size': 6})
    plt.savefig(os.path.join(save_dir, spread_name + 'subset_pr_family_thresh.png'), bbox_inches='tight')

    plt.close()
    return


def plot_threshold_v_pr_family(labels_seen, predictions_seen, labels_unseen, predictions_unseen,
                               spread_seen, spread_unseen, save_dir,
                               proportions_included=(0.161, 0.4, 0.6, 0.7, 1.0),
                               resolution=200, spread_name='mutual_info'):

    # todo: finish this off
    num_seen = len(labels_seen)
    num_unseen = len(labels_unseen)
    num_total = num_seen + num_unseen
    labels_seen, predictions_seen, spread_seen = _sort_predictions_labels(labels_seen, predictions_seen, spread_seen)
    labels_unseen, predictions_unseen, spread_unseen = _sort_predictions_labels(labels_unseen, predictions_unseen, spread_unseen)
    labels_total, predictions_total, spread_total = _sort_predictions_labels(np.hstack((labels_seen, labels_unseen)),
                                                                             np.hstack((predictions_seen, predictions_unseen)),
                                                                             np.hstack((spread_seen, spread_unseen)))

    # Set a color map
    viridis = plt.get_cmap('viridis')
    jet = plt.get_cmap('jet')
    c_norm = matplotlib.colors.Normalize(vmin=0, vmax=len(proportions_included))
    scalar_map_seen = matplotlib.cm.ScalarMappable(norm=c_norm, cmap=viridis)
    scalar_map_unseen = matplotlib.cm.ScalarMappable(norm=c_norm, cmap=jet)

    for i in range(len(proportions_included)):
        proportion = proportions_included[i]

        # Extract the last index for each subset
        max_val_idx = int(math.floor(num_total * proportion))
        max_spread = spread_total[min(max_val_idx, num_total - 1)]
        last_idx_seen = np.argmax(spread_seen >= max_spread)
        last_idx_unseen = np.argmax(spread_unseen >= max_spread)

        # Threshold the predictions based on sorted order:
        pred_seen_thresh = predictions_seen.copy()
        pred_unseen_thresh = predictions_unseen.copy()
        pred_seen_thresh[last_idx_seen:] = 0.
        pred_unseen_thresh[last_idx_unseen:] = 0.

        # Get the appriopriate colors
        color_val_seen = scalar_map_seen.to_rgba(i)
        color_val_unseen = scalar_map_unseen.to_rgba(i)

        # Add lines to plot
        plot_precision_recall_balance_single(labels_seen, pred_seen_thresh, labels_unseen, pred_unseen_thresh, save_dir,
                                             color_seen=color_val_seen, color_unseen=color_val_unseen)
        del pred_seen_thresh, pred_unseen_thresh


def calc_precision_recall_metrics(labels_seen, predictions_seen, labels_unseen, predictions_unseen, resolution=200):
    precision, recall, thresholds = precision_recall_curve(np.hstack((labels_seen, labels_unseen)),
                                                           np.hstack((predictions_seen, predictions_unseen)))
    # Get rid of the last values which are set manually by the sklearn algorithm
    precision = precision[:-1]
    recall = recall[:-1]

    # Optimise the number of thresholds to plot
    # First sort by recall value
    sort_idx = np.argsort(recall)
    recall = recall[sort_idx]
    precision = precision[sort_idx]
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
    precision_opt = np.extract(select, precision)
    thresholds_opt = np.extract(select, thresholds)

    # Get the precision and recalls on datasets
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
    return precision_seen[:-1], precision_unseen[:-1], precision_opt[:-1], recall_seen[:-1], recall_unseen[:-1], recall_opt[:-1]


def plot_precision_recall_balance_legacy(labels_seen, predictions_seen, labels_unseen, predictions_unseen, save_dir,
                                         num_thresh=500):
    labels_total = np.hstack((labels_seen, labels_unseen))
    predictions_total = np.hstack((predictions_seen, predictions_unseen))

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


def main(args):
    start_time = time.time()

    models_parent_dir = '/home/miproj/urop.2018/bkm28/seed_experiments'
    model_dirs = [os.path.join(models_parent_dir, "atm_seed_{}".format(int(i))) for i in range(1, 11)]

    # Below seen refers to 'seen-seen' and unseen refers to 'unseen-unseen' examples.
    labels_seen, ensemble_pred_seen = get_ensemble_predictions(model_dirs,
                                                               rel_labels_filepath=args.rel_labels_path_seen)
    labels_unseen, ensemble_pred_unseen = get_ensemble_predictions(model_dirs,
                                                                   rel_labels_filepath=args.rel_labels_path_unseen)
    print("Label Data Loaded. Time: ", time.time() - start_time)

    # Invert everything so that off-topic is 1:
    labels_seen, ensemble_pred_seen, labels_unseen, ensemble_pred_unseen = map(lambda x: 1 - x,
                                                                               [labels_seen, ensemble_pred_seen,
                                                                                labels_unseen, ensemble_pred_unseen])
    metrics_seen = calc_metrics(labels_seen, ensemble_pred_seen)
    metrics_unseen = calc_metrics(labels_unseen, ensemble_pred_unseen)

    print("Metrics calculated. Time: ", time.time() - start_time)
    print("Ratio seen-seen dataset to unseen-unseen dataset:",
          float(len(labels_seen)) / float(len(labels_unseen) + len(labels_seen)))

    # Make the plots:
    save_dir = args.savedir
    plot_precision_recall_balance(labels_seen, metrics_seen['avg_predictions'], labels_unseen,
                                  metrics_unseen['avg_predictions'], save_dir)
    print("Made triple PR plot with subset split. Time taken: ", time.time() - start_time)
    plot_precision_recall_balance_v_spread(labels_seen, metrics_seen['avg_predictions'], labels_unseen,
                                           metrics_unseen['avg_predictions'], metrics_seen['mutual_information'],
                                           metrics_unseen['mutual_information'], save_dir, spread_name='mutual_info')
    print("Made triple PR plot family with subset split by mutual information. Time taken: ", time.time() - start_time)
    plot_precision_recall_balance_v_spread(labels_seen, metrics_seen['avg_predictions'], labels_unseen,
                                           metrics_unseen['avg_predictions'], metrics_seen['range_spread'],
                                           metrics_unseen['range_spread'], save_dir, spread_name='range')
    print("Made triple PR plot family with subset split by range. Time taken: ", time.time() - start_time)

    # plot_datasets_histogram(metrics_seen['mutual_information'], metrics_unseen['mutual_information'], save_dir)
    # Same plot as above, but use spread threshold instead of proportion included to disctiminate between curves
    plot_pr_balance_v_spread_thresh(labels_seen, metrics_seen['avg_predictions'], labels_unseen,
                                    metrics_unseen['avg_predictions'], metrics_seen['mutual_information'],
                                    metrics_unseen['mutual_information'], save_dir, spread_name='mutual_info')

    # Threshold plots:
    # todo:

    plt.clf()
    # Cumulative density functions:
    # For the seen examples:
    plot_cum_density_family(metrics_seen['avg_predictions'], labels_seen, metrics_seen['mutual_information'], spread_thresholds=[0.01, 0.02, 0.05, 0.1])
    plt.savefig(os.path.join(save_dir, 'mutual_info_cum_density_family_seen.png'), bbox_inches='tight')
    plt.clf()

    # For the unseen examples:
    plot_cum_density_family(metrics_unseen['avg_predictions'], labels_unseen, metrics_unseen['mutual_information'], spread_thresholds=[0.01, 0.05, 0.1, 0.2, 0.3])
    plt.savefig(os.path.join(save_dir, 'mutual_info_cum_density_family_unseen.png'), bbox_inches='tight')
    plt.clf()

    # For all examples:
    plot_cum_density_family(np.hstack((metrics_seen['avg_predictions'], metrics_unseen['avg_predictions'])),
                            np.hstack((labels_seen, labels_unseen)),
                            np.hstack((metrics_seen['mutual_information'], metrics_unseen['mutual_information'])),
                            spread_thresholds=[0.01, 0.02, 0.05, 0.1, 0.2, 0.3])
    plt.savefig(os.path.join(save_dir, 'mutual_info_cum_density_family_all.png'), bbox_inches='tight')
    plt.clf()


    return


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
