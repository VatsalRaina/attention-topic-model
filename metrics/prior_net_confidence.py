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
from sklearn.metrics import auc

sns.set(style='whitegrid')

parser = argparse.ArgumentParser(description='Plot useful graphs for evaluation.')
parser.add_argument('models_parent_dir', type=str, help='Path to directory with models')
parser.add_argument('save_dir', type=str, default='.',
                    help='Path to directory where to save the plots')
parser.add_argument('--model_base_name', type=str, default='atm_prior_net_stats')
parser.add_argument('--unseen_eval_dir', type=str, default='eval_linsk_ALL')
parser.add_argument('--seen_eval_dir', type=str, default='eval4_CDE')
parser.add_argument('--num_trained_models', type=int, default=10)
parser.add_argument('--make_plots', action='store_true', help='Whether to make plots, or just get the numerical'
                                                              'results.')

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


def plot_uncertainty_histogram(uncertainty_seen, uncertainty_unseen, num_bins=50):
    clrs = sns.xkcd_palette(["windows blue", "amber"])

    # Make the bin divisions
    upper_edge = max(np.max(uncertainty_seen), np.max(uncertainty_unseen))
    lower_edge = min(np.min(uncertainty_seen), np.min(uncertainty_unseen))
    bins = np.linspace(lower_edge, upper_edge, num_bins+1)

    plt.hist(uncertainty_seen, bins=bins, density=True, label='seen-seen', color=clrs[0], alpha=0.5)
    plt.hist(uncertainty_unseen, bins=bins, density=True, label='unseen-unseen', color=clrs[1], alpha=0.5)
    plt.legend()
    return


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

    plt.plot(proportions_included, roc_auc_scores, color=color, alpha=0.75)
    plt.xlabel("Percentage examples included as sorted by " + sort_by_name + " of Prior Net output.")
    plt.ylabel("ROC AUC score on the subset examples included")
    plt.xlim(0, 1)
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
                                    bg_alpha=0.25, line_color='black', linewidth=1.4):
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
    percentage_seen = np.cumsum(is_seen_sorted, dtype=np.float32) / np.arange(1, num_examples + 1, dtype=np.float32)

    # Plot the proportions included
    x = np.arange(num_examples, dtype=np.float32) / num_examples
    clrs = sns.color_palette("husl", 2)
    plt.fill_between(x, 0., percentage_seen, facecolor=clrs[0], alpha=bg_alpha)
    plt.fill_between(x, percentage_seen, 1., facecolor=clrs[1], alpha=bg_alpha)

    # Plot the ROC_AUC vs. proportions included
    plt.plot(proportions_included, roc_auc_scores, color=line_color, linewidth=linewidth, alpha=0.8)

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Percentage examples included as sorted by " + sort_by_name + " of Prior Net output.")
    return


def plot_auc_vs_percentage_included_seen_vs_unseen(labels_seen, labels_unseen, predictions_seen, predictions_unseen,
                                                   sort_by_array_seen, sort_by_array_unseen, resolution=100,
                                                   sort_by_name='diff. entropy'):
    labels = np.hstack((labels_seen, labels_unseen))
    predictions = np.hstack((predictions_seen, predictions_unseen))
    sort_by_array = np.hstack((sort_by_array_seen, sort_by_array_unseen))

    num_examples = len(labels)

    # Array with ones if the example is seen-seen and zero otherwise
    is_seen = np.hstack((np.ones_like(labels_seen), np.zeros_like(labels_unseen)))

    proportions_included, roc_auc_scores, sorted_order = _get_cum_roc_auc_with_sort(labels, predictions, sort_by_array,
                                                                                    resolution=resolution)
    is_seen_sorted = is_seen[sorted_order]
    percentage_seen = np.cumsum(is_seen_sorted, dtype=np.float32) / np.arange(1, num_examples + 1, dtype=np.float32)

    clrs = sns.color_palette("husl", 2)
    score_color = clrs[0]
    proportion_color = clrs[1]

    # Plot the proportion seen-seen vs. proportions included
    plt.plot(np.arange(num_examples, dtype=np.float32) / num_examples, percentage_seen, color=proportion_color, alpha=0.5, linewidth=0.8)

    # Plot the ROC_AUC vs. proportions included
    plt.plot(proportions_included, roc_auc_scores, color=score_color, linewidth=1.2, alpha=0.8)
    return


def run_misclassification_detection(misclassification_labels, uncertainty):
    precision, recall, thresholds = precision_recall_curve(misclassification_labels, uncertainty, pos_label=1)
    aupr_pos = auc(recall, precision)

    precision, recall, thresholds = precision_recall_curve(misclassification_labels, -uncertainty, pos_label=0)
    aupr_neg = auc(recall, precision)

    roc_auc = roc_auc_score(misclassification_labels, uncertainty)

    return [roc_auc, aupr_pos, aupr_neg]


def run_misclassification_detection_over_ensemble(eval_stats_list, uncertainty_attr_name, evaluation_name, savedir=None):
    """
    This functions runs a uncertainty based miclassification detection experiment
    """

    auc_array_uncertainty = []
    accuracies = []

    for eval_stats in eval_stats_list:
        uncertainty = getattr(eval_stats, uncertainty_attr_name)
        predictions = eval_stats.preds
        labels = eval_stats.labels

        off_topic_probabilities = 1.0 - predictions

        on_topic_probabilities = np.expand_dims(predictions, axis=1)
        off_topic_probabilities = np.expand_dims(off_topic_probabilities, axis=1)

        probabilities = np.concatenate([off_topic_probabilities, on_topic_probabilities], axis=1)

        # Threshold the predictions with threshold 0.5 (take prediction to be the class with max prob.)
        predictions = np.argmax(probabilities, axis=1)  # on-topic is one

        misclassification = np.asarray(labels != predictions, dtype=np.float32)
        correct = np.asarray(labels == predictions, dtype=np.float32)

        accuracies.append(np.mean(correct))

        auc = run_misclassification_detection(misclassification, uncertainty)
        auc_array_uncertainty.append(auc)

    accuracies = np.asarray(accuracies)

    auc_uncertainty = np.stack(auc_array_uncertainty, axis=0)
    auc_uncertainty_mean, auc_uncertainty_std = np.mean(auc_uncertainty, axis=0), np.std(auc_uncertainty, axis=0)

    res_string = evaluation_name + ':\nMean Accuracy = {} +/- {}\nmisclassification ROC AUC: {} +/- {}\n' \
                 'misclassification AUPR POS: {} +/- {}\n' \
                 'misclassification AUPR NEG: {} +/- {}\n'.format(
        str(accuracies.mean()), str(accuracies.std()), auc_uncertainty_mean[0], auc_uncertainty_std[0], auc_uncertainty_mean[1],
        auc_uncertainty_std[1], auc_uncertainty_mean[2], auc_uncertainty_std[2])
    print(res_string)

    if savedir:
        with open(os.path.join(savedir, 'misclassification_detect_individual.txt'), 'a+') as f:
            f.write(res_string)
    return


def run_roc_auc_over_ensemble(eval_stats_list, evaluation_name, savedir=None):
    roc_auc_list = []
    for eval_stats in eval_stats_list:
        predictions = eval_stats.preds
        labels = eval_stats.labels

        roc_auc = roc_auc_score(labels, predictions)
        roc_auc_list.append(roc_auc)


    roc_auc_array = np.stack(roc_auc_list)

    res_string = evaluation_name + ':\nIndividual ROC-AUC\'s: {}\n' \
                                   'Mean per model ROC-AUC: {} +/- {}\n'.format(roc_auc_list, roc_auc_array.mean(),
                                                                                roc_auc_array.std())
    print(res_string)
    if savedir:
        with open(os.path.join(savedir, 'roc_auc_results.txt'), 'a+') as f:
            f.write(res_string)
    return


def run_rejection_plot(eval_stats_list, uncertainty_attr_name, evaluation_name, savedir=None, make_plot=False,
                       resolution=100):
    """

    :param make_plot:
    :param resolution:
    :return: (rejection_ratios, y_points, rejection_curve_auc)
    rejection_ratios are the exact rejection ratios used to calculate each ROC AUC data point for each model
    y_points is a list of arrays of ROC-AUC scores corresponding to each rejection ratio in rejection ratios, for every
    model
    rejection_curve_auc is a list of areas under the curve of the rejection plot for each model
    """
    auc_array_uncertainty = []

    # Calculate the number of examples to include for each data point
    num_preds = len(eval_stats_list[0])
    examples_included_arr = np.floor(np.linspace(0., 1., num=resolution, endpoint=False) * num_preds).astype(
        np.int32)
    examples_included_arr = np.flip(examples_included_arr)
    rejection_ratios = 1. - (examples_included_arr.astype(np.float32) / num_preds)

    y_points = []  # List to store the y_coordinates for the plots for each model
    y_points_oracle = []
    rejection_curve_auc = []
    oracle_rejection_curve_auc = []

    for eval_stats in eval_stats_list:
        uncertainty = getattr(eval_stats, uncertainty_attr_name)
        predictions = eval_stats.preds
        labels = eval_stats.labels


        # Sort by uncertainty
        sort_idx = np.argsort(uncertainty)
        predictions = predictions[sort_idx]
        labels = labels[sort_idx]

        roc_auc_list = []
        # Calculate ROC-AUC at different ratios
        for examples_included in examples_included_arr:
            predictions_with_rejection = np.hstack([predictions[:examples_included], labels[examples_included:]])
            roc_auc_list.append(roc_auc_score(labels, predictions_with_rejection))

        roc_auc_arr = np.array(roc_auc_list, dtype=np.float32)

        y_points.append(roc_auc_arr)
        rejection_curve_auc.append(auc(rejection_ratios, roc_auc_arr))


        # Get the oracle results
        # Sort by how wrong the model is
        sort_idx = np.argsort(np.abs(labels.astype(np.float32) - predictions))
        predictions = predictions[sort_idx]
        labels = labels[sort_idx]

        roc_auc_list = []
        # Calculate ROC-AUC at different ratios
        for examples_included in examples_included_arr:
            predictions_with_rejection = np.hstack([predictions[:examples_included], labels[examples_included:]])
            roc_auc_list.append(roc_auc_score(labels, predictions_with_rejection))
        roc_auc_arr = np.array(roc_auc_list, dtype=np.float32)

        y_points_oracle.append(roc_auc_arr)
        oracle_rejection_curve_auc.append(auc(rejection_ratios, roc_auc_arr))

    rejection_curve_auc = np.array(rejection_curve_auc)
    oracle_rejection_curve_auc = np.array(oracle_rejection_curve_auc)


    res_string = evaluation_name + ':\nRejection ratio AUC = {} +/- {}\n' \
                                   'Oraclo RR-curve AUC: {} +/- {}\n'.format(
        str(rejection_curve_auc.mean()), str(rejection_curve_auc.std()), str(oracle_rejection_curve_auc.mean()),
        str(oracle_rejection_curve_auc.std()))
    print(res_string)

    if savedir:
        with open(os.path.join(savedir, 'rejection_ratio_auc.txt'), 'a+') as f:
            f.write(res_string)

    return rejection_ratios, y_points, y_points_oracle


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


    # Combine various metrics into single arrays (for ensemble)
    diff_entropy_seen_combined = reduce(lambda x, y: np.hstack((x, y)),
                                        map(lambda x: x.diff_entropy, all_evaluation_stats_seen))
    diff_entropy_unseen_combined = reduce(lambda x, y: np.hstack((x, y)),
                                          map(lambda x: x.diff_entropy, all_evaluation_stats_unseen))
    mutual_info_seen_combined = reduce(lambda x, y: np.hstack((x, y)),
                                       map(lambda x: x.mutual_info, all_evaluation_stats_seen))
    mutual_info_unseen_combined = reduce(lambda x, y: np.hstack((x, y)),
                                         map(lambda x: x.mutual_info, all_evaluation_stats_unseen))
    # labels_seen = all_evaluation_stats_seen[0].labels
    # labels_unseen = all_evaluation_stats_unseen[0].labels
    #
    # predictions_seen_combined = reduce(lambda x, y: np.hstack((x, y)),
    #                                    map(lambda x: x.preds, all_evaluation_stats_seen))
    # predictions_unseen_combined = reduce(lambda x, y: np.hstack((x, y)),
    #                                      map(lambda x: x.preds, all_evaluation_stats_unseen))



    # Calculate the metrics and numerical results:
    run_misclassification_detection_over_ensemble(all_evaluation_stats_seen, uncertainty_attr_name='mutual_info',
                                                  evaluation_name='Seen-seen Mutual Info', savedir=args.save_dir)
    run_misclassification_detection_over_ensemble(all_evaluation_stats_seen, uncertainty_attr_name='diff_entropy',
                                                  evaluation_name='Seen-seen Diff. Entropy', savedir=args.save_dir)
    run_misclassification_detection_over_ensemble(all_evaluation_stats_unseen, uncertainty_attr_name='mutual_info',
                                                  evaluation_name='Unseen-unseen Mutual Info', savedir=args.save_dir)
    run_misclassification_detection_over_ensemble(all_evaluation_stats_unseen, uncertainty_attr_name='diff_entropy',
                                                  evaluation_name='Unseen-unseen Diff. Entropy', savedir=args.save_dir)

    run_roc_auc_over_ensemble(all_evaluation_stats_seen, evaluation_name='Seen-seen')
    run_roc_auc_over_ensemble(all_evaluation_stats_unseen, evaluation_name='Uneen-unseen')

    rejection_ratios, y_points, y_points_oracle = run_rejection_plot(all_evaluation_stats_seen,
                                                                     uncertainty_attr_name='diff_entropy',
                                                                     evaluation_name='Diff. Entropy seen-seen')
    rejection_ratios, y_points, y_points_oracle = run_rejection_plot(all_evaluation_stats_unseen,
                                                                     uncertainty_attr_name='diff_entropy',
                                                                     evaluation_name='Diff. Entropy unseen-unseen')

    if args.make_plots:
        # MAKE THE PLOTS WITH THE BACKGROUND SHOWING PROPORTIONS INCLUDED
        # Make AUC vs. cumulative samples included for N models - DIFF. ENTROPY
        # Seen-seen and unseen-unseen
        for i in range(args.num_trained_models):
            eval_stats_seen = all_evaluation_stats_seen[i]
            eval_stats_unseen = all_evaluation_stats_unseen[i]
            plot_auc_vs_percentage_included_with_proportions(eval_stats_seen.labels, eval_stats_unseen.labels,
                                                             eval_stats_seen.preds, eval_stats_unseen.preds,
                                                             eval_stats_seen.diff_entropy, eval_stats_unseen.diff_entropy,
                                                             sort_by_name='diff. entropy', bg_alpha=0.10)
        plt.savefig(
            os.path.join(args.save_dir, 'auc_vs_cum_samples_incl_diff_entropy_seen_and_unseen_bg_{}.png'.format(args.model_base_name)),
            bbox_inches='tight')
        plt.clf()

        # Make AUC vs. cumulative samples included for N models - MUTUAL INFO
        # Seen-seen and unseen-unseen
        for i in range(args.num_trained_models):
            eval_stats_seen = all_evaluation_stats_seen[i]
            eval_stats_unseen = all_evaluation_stats_unseen[i]
            plot_auc_vs_percentage_included_with_proportions(eval_stats_seen.labels, eval_stats_unseen.labels,
                                                             eval_stats_seen.preds, eval_stats_unseen.preds,
                                                             eval_stats_seen.mutual_info, eval_stats_unseen.mutual_info,
                                                             sort_by_name='mutual information', bg_alpha=0.10)
        plt.savefig(
            os.path.join(args.save_dir, 'auc_vs_cum_samples_incl_mutual_info_seen_and_unseen_bg_{}.png'.format(args.model_base_name)),
            bbox_inches='tight')
        plt.clf()




        # MAKE THE PLOTS WITH THE LINE SHOWING PROPORTIONS INCLUDED FROM EACH SEEN-SEEN AND UNSEEN-UNSEEN
        # Make AUC vs. cumulative samples included for N models - DIFF. ENTROPY
        # Seen-seen and unseen-unseen
        for i in range(args.num_trained_models):
            eval_stats_seen = all_evaluation_stats_seen[i]
            eval_stats_unseen = all_evaluation_stats_unseen[i]
            plot_auc_vs_percentage_included_seen_vs_unseen(eval_stats_seen.labels, eval_stats_unseen.labels,
                                                           eval_stats_seen.preds, eval_stats_unseen.preds,
                                                           eval_stats_seen.diff_entropy, eval_stats_unseen.diff_entropy,
                                                           sort_by_name='diff. entropy')
        plt.savefig(
            os.path.join(args.save_dir,
                         'auc_vs_cum_samples_incl_diff_entropy_seen_and_unseen_line{}.png'.format(args.model_base_name)),
            bbox_inches='tight')
        plt.clf()

        # Make AUC vs. cumulative samples included for N models - MUTUAL INFO
        # Seen-seen and unseen-unseen
        for i in range(args.num_trained_models):
            eval_stats_seen = all_evaluation_stats_seen[i]
            eval_stats_unseen = all_evaluation_stats_unseen[i]
            plot_auc_vs_percentage_included_seen_vs_unseen(eval_stats_seen.labels, eval_stats_unseen.labels,
                                                           eval_stats_seen.preds, eval_stats_unseen.preds,
                                                           eval_stats_seen.mutual_info, eval_stats_unseen.mutual_info,
                                                           sort_by_name='mutual information')
        plt.savefig(
            os.path.join(args.save_dir,
                         'auc_vs_cum_samples_incl_mutual_info_seen_and_unseen_line{}.png'.format(args.model_base_name)),
            bbox_inches='tight')
        plt.clf()



        # Diff. Entropy
        plot_uncertainty_histogram(diff_entropy_seen_combined, diff_entropy_unseen_combined)
        plt.savefig(
            os.path.join(args.save_dir,
                         'diff_entropy_histogram_{}.png'.format(args.model_base_name)),
            bbox_inches='tight')
        plt.clf()

        # Mutual Info
        plot_uncertainty_histogram(mutual_info_seen_combined, mutual_info_unseen_combined)
        plt.savefig(
            os.path.join(args.save_dir,
                         'mutual_info_histogram_{}.png'.format(args.model_base_name)),
            bbox_inches='tight')
        plt.clf()



        clrs = sns.dark_palette("muted purple", n_colors=args.num_trained_models, input="xkcd")
        # Make AUC vs. cumulative samples included for N models - DIFF. ENTROPY
        # Seen-seen
        for i in range(args.num_trained_models):
            eval_stats = all_evaluation_stats_seen[i]
            plot_auc_vs_percentage_included_single(eval_stats.labels, eval_stats.preds, eval_stats.diff_entropy,
                                                   resolution=200, sort_by_name='diff. entropy', color=clrs[i])
        plt.savefig(
            os.path.join(args.save_dir, 'auc_vs_cum_samples_incl_diff_entropy_seen_{}.png'.format(args.model_base_name)),
            bbox_inches='tight')
        plt.clf()

        # unseen-unseen
        for i in range(args.num_trained_models):
            eval_stats = all_evaluation_stats_unseen[i]
            plot_auc_vs_percentage_included_single(eval_stats.labels, eval_stats.preds, eval_stats.diff_entropy,
                                                   resolution=200, sort_by_name='diff. entropy', color=clrs[i])
        plt.savefig(
            os.path.join(args.save_dir, 'auc_vs_cum_samples_incl_diff_entropy_unseen_{}.png'.format(args.model_base_name)),
            bbox_inches='tight')
        plt.clf()

        # Make AUC vs. cumulative samples included for N models - MUTUAL INFORMATION
        # Seen-seen
        for i in range(args.num_trained_models):
            eval_stats = all_evaluation_stats_seen[i]
            plot_auc_vs_percentage_included_single(eval_stats.labels, eval_stats.preds, eval_stats.mutual_info,
                                                   resolution=200, sort_by_name='mutual information', color=clrs[i])
        plt.savefig(
            os.path.join(args.save_dir, 'auc_vs_cum_samples_incl_mutual_info_seen_{}.png'.format(args.model_base_name)),
            bbox_inches='tight')
        plt.clf()

        # unseen-unseen
        for i in range(args.num_trained_models):
            eval_stats = all_evaluation_stats_unseen[i]
            plot_auc_vs_percentage_included_single(eval_stats.labels, eval_stats.preds, eval_stats.mutual_info,
                                                   resolution=200, sort_by_name='mutual information', color=clrs[i])
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
