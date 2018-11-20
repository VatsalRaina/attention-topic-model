"Misclassification detection test, but for an ensemble"
from __future__ import print_function, division

import context

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

sns.set()

parser = argparse.ArgumentParser(description='Plot useful graphs for evaluation.')
parser.add_argument('models_parent_dir', type=str, help='Path to directory with models')
parser.add_argument('save_dir', type=str, default='.',
                    help='Path to directory where to save the plots')
parser.add_argument('--model_base_name', type=str, default='atm_prior_net_stats')
parser.add_argument('--unseen_eval_dir', type=str, default='eval_linsk_ALL')
parser.add_argument('--seen_eval_dir', type=str, default='eval4_CDE')
parser.add_argument('--num_trained_models', type=int, default=10)

matplotlib.rcParams['savefig.dpi'] = 200

class EnsembleStats(object):
    def __init__(self, eval_stats_list):
        self.num_models = len(eval_stats_list)

        probs = []
        labels = []
        for eval_stats in eval_stats_list:
            probs.append(eval_stats.probs)
            labels.append(eval_stats.labels)

        probs = np.stack(probs, axis=1)
        self.all_probs = probs
        self.probs = probs.mean(axis=1)

        for i in range(self.num_models - 1):
            assert np.all(labels[i] == labels[i + 1])
        self.labels = labels[0]

        # Calculate the measures of uncertainty
        self.mutual_info, self.entropy_of_expected, self.expected_entropy = self.calc_mutual_info()

        # Calculate misclassifications
        self.predictions = np.asarray(self.probs >= 0.5, dtype=np.float32)
        self.misclassifications = np.asarray(self.labels != self.predictions, dtype=np.float32)
        self.correct = np.asarray(self.labels == self.predictions, dtype=np.float32)
        assert np.all(self.misclassifications != self.correct)

        self.size = len(self.labels)

    def __len__(self):
        return self.size

    def calc_mutual_info(self):
        # Calculate entropy of expected distribution (also the entropy of the overall ensemble predictions)
        entropy_of_expected = calc_entropy(self.probs)
        # Calculate the entropy of each model
        entropy = calc_entropy(self.all_probs)
        # Calculate the expected entropy of each distribution
        expected_entropy = np.mean(entropy, axis=1)
        # Mutual information can be expressed as the difference between the two
        mutual_information = entropy_of_expected - expected_entropy
        return mutual_information, entropy_of_expected, expected_entropy

    def calc_roc_auc(self):
        return roc_auc_score(self.labels, self.probs)


class ModelEvaluationStats(object):
    def __init__(self, model_dir_path, eval_dir='eval4_CDE'):
        self.model_dir_path = model_dir_path
        self.eval_dir = eval_dir
        self.eval_dir_path = os.path.join(model_dir_path, eval_dir)

        # Get the evaluation outputs
        self.labels, self.probs = get_labels_predictions(self.eval_dir_path)

        self.size = len(self.labels)

    def __len__(self):
        return self.size


def calc_entropy(probabilities):
    entropy = - probabilities * np.log(probabilities) - (1. - probabilities) * np.log((1. - probabilities))
    return entropy


def get_labels_predictions(eval_dir):
    labels = []
    with open(os.path.join(eval_dir, 'labels.txt'), "r") as file:
        for line in file.readlines():
            single_example = line.strip()
            label = float(single_example)
            labels.append(label)

    predictions = []
    with open(os.path.join(eval_dir, 'predictions.txt'), "r") as file:
        for line in file.readlines():
            single_example = line.strip()
            prediction = float(single_example)
            predictions.append(prediction)
    labels_array = np.array(labels, dtype=np.float32)
    preds_array = np.array(predictions, dtype=np.float32)
    return labels_array, preds_array


def run_misclassification_detection(misclassification_labels, uncertainty):
    precision, recall, thresholds = precision_recall_curve(misclassification_labels, uncertainty, pos_label=1)
    aupr_pos = auc(recall, precision)

    precision, recall, thresholds = precision_recall_curve(misclassification_labels, -uncertainty, pos_label=0)
    aupr_neg = auc(recall, precision)

    roc_auc = roc_auc_score(misclassification_labels, uncertainty)
    return [roc_auc, aupr_pos, aupr_neg]


def run_misclassification_detection_experiment(misclassification_labels, uncertainty, evaluation_name, save_dir=None):
    roc_auc, aupr_pos, aupr_neg = run_misclassification_detection(misclassification_labels, uncertainty)
    res_string = evaluation_name + ':\nMisclassification ROC-AUC: {:.3f}\n'.format(roc_auc) + \
                 'Misclassification AUPR-pos: {:.3f}\n'.format(aupr_pos) + \
                 'Misclassification AUPR-neg: {:.3f}\n\n'.format(aupr_neg)
    if save_dir:
        with open(os.path.join(save_dir, 'misclassification_detection_results.txt'), 'a+') as f:
            f.write(res_string)
    return


# def _plot_rejection_plot_data_single(rejection_ratios, roc_auc_scores, legend_label, color='red'):
#     mean_roc = roc_auc_scores.mean(axis=1)
#     std_roc = roc_auc_scores.std(axis=1)
#
#     plt.plot(rejection_ratios[~np.isnan(mean_roc)], mean_roc[~np.isnan(mean_roc)], label=legend_label, color=color)
#     plt.fill_between(rejection_ratios[~np.isnan(mean_roc)],
#                      mean_roc[~np.isnan(mean_roc)] - std_roc[~np.isnan(mean_roc)],
#                      mean_roc[~np.isnan(mean_roc)] + std_roc[~np.isnan(mean_roc)], alpha=.2, color=color)
#
#     return mean_roc, std_roc


# def _calc_auc_rr_single(rejection_ratios, roc_auc_scores):
#     num_models = roc_auc_scores.shape[-1]
#     rr_aucs = np.empty(shape=num_models, dtype=np.float32)
#     for i in range(num_models):
#         auc_model = auc(rejection_ratios, roc_auc_scores[:, i])
#
#         roc_auc_wo_rejection = roc_auc_scores[0, i]
#         auc_baseline = auc(np.array([0., 1.]), np.array([roc_auc_wo_rejection, 1.]))
#
#         rr_aucs[i] = auc_model - auc_baseline
#     return rr_aucs


def _calc_rejection_plot_data_single(labels, probs, uncertainty, examples_included_arr):
    resolution = len(examples_included_arr)
    # Sort by uncertainty
    sort_idx = np.argsort(uncertainty)
    probs_sorted = probs[sort_idx]
    labels_sorted = labels[sort_idx]

    roc_auc_scores = np.zeros(shape=len(examples_included_arr), dtype=np.float32)

    # Calculate ROC-AUC at different ratios
    for i in xrange(resolution):
        examples_included = examples_included_arr[i]
        probs_post_rejection = np.hstack(
            [probs_sorted[:examples_included], labels_sorted[examples_included:]])
        try:
            roc_auc_scores[i] = roc_auc_score(labels_sorted, probs_post_rejection)
        except ValueError:
            roc_auc_scores[i] = np.nan
    return roc_auc_scores


def make_rejection_plot(labels, probs, uncertainty_metrics, uncertainty_display_names, evaluation_name,
                        savedir=None, resolution=100):
    """
    :param make_plot:
    :param resolution:
    :return: (rejection_ratios, y_points, rejection_curve_auc)
    rejection_ratios are the exact rejection ratios used to calculate each ROC AUC data point for each model
    y_points is a list of arrays of ROC-AUC scores corresponding to each rejection ratio in rejection ratios, for every
    model
    rejection_curve_auc is a list of areas under the curve of the rejection plot for each model
    """
    # Calculate the number of examples to include for each data point
    num_examples = len(labels)
    num_uncertainty_metrics = len(uncertainty_metrics)
    assert len(uncertainty_display_names) == num_uncertainty_metrics

    examples_rejected_arr = np.floor(np.linspace(0., 1., num=resolution, endpoint=False) * num_examples).astype(
        np.int32)
    examples_included_arr = num_examples - examples_rejected_arr
    rejection_ratios = examples_rejected_arr.astype(np.float32) / num_examples

    roc_auc_scores_list = []

    for i in range(num_uncertainty_metrics):
        uncertainty = uncertainty_metrics[i]
        roc_auc_scores = _calc_rejection_plot_data_single(labels, probs, uncertainty, examples_included_arr)
        roc_auc_scores_list.append(roc_auc_scores)

    # Calculate the curve for the oracle
    l1_error = np.abs(probs - labels).astype(np.float32)
    roc_auc_scores_oracle = _calc_rejection_plot_data_single(labels, probs, l1_error, examples_included_arr)

    # Make the plot
    plt.clf()
    clrs = sns.color_palette("hls", num_uncertainty_metrics + 1)
    for i in range(num_uncertainty_metrics):
        roc_auc_scores = roc_auc_scores_list[i]
        plt.plot(rejection_ratios[~np.isnan(roc_auc_scores)], roc_auc_scores[~np.isnan(roc_auc_scores)],
                 label=uncertainty_display_names[i], color=clrs[i], alpha=.9)

    # Make thdk curve for the oracle
    plt.plot(rejection_ratios[~np.isnan(roc_auc_scores_oracle)],
             roc_auc_scores_oracle[~np.isnan(roc_auc_scores_oracle)], label='Oracle', color=clrs[-1], alpha=.9)

    # Plot the random baseline
    plt.plot([0.0, 1.0], [roc_auc_scores_oracle[0], 1.], 'k--', lw=3)

    plt.xlabel("Percentage examples rejected to revaluation")
    plt.ylabel("ROC AUC score after revaluation")
    plt.xlim(0.0, 1.0)
    plt.ylim(ymax=1.0)
    plt.legend(title='Uncertainty Metric', loc='lower right')

    if savedir is not None:
        # Calculate the AUC_RR scores
        # Baseline
        baseline_auc = auc([0., 1.], [roc_auc_scores_oracle[0], 1.])

        # For the oracle
        oracle_auc = auc(rejection_ratios, roc_auc_scores_oracle)
        for i in range(num_uncertainty_metrics):
            curve_auc = auc(rejection_ratios, roc_auc_scores_list[i])
            auc_rr = (curve_auc - baseline_auc) / (oracle_auc - baseline_auc)
            with open(os.path.join(savedir, 'auc_rr.txt'), 'a') as f:
                f.write(evaluation_name + ' | Uncertainty: ' + uncertainty_display_names[i] + '\n')
                f.write('ROC AUC RR of Ensemble: {:.3f}\n'.format(auc_rr))

        # Save the plot
        plt.savefig(os.path.join(savedir, 'auc_rr_{}.png'.format(evaluation_name), bbox_inches='tight'))
    return


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

    ensemble_seen = EnsembleStats(all_evaluation_stats_seen)
    ensemble_unseen = EnsembleStats(all_evaluation_stats_unseen)

    # Clear the contents of the file
    open(os.path.join(args.save_dir, 'misclassification_detection_results.txt'), 'w').close()
    # Calculate the metrics and numerical results:

    with open(os.path.join(args.save_dir, 'misclassification_detection_results.txt'), 'a') as f:
        # Write the accuracy
        f.write("Accuracy (seen-seen): {:3f}\n".format(ensemble_seen.correct.mean()))
        f.write("Accuracy (unseen-unseen): {:3f}\n\n".format(ensemble_unseen.correct.mean()))

    run_misclassification_detection_experiment(ensemble_seen.misclassifications, ensemble_seen.mutual_info,
                                               evaluation_name='Seen-seen Mutual Info', save_dir=args.save_dir)
    run_misclassification_detection_experiment(ensemble_seen.misclassifications, ensemble_seen.expected_entropy,
                                               evaluation_name='Seen-seen Expected Entropy', save_dir=args.save_dir)
    run_misclassification_detection_experiment(ensemble_seen.misclassifications, ensemble_seen.entropy_of_expected,
                                               evaluation_name='Seen-seen Entropy of Expected', save_dir=args.save_dir)
    run_misclassification_detection_experiment(ensemble_unseen.misclassifications, ensemble_unseen.mutual_info,
                                               evaluation_name='Unseen-unseen Mutual Info', save_dir=args.save_dir)
    run_misclassification_detection_experiment(ensemble_unseen.misclassifications, ensemble_unseen.expected_entropy,
                                               evaluation_name='Unseen-unseen Expected Entropy', save_dir=args.save_dir)
    run_misclassification_detection_experiment(ensemble_unseen.misclassifications, ensemble_unseen.entropy_of_expected,
                                               evaluation_name='Unseen-unseen Entropy of Expected',
                                               save_dir=args.save_dir)

    # Calculate the ROC AUC scores
    with open(os.path.join(args.save_dir, 'roc_auc_results.txt'), 'w') as f:
        write_str = 'Seen-seen ROC-AUC: {}\n'.format(
            ensemble_seen.calc_roc_auc()) + 'Unseen-unseen ROC-AUC: {:.3f}\n\n'.format(ensemble_unseen.calc_roc_auc())
        f.write(write_str)

    # Calculate AUC_RR
    open(os.path.join(args.save_dir, 'auc_rr.txt'), 'w').close()
    make_rejection_plot(ensemble_seen.labels, ensemble_seen.probs, uncertainty_metrics=[ensemble_seen.entropy_of_expected,
                                                                                        ensemble_seen.mutual_info,
                                                                                        ensemble_seen.expected_entropy],
                        uncertainty_display_names=['Entropy of Avg.', 'Mutual Info.', 'Avg. Entropy'],
                        evaluation_name='Seen-seen', savedir=args.save_dir)
    make_rejection_plot(ensemble_unseen.labels, ensemble_unseen.probs,
                        uncertainty_metrics=[ensemble_unseen.entropy_of_expected,
                                             ensemble_unseen.mutual_info,
                                             ensemble_unseen.expected_entropy],
                        uncertainty_display_names=['Entropy of Avg.', 'Mutual Info.', 'Avg. Entropy'],
                        evaluation_name='Unseen-unseen', savedir=args.save_dir)
    return


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
