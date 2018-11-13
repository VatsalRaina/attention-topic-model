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

# matplotlib.rcParams['savefig.dpi'] = 200


class ModelEvaluationStats(object):
    def __init__(self, model_dir_path, eval_dir='eval4_CDE'):
        self.model_dir_path = model_dir_path
        self.eval_dir = eval_dir
        self.eval_dir_path = os.path.join(model_dir_path, eval_dir)

        # Get the evaluation outputs
        self.labels, self.logits, self.probs = get_labels_logits_predictions(self.eval_dir_path)
        self.alphas = np.exp(self.logits)

        # Calculate the measures of uncertainty
        self.diff_entropy = calc_dirich_diff_entropy(self.alphas)
        self.mutual_info = calc_dirich_mutual_info(self.alphas)

        # Calculate misclassifications
        self.predictions = np.asarray(self.probs >= 0.5, dtype=np.float32)
        self.misclassifications = np.asarray(self.labels != self.predictions, dtype=np.float32)
        self.correct = np.asarray(self.labels == self.predictions, dtype=np.float32)
        assert np.all(self.misclassifications != self.correct)

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


def run_misclassification_detection(misclassification_labels, uncertainty):
    precision, recall, thresholds = precision_recall_curve(misclassification_labels, uncertainty, pos_label=1)
    aupr_pos = auc(recall, precision)

    precision, recall, thresholds = precision_recall_curve(misclassification_labels, -uncertainty, pos_label=0)
    aupr_neg = auc(recall, precision)

    roc_auc = roc_auc_score(misclassification_labels, uncertainty)

    return [roc_auc, aupr_pos, aupr_neg]


def run_misclassification_detection_over_ensemble(eval_stats_list, uncertainty_attr_name, evaluation_name,
                                                  save_dir=None):
    """
    This functions runs a uncertainty based miclassification detection experiment given uncertainty measurement.
    """
    roc_auc_array = []
    aupr_pos_array = []
    aupr_neg_array = []
    accuracies = []

    for eval_stats in eval_stats_list:
        uncertainty = getattr(eval_stats, uncertainty_attr_name)

        accuracies.append(np.mean(eval_stats.correct))

        roc_auc, aupr_pos, aupr_neg = run_misclassification_detection(eval_stats.misclassifications, uncertainty)
        roc_auc_array.append(roc_auc)
        aupr_pos_array.append(aupr_pos)
        aupr_neg_array.append(aupr_neg)

    accuracies, roc_auc_array, aupr_pos_array, aupr_neg_array = map(lambda x: np.asarray(x),
                                                                    [accuracies, roc_auc_array, aupr_pos_array,
                                                                     aupr_neg_array])

    res_string = evaluation_name + ':\nMean Accuracy = {} +/- {}\n'.format(accuracies.mean(), accuracies.std()) + \
                 'misclassification ROC AUC: {} +/- {}\n'.format(roc_auc_array.mean(), roc_auc_array.std()) + \
                 'misclassification AUPR POS: {} +/- {}\n'.format(aupr_pos_array.mean(), aupr_pos_array.std()) + \
                 'misclassification AUPR NEG: {} +/- {}\n\n'.format(aupr_neg_array.mean(), aupr_neg_array.std())
    print(res_string)

    if save_dir:
        with open(os.path.join(save_dir, 'misclassification_detect_individual.txt'), 'a') as f:
            f.write(res_string)
    return


def run_roc_auc_over_ensemble(eval_stats_list, evaluation_name, save_dir=None):
    roc_auc_list = []

    for eval_stats in eval_stats_list:
        predictions = eval_stats.preds
        labels = eval_stats.labels

        roc_auc = roc_auc_score(labels, predictions)
        roc_auc_list.append(roc_auc)

    roc_auc_array = np.stack(roc_auc_list)

    res_string = evaluation_name + ':\nIndividual ROC-AUC\'s: {}\n'.format(roc_auc_list) + \
                 'Mean per model ROC-AUC: {} +/- {}\n\n'.format(roc_auc_array.mean(), roc_auc_array.std())
    print(res_string)

    if save_dir:
        with open(os.path.join(save_dir, 'roc_auc_results.txt'), 'a+') as f:
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

    # Clear the contents of the file
    open(os.path.join(args.save_dir, 'misclassification_detection_results.txt'), 'w').close()
    # Calculate the metrics and numerical results:
    run_misclassification_detection_over_ensemble(all_evaluation_stats_seen, uncertainty_attr_name='mutual_info',
                                                  evaluation_name='Seen-seen Mutual Info', save_dir=args.save_dir)
    run_misclassification_detection_over_ensemble(all_evaluation_stats_seen, uncertainty_attr_name='diff_entropy',
                                                  evaluation_name='Seen-seen Diff. Entropy', save_dir=args.save_dir)
    run_misclassification_detection_over_ensemble(all_evaluation_stats_unseen, uncertainty_attr_name='mutual_info',
                                                  evaluation_name='Unseen-unseen Mutual Info', save_dir=args.save_dir)
    run_misclassification_detection_over_ensemble(all_evaluation_stats_unseen, uncertainty_attr_name='diff_entropy',
                                                  evaluation_name='Unseen-unseen Diff. Entropy', save_dir=args.save_dir)

    # Calculate the average ROC AUC scores
    open(os.path.join(args.save_dir, 'roc_auc_results.txt'), 'w').close()
    run_roc_auc_over_ensemble(all_evaluation_stats_seen, evaluation_name='Seen-seen')
    run_roc_auc_over_ensemble(all_evaluation_stats_unseen, evaluation_name='Uneen-unseen')


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
