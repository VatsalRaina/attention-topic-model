"Misclassification detection test, but for a single standard model"
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

from metrics.misclassification_test_pn import make_rejection_plot

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

class ModelEvaluationStats(object):
    def __init__(self, model_dir_path, eval_dir='eval4_CDE'):
        self.model_dir_path = model_dir_path
        self.eval_dir = eval_dir
        self.eval_dir_path = os.path.join(model_dir_path, eval_dir)

        # Get the evaluation outputs
        self.labels, self.probs = get_labels_predictions(self.eval_dir_path)

        # Calculate the measures of uncertainty
        self.entropy = calc_entropy(self.probs)

        # Calculate misclassifications
        self.predictions = np.asarray(self.probs >= 0.5, dtype=np.float32)
        self.misclassifications = np.asarray(self.labels != self.predictions, dtype=np.float32)
        self.correct = np.asarray(self.labels == self.predictions, dtype=np.float32)
        assert np.all(self.misclassifications != self.correct)

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

    res_string = evaluation_name + ':\nMean Accuracy = {:.3f} +/- {:.3f}\n'.format(accuracies.mean(), accuracies.std()) + \
                 'misclassification ROC AUC: {:.3f} +/- {:.3f}\n'.format(roc_auc_array.mean(), roc_auc_array.std()) + \
                 'misclassification AUPR POS: {:.3f} +/- {:.3f}\n'.format(aupr_pos_array.mean(), aupr_pos_array.std()) + \
                 'misclassification AUPR NEG: {:.3f} +/- {:.3f}\n\n'.format(aupr_neg_array.mean(), aupr_neg_array.std())
    print(res_string)

    if save_dir:
        with open(os.path.join(save_dir, 'misclassification_detection_results.txt'), 'a') as f:
            f.write(res_string)
    return


def run_roc_auc_over_ensemble(eval_stats_list, evaluation_name, save_dir=None):
    roc_auc_list = []

    for eval_stats in eval_stats_list:
        probs = eval_stats.probs
        labels = eval_stats.labels

        roc_auc = roc_auc_score(labels, probs)
        roc_auc_list.append(roc_auc)

    roc_auc_array = np.stack(roc_auc_list)

    res_string = evaluation_name + ':\nIndividual ROC-AUC\'s: {}\n'.format(roc_auc_list) + \
                 'Mean per model ROC-AUC: {:.3f} +/- {:.3f}\n\n'.format(roc_auc_array.mean(), roc_auc_array.std())
    print(res_string)

    if save_dir:
        with open(os.path.join(save_dir, 'roc_auc_results.txt'), 'a+') as f:
            f.write(res_string)
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

    # Clear the contents of the file
    open(os.path.join(args.save_dir, 'misclassification_detection_results.txt'), 'w').close()
    # Calculate the metrics and numerical results:
    run_misclassification_detection_over_ensemble(all_evaluation_stats_seen, uncertainty_attr_name='entropy',
                                                  evaluation_name='Seen-seen Entropy Misclassification',
                                                  save_dir=args.save_dir)
    run_misclassification_detection_over_ensemble(all_evaluation_stats_unseen, uncertainty_attr_name='entropy',
                                                  evaluation_name='Unseen-unseen Entropy Misclassification',
                                                  save_dir=args.save_dir)

    # Calculate the average ROC AUC scores
    open(os.path.join(args.save_dir, 'roc_auc_results.txt'), 'w').close()
    run_roc_auc_over_ensemble(all_evaluation_stats_seen, evaluation_name='Seen-seen', save_dir=args.save_dir)
    run_roc_auc_over_ensemble(all_evaluation_stats_unseen, evaluation_name='Uneen-unseen', save_dir=args.save_dir)

    # Calculate the AUC_RR
    open(os.path.join(args.save_dir, 'auc_rr.txt'), 'w').close()
    make_rejection_plot(all_evaluation_stats_seen, uncertainty_attr_names=['entropy'],
                        uncertainty_display_names=['Entropy'],
                        evaluation_name='Seen-seen', savedir=args.save_dir)
    make_rejection_plot(all_evaluation_stats_unseen, uncertainty_attr_names=['entropy'],
                        uncertainty_display_names=['Entropy'],
                        evaluation_name='Unseen-unseen', savedir=args.save_dir)
    return


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
