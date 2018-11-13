"Misclassification detection test, but for an ensemble"
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
        self.mutual_info = self.calc_mutual_info()

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
        return mutual_information

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
    res_string = evaluation_name + ':\nMisclassification ROC-AUC: {}\n'.format(roc_auc) + \
                 'Misclassification AUPR-pos: {}\n'.format(aupr_pos) + \
                 'Misclassification AUPR-neg: {}\n\n'.format(aupr_neg)
    if save_dir:
        with open(os.path.join(save_dir, 'misclassification_detection_results.txt'), 'a+') as f:
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

    ensemble_seen = EnsembleStats(all_evaluation_stats_seen)
    ensemble_unseen = EnsembleStats(all_evaluation_stats_unseen)

    # Clear the contents of the file
    open(os.path.join(args.save_dir, 'misclassification_detection_results.txt'), 'w').close()
    # Calculate the metrics and numerical results:
    run_misclassification_detection_experiment(ensemble_seen.misclassifications, ensemble_seen.mutual_info,
                                               evaluation_name='Seen-seen Mutual Info', save_dir=args.save_dir)
    run_misclassification_detection_experiment(ensemble_unseen.misclassifications, ensemble_unseen.mutual_info,
                                               evaluation_name='Unseen-unseen Mutual Info', save_dir=args.save_dir)

    # Calculate the ROC AUC scores
    with open(os.path.join(args.save_dir, 'roc_auc_results.txt'), 'w') as f:
        write_str = 'Seen-seen ROC-AUC: {}\n'.format(
            ensemble_seen.calc_roc_auc()) + 'Unseen-unseen ROC-AUC: {}\n\n'.format(ensemble_unseen.calc_roc_auc())
        f.write()
    return


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
