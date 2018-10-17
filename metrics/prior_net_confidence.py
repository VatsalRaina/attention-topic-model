from __future__ import print_function, division

import sys
import os
import numpy as np
from numpy import ma
import scipy.stats
import math
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

from metrics.seed_spread_confidence import get_label_predictions

parser = argparse.ArgumentParser(description='Plot useful graphs for evaluation.')
parser.add_argument('model_parent_dir', type=str, help='Path to model directory')
parser.add_argument('--savedir', type=str, default='./',
                    help='Path to directory where to save the plots')
parser.add_argument('--rel_labels_path', type=str, default='eval4_naive/labels-probs.txt')

matplotlib.rcParams['savefig.dpi'] = 200


def main(args):



if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
