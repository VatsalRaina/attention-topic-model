import os
import time

import tensorflow as tf

import numpy as np

import context
from core.basemodel import BaseModel
import core.utilities.utilities as ut


class Ensemble:
    def __init__(self, base_learner_list, combination_fn=tf.reduce_mean):
        self.base_learner_list = base_learner_list
        self.combination_fn = combination_fn
        self.ensemble_size = len(base_learner_list)

    def infere(self, a_input, a_seqlens, n_samples, q_input, q_seqlens, maxlen, batch_size, keep_prob=1.0):
        all_probabilities = []
        for base_learner in self.base_learner_list:
            base_learner_probs = base_learner.infere(a_input=a_input,
                                                     a_seqlens=a_seqlens,
                                                     n_samples=n_samples,
                                                     q_input=q_input,
                                                     q_seqlens=q_seqlens,
                                                     maxlen=maxlen,
                                                     batch_size=batch_size,
                                                     keep_prob=keep_prob)
            all_probabilities.append(base_learner_probs)
        ensemble_probs = self.combination_fn(np.stack(all_probabilities), axis=1)
        return ensemble_probs
