import os
import sys
import random

import numpy as np
import tensorflow as tf
from tensorflow.contrib.data import group_by_window
import tensorflow.contrib.slim as slim
try:
    import cPickle as pickle
except:
    import pickle


class BaseModel(object):
    def __init__(self, network_architecture, name=None, save_path=None, load_path=None, debug_mode=0, seed=100):

        # Misc book-keeping parameters
        self._debug_mode = debug_mode
        self._save_path = save_path
        self._name = name

        # Specify default network. Generic enough for any model.
        self.network_architecture = network_architecture

        # Add check that network architecture has all the necessary parameters???
        # If new model, need to update architecture
        if load_path is None:
            assert network_architecture is not None
        else:
            # Load data (deserialize) architecture from path
            arch_path = os.path.join(load_path, 'model/net_arch.pickle')
            with open(arch_path, 'rb') as handle:
                self.network_architecture = pickle.load(handle)

        if (os.path.isfile(os.path.join(self._save_path, 'LOG.txt')) or os.path.isfile(
                os.path.join(self._save_path, 'model/weights.ckpt')) or os.path.isfile(
                os.path.join(self._save_path, 'model/net_arch.pickle'))) and load_path is None:
            print 'Model exists in directory - exiting.'
            sys.exit()
        if load_path is None:
            with open(os.path.join(self._save_path, 'LOG.txt'), 'w') as f:
                f.write('Creating Grader Model with configuration:\n')
                f.write('----------------------------------------------------------\n')
                for key in sorted(self.network_architecture.keys()):
                    f.write(key + ': ' + str(self.network_architecture[key]) + '\n')
                f.write('----------------------------------------------------------\n')

        # Parameters for training
        self._seed = seed
        self.initializer = self.network_architecture['initializer']

        # Tensorflow graph bookeeping
        self._graph = tf.Graph()
        # Construct Graph
        with self._graph.as_default():
            config = tf.ConfigProto()
            config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
            tf.set_random_seed(self._seed)
            np.random.seed(self._seed)
            random.seed(self._seed)
            self.sess = tf.Session(config=config)

    # Model loading/saving functions

    def save(self, step=None):
        """ Saves model and parameters to self._save_path """
        with self._graph.as_default():
            path = os.path.join(self._save_path, 'model/weights.ckpt')
            if step is not None:
                self._saver.save(self.sess, path, global_step=step)
            else:
                self._saver.save(self.sess, path)
            with open(os.path.join(self._save_path, 'LOG.txt'), 'a') as f:
                f.write('Saving Model to: ' + path + '\n')

        # Pickle network architecture into a file.
        path = os.path.join(self._save_path, 'model/net_arch.pickle')
        with open(path, 'wb') as handle:
            pickle.dump(self.network_architecture, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, load_path, step=None):
        with self._graph.as_default():
            # If necessary, restore model from previous
            print 'loading model...'
            weights_path = 'model/weights.ckpt'
            if step is not None:
                weights_path = 'model/weights.ckpt-' + str(step)

            path = os.path.join(load_path, 'model/net_arch.pickle')
            with open(path, 'rb') as handle:
                self.network_architecture = pickle.load(handle)

            path = os.path.join(load_path, weights_path)
            with open(os.path.join(self._save_path, 'LOG.txt'), 'a') as f:
                f.write('Restoring Model paratemters from: ' + path + '\n')
            self._saver.restore(self.sess, path)

    def _load_variables(self, load_scope, new_scope, load_path, trainable=False):
        # Restore parameters to DDN we are sampling from...
        if trainable:
            model_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, ".*" + new_scope + ".*")
        else:
            model_variables = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, ".*" + new_scope + ".*")
        dict = {}
        for model_var in model_variables:
            #print model_var.op.name, model_var.op.name.replace(new_scope, load_scope)
            dict[model_var.op.name.replace(new_scope, load_scope)] = model_var
        sampling_saver = tf.train.Saver(dict)
        param_path = os.path.join(load_path, 'model/weights.ckpt')
        sampling_saver.restore(self.sess, param_path)

    # Data Loading Pipleline functions

    def _construct_dataset_from_tfrecord(self,
                                         filenames,
                                         _parse_func,
                                         _map_func,
                                         _batch_func,
                                         batch_size,
                                         capacity_mul=1000,
                                         num_threads=4,
                                         augment=False,
                                         train=False):
        with tf.device('/cpu:0'):
            capacity = capacity_mul*batch_size
            dataset = tf.data.TFRecordDataset(filenames)
            dataset = dataset.map(_parse_func, num_parallel_calls=num_threads).cache()

            # Apply any other possible mapping possible mapping.
            dataset = _map_func(dataset, num_threads, capacity, augment=augment)

            # Apply shuffle dataset and repeat it indefinitely
            if train: dataset = dataset.shuffle(capacity, self._seed).repeat()

            # Apply shuffle dataset and repeat it indefinitely
            dataset = _batch_func(dataset, batch_size)

            if train:
                # Create an iterator for the dataset
                iterator = dataset.make_one_shot_iterator()
                return iterator.get_next(name='input_data')
            else:
                iterator = dataset.make_initializable_iterator()
                return iterator

    def _parse_func(self, example_proto):
        contexts, features = tf.parse_single_sequence_example(
            serialized=example_proto,
            context_features={"targets": tf.FixedLenFeature([], tf.float32),
                              "grade": tf.FixedLenFeature([], tf.float32),
                              "spkr" : tf.FixedLenFeature([], tf.string),
                              "q_id": tf.FixedLenFeature([], tf.int64)},
            sequence_features={'response': tf.FixedLenSequenceFeature([], dtype=tf.int64),
                               'prompt': tf.FixedLenSequenceFeature([], dtype=tf.int64)})

        return contexts['targets'], contexts['q_id'], features['response'], features['prompt']

    def _map_func(self, dataset, num_threads, capacity, augment=None):

        dataset =  dataset.map(lambda targets, q_id, resp, prompt: (targets,
                                                                    tf.cast(q_id, dtype=tf.int32),
                                                                    tf.cast(resp, dtype=tf.int32),
                                                                    tf.cast(prompt, dtype=tf.int32)),
                               num_parallel_calls=num_threads).prefetch(capacity)

        return dataset.map(lambda targets, q_id, resp, prompt: (targets, q_id, resp, tf.size(resp), prompt, tf.size(prompt)),
                           num_parallel_calls=num_threads).prefetch(capacity)

    def _batch_func(self, dataset, batch_size, num_buckets=10, bucket_width=10):
        # Bucket by source sequence length (buckets for lengths 0-9, 10-19, ...)
        def batching_func(x):
            return x.padded_batch(
                batch_size,
                # The first three entries are the source and target line rows;
                # these have unknown-length vectors.  The last two entries are
                # the source and target row sizes; these are scalars.
                padded_shapes=(
                    tf.TensorShape([]), # targets -- unused
                    tf.TensorShape([]), # q_id -- unused
                    tf.TensorShape([None]),   # resp
                    tf.TensorShape([]), # resp len -- unused
                    tf.TensorShape([None]),  # prompt
                    tf.TensorShape([])),  # prompt len -- unused
                padding_values=(
                    0.0, # targets -- unused
                    np.int32(0), # q_id -- unused
                    np.int32(0), # resp
                    np.int32(0), # resp len -- unused
                    np.int32(0),  # resp len -- unused
                    np.int32(0)))#.filter(lambda targets, q_id, resp, size, prompt: tf.equal(tf.size(size), batch_size))  # prompt

        def key_func(unused_1, unused_2, unused_3, resp_len, unused_4, unused_5):
            # Calculate bucket_width by maximum source sequence length.
            # Pairs with length [0, bucket_width) go to bucket 0, length
            # [bucket_width, 2 * bucket_width) go to bucket 1, etc.  Pairs with length
            # over ((num_bucket-1) * bucket_width) words all go into the last bucket.

            # Bucket sentence pairs by the length of their response
            bucket_id = resp_len // bucket_width
            return tf.to_int64(tf.minimum(num_buckets, bucket_id))

        def reduce_func(unused_key, windowed_data):
            return batching_func(windowed_data)

        batched_dataset = dataset.apply(group_by_window(key_func=key_func,
                                                  reduce_func=reduce_func,
                                                  window_size=batch_size))

        return batched_dataset

    # Trainnig cost/attention sampling functions

    def _sampling_function(self, targets, q_ids, unigram_path, batch_size, n_samples, name, distortion=1.0):
        sampled_indecies, _, _ = tf.nn.fixed_unigram_candidate_sampler(tf.cast(tf.expand_dims(q_ids, axis=1), dtype=tf.int64),
                                                                       num_true=1,
                                                                       num_sampled=batch_size * n_samples,
                                                                       unique=False,
                                                                       distortion=distortion,
                                                                       range_max=self.network_architecture['n_topics'],
                                                                       vocab_file=unigram_path,
                                                                       seed=self._seed,
                                                                       name='Unigram_Sampler_'+name)
        sampled_indecies=tf.cast(sampled_indecies, dtype=tf.int32)
        targets_sampled = tf.where(tf.equal(tf.tile(q_ids, [n_samples]), sampled_indecies),
                                   tf.ones(shape=[batch_size * n_samples], dtype=tf.float32),
                                   tf.zeros(shape=[batch_size * n_samples], dtype=tf.float32))

        q_ids = tf.concat([q_ids, sampled_indecies], axis=0)
        targets = tf.concat([targets, targets_sampled], axis=0)
        targets = tf.expand_dims(targets, axis=1)

        return targets, q_ids

    def _bahdanau_attention(self, memory, seq_lens, maxlen, query, size, batch_size, name='Attention'):
        WD = self.network_architecture['L2']
        with tf.variable_scope(name) as scope:
            with slim.arg_scope([slim.model_variable],
                                initializer=self.network_architecture['initializer'](self._seed),
                                regularizer=slim.l2_regularizer(WD),
                                device='/GPU:0'):
                # Define Attention Parameters
                v = slim.model_variable('v', shape=[1, size])
                U = slim.model_variable('u', shape=[size, size])
                W = slim.model_variable('w', shape=[size, size])
                biases = slim.model_variable('biases', shape=[size], initializer=tf.constant_initializer(0.1))

                tmp_a = tf.reshape(memory, [-1, size])
                tmp_a = tf.matmul(tmp_a, U)
                tmp_a = tf.reshape(tmp_a, [batch_size, -1, size])
                tmp_q = tf.matmul(query, W)
                tmp_q = tf.expand_dims(tmp_q, axis=1)
                tmp = tf.nn.tanh(tmp_q + tmp_a + biases)
                tmp = tf.reshape(tmp, [-1, size])
                tmp = tf.matmul(tmp, v, transpose_b=True)
                tmp = tf.reshape(tmp, [batch_size, -1])
                mask = tf.sequence_mask(seq_lens, maxlen=maxlen, dtype=tf.float32)
                a = tf.exp(tmp) * mask
                attention = a / tf.reduce_sum(a, axis=1, keep_dims=True)
                outputs = tf.reduce_sum(tf.expand_dims(attention, 2) * memory, axis=1)

                return outputs, attention

    def _construct_xent_cost(self, targets, logits, pos_weight, is_training=False):
        print 'Constructing XENT cost'
        cost = tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=targets, pos_weight=pos_weight,
                                                     name='total_xentropy_per_batch')) / float(pos_weight)

        if self._debug_mode > 1:
            tf.scalar_summary('XENT', cost)

        if is_training:
            tf.add_to_collection('losses', cost)
            # The total loss is defined as the target loss plus all of the weight
            # decay terms (L2 loss).
            total_cost = tf.add_n(tf.get_collection('losses'), name='total_cost')
            return cost, total_cost
        else:
            return cost
