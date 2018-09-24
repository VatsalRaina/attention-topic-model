import os
import time
from __future__ import print_function

import matplotlib
import numpy as np

matplotlib.use('agg')
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score as roc

import tensorflow as tf
import tensorflow.contrib.slim as slim

import context
from core.basemodel import BaseModel
import core.utilities.utilities as util


class AttentionTopicModel(BaseModel):
    def __init__(self, network_architecture=None, name=None, save_path='./', load_path=None, debug_mode=0, seed=100,
                 epoch=None):

        BaseModel.__init__(self, network_architecture=network_architecture, seed=seed, name=name, save_path=save_path,
                           load_path=load_path, debug_mode=debug_mode)

        with self._graph.as_default():
            with tf.variable_scope('input') as scope:
                self._input_scope = scope
                self.x_a = tf.placeholder(tf.int32, [None, None])
                self.x_q = tf.placeholder(tf.int32, [None, None])
                self.qlens = tf.placeholder(tf.int32, [None])
                self.alens = tf.placeholder(tf.int32, [None])
                self.y = tf.placeholder(tf.float32, [None, 1])
                self.maxlen = tf.placeholder(dtype=tf.int32, shape=[])

                self.dropout = tf.placeholder(tf.float32, [])
                self.batch_size = tf.placeholder(tf.int32, [])

            with tf.variable_scope('atm') as scope:
                self._model_scope = scope
                self._predictions, \
                self._probabilities, \
                self._logits, \
                self.attention, = self._construct_network(a_input=self.x_a,
                                                          a_seqlens=self.alens,
                                                          q_input=self.x_q,
                                                          q_seqlens=self.qlens,
                                                          n_samples=0,
                                                          maxlen=self.maxlen,
                                                          batch_size=self.batch_size,
                                                          keep_prob=self.dropout)

            self._saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)

        if load_path == None:
            with self._graph.as_default():
                init = tf.global_variables_initializer()
                self.sess.run(init)

                # If necessary, restore model from previous
        elif load_path != None:
            self.load(load_path=load_path, step=epoch)

    # def _construct_network(self, a_input, a_seqlens, n_samples, q_input, q_seqlens, maxlen,
    #                        batch_size, keep_prob=1.0):
    #     """ Construct RNNLM network
    #     Args:
    #       ?
    #     Returns:
    #       predictions, logits
    #     """
    #
    #     L2 = self.network_architecture['L2']
    #     initializer = self.network_architecture['initializer']
    #
    #     # Question Encoder RNN
    #     with tf.variable_scope('Embeddings', initializer=initializer(self._seed)) as scope:
    #         embedding = slim.model_variable('word_embedding',
    #                                         shape=[self.network_architecture['n_in'],
    #                                                self.network_architecture['n_ehid']],
    #                                         initializer=tf.truncated_normal_initializer(stddev=0.1),
    #                                         regularizer=slim.l2_regularizer(L2),
    #                                         device='/GPU:0')
    #         a_inputs = tf.nn.dropout(tf.nn.embedding_lookup(embedding, a_input, name='embedded_data'),
    #                                  keep_prob=keep_prob, seed=self._seed + 1)
    #         q_inputs = tf.nn.dropout(tf.nn.embedding_lookup(embedding, q_input, name='embedded_data'),
    #                                  keep_prob=keep_prob, seed=self._seed + 2)
    #
    #     with tf.variable_scope('RNN_Q', initializer=initializer(self._seed)) as scope:
    #         cell_fw = tf.contrib.rnn.BasicLSTMCell(num_units=self.network_architecture['n_phid'],
    #                                                forget_bias=1.0,
    #                                                activation=self.network_architecture['r_activation_fn'],
    #                                                state_is_tuple=True)
    #         cell_bw = tf.contrib.rnn.BasicLSTMCell(num_units=self.network_architecture['n_phid'],
    #                                                forget_bias=1.0,
    #                                                activation=self.network_architecture['r_activation_fn'],
    #                                                state_is_tuple=True)
    #
    #         cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=keep_prob)
    #         cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=keep_prob)
    #
    #         initial_state_fw = cell_fw.zero_state(batch_size=batch_size * (n_samples + 1), dtype=tf.float32)
    #         initial_state_bw = cell_bw.zero_state(batch_size=batch_size * (n_samples + 1), dtype=tf.float32)
    #
    #         _, state = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
    #                                                    cell_bw=cell_bw,
    #                                                    inputs=q_inputs,
    #                                                    sequence_length=q_seqlens,
    #                                                    initial_state_fw=initial_state_fw,
    #                                                    initial_state_bw=initial_state_bw,
    #                                                    dtype=tf.float32,
    #                                                    parallel_iterations=32,
    #                                                    scope=scope)
    #
    #         question_embeddings = tf.concat([state[0][1], state[1][1]], axis=1)
    #         question_embeddings = tf.nn.dropout(question_embeddings, keep_prob=keep_prob, seed=self._seed)
    #
    #     # Response Encoder RNN
    #     with tf.variable_scope('RNN_A', initializer=initializer(self._seed)) as scope:
    #         cell_fw = tf.contrib.rnn.BasicLSTMCell(num_units=self.network_architecture['n_rhid'],
    #                                                forget_bias=1.0,
    #                                                activation=self.network_architecture['r_activation_fn'],
    #                                                state_is_tuple=True)
    #         cell_bw = tf.contrib.rnn.BasicLSTMCell(num_units=self.network_architecture['n_rhid'],
    #                                                forget_bias=1.0,
    #                                                activation=self.network_architecture['r_activation_fn'],
    #                                                state_is_tuple=True)
    #
    #         initial_state_fw = cell_fw.zero_state(batch_size=batch_size, dtype=tf.float32)
    #         initial_state_bw = cell_bw.zero_state(batch_size=batch_size, dtype=tf.float32)
    #
    #         cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=keep_prob)
    #         cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=keep_prob)
    #
    #         outputs, state = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
    #                                                          cell_bw=cell_bw,
    #                                                          inputs=a_inputs,
    #                                                          sequence_length=a_seqlens,
    #                                                          initial_state_fw=initial_state_fw,
    #                                                          initial_state_bw=initial_state_bw,
    #                                                          dtype=tf.float32,
    #                                                          parallel_iterations=32,
    #                                                          scope=scope)
    #
    #         a_seqlens = tf.tile(a_seqlens, [n_samples + 1])
    #         outputs = tf.concat([outputs[0], outputs[1]], axis=2)
    #         outputs = tf.tile(outputs, [1 + n_samples, 1, 1])
    #
    #     print outputs.get_shape(), a_seqlens.get_shape()
    #
    #     hidden, attention = self._bahdanau_attention(memory=outputs, seq_lens=a_seqlens, maxlen=maxlen,
    #                                                  query=question_embeddings,
    #                                                  size=2 * self.network_architecture['n_rhid'],
    #                                                  batch_size=batch_size * (n_samples + 1))
    #
    #     with tf.variable_scope('Grader') as scope:
    #         for layer in xrange(self.network_architecture['n_flayers']):
    #             hidden = slim.fully_connected(hidden,
    #                                           self.network_architecture['n_fhid'],
    #                                           activation_fn=self.network_architecture['f_activation_fn'],
    #                                           weights_regularizer=slim.l2_regularizer(L2),
    #                                           scope="hidden_layer_" + str(layer))
    #             hidden = tf.nn.dropout(hidden, keep_prob=keep_prob, seed=self._seed + layer)
    #
    #         logits = slim.fully_connected(hidden,
    #                                       self.network_architecture['n_out'],
    #                                       activation_fn=None,
    #                                       scope="output_layer")
    #         probabilities = self.network_architecture['output_fn'](logits)
    #         predictions = tf.cast(tf.round(probabilities), dtype=tf.float32)
    #
    #     return predictions, probabilities, logits, attention

    def _construct_network(self, a_input, a_seqlens, n_samples, q_input, q_seqlens, maxlen, batch_size, keep_prob=1.0):
        """ Construct RNNLM network
        Args:
          ?
        Returns:
          predictions, probabilities, logits, attention
        """

        L2 = self.network_architecture['L2']
        initializer = self.network_architecture['initializer']

        # Question Encoder RNN
        with tf.variable_scope('Embeddings', initializer=initializer(self._seed)) as scope:
            embedding = slim.model_variable('word_embedding',
                                            shape=[self.network_architecture['n_in'],
                                                   self.network_architecture['n_ehid']],
                                            initializer=tf.truncated_normal_initializer(stddev=0.1),
                                            regularizer=slim.l2_regularizer(L2),
                                            device='/GPU:0')
            a_inputs = tf.nn.dropout(tf.nn.embedding_lookup(embedding, a_input, name='embedded_data'),
                                     keep_prob=keep_prob, seed=self._seed + 1)
            q_inputs = tf.nn.dropout(tf.nn.embedding_lookup(embedding, q_input, name='embedded_data'),
                                     keep_prob=keep_prob, seed=self._seed + 2)

            q_inputs_fw = tf.transpose(q_inputs, [1, 0, 2])
            q_inputs_bw = tf.transpose(tf.reverse_sequence(q_inputs, seq_lengths=q_seqlens, seq_axis=1, batch_axis=0),
                                       [1, 0, 2])

            a_inputs_fw = tf.transpose(a_inputs, [1, 0, 2])
            a_inputs_bw = tf.transpose(tf.reverse_sequence(a_inputs, seq_lengths=a_seqlens, seq_axis=1, batch_axis=0),
                                       [1, 0, 2])

        # Prompt Encoder RNN
        with tf.variable_scope('RNN_Q_FW', initializer=initializer(self._seed)) as scope:
            rnn_fw = tf.contrib.rnn.LSTMBlockFusedCell(num_units=self.network_architecture['n_phid'])
            _, state_fw = rnn_fw(q_inputs_fw, sequence_length=q_seqlens, dtype=tf.float32)

        with tf.variable_scope('RNN_Q_BW', initializer=initializer(self._seed)) as scope:
            rnn_bw = tf.contrib.rnn.LSTMBlockFusedCell(num_units=self.network_architecture['n_phid'])
            _, state_bw = rnn_bw(q_inputs_bw, sequence_length=q_seqlens, dtype=tf.float32)

        question_embeddings = tf.concat([state_fw[1], state_bw[1]], axis=1)
        question_embeddings = tf.nn.dropout(question_embeddings, keep_prob=keep_prob, seed=self._seed)

        # Response Encoder RNN
        with tf.variable_scope('RNN_A_FW', initializer=initializer(self._seed)) as scope:
            rnn_fw = tf.contrib.rnn.LSTMBlockFusedCell(num_units=self.network_architecture['n_phid'])
            outputs_fw, _ = rnn_fw(a_inputs_fw, sequence_length=a_seqlens, dtype=tf.float32)

        with tf.variable_scope('RNN_A_BW', initializer=initializer(self._seed)) as scope:
            rnn_bw = tf.contrib.rnn.LSTMBlockFusedCell(num_units=self.network_architecture['n_phid'])
            outputs_bw, _ = rnn_bw(a_inputs_bw, sequence_length=a_seqlens, dtype=tf.float32)

        outputs = tf.concat([outputs_fw, outputs_bw], axis=2)
        outputs = tf.transpose(outputs, [1, 0, 2])
        outputs = tf.nn.dropout(outputs, keep_prob=keep_prob, seed=self._seed)

        a_seqlens = tf.tile(a_seqlens, [n_samples + 1])
        outputs = tf.tile(outputs, [1 + n_samples, 1, 1])

        hidden, attention = self._bahdanau_attention(memory=outputs, seq_lens=a_seqlens, maxlen=maxlen,
                                                     query=question_embeddings,
                                                     size=2 * self.network_architecture['n_rhid'],
                                                     batch_size=batch_size * (n_samples + 1))

        with tf.variable_scope('Grader') as scope:
            for layer in xrange(self.network_architecture['n_flayers']):
                hidden = slim.fully_connected(hidden,
                                              self.network_architecture['n_fhid'],
                                              activation_fn=self.network_architecture['f_activation_fn'],
                                              weights_regularizer=slim.l2_regularizer(L2),
                                              scope="hidden_layer_" + str(layer))
                hidden = tf.nn.dropout(hidden, keep_prob=keep_prob, seed=self._seed + layer)

            logits = slim.fully_connected(hidden,
                                          self.network_architecture['n_out'],
                                          activation_fn=None,
                                          scope="output_layer")
            probabilities = self.network_architecture['output_fn'](logits)
            predictions = tf.cast(tf.round(probabilities), dtype=tf.float32)

        return predictions, probabilities, logits, attention

    def fit(self,
            train_data,
            valid_data,
            load_path,
            topics,
            topic_lens,
            unigram_path,
            train_size=100,
            valid_size=100,
            learning_rate=1e-2,
            lr_decay=0.8,
            dropout=1.0,
            batch_size=50,
            distortion=1.0,
            optimizer=tf.train.AdamOptimizer,
            optimizer_params={},
            n_epochs=30,
            n_samples=1,  # Number of negative samples to generate per positive sample
            epoch=1):
        with self._graph.as_default():
            # Compute number of training examples and batch size
            n_examples = train_size * (1 + n_samples)
            n_batches = n_examples / (batch_size * (1 + n_samples))

            # If some variables have been initialized - get them into a set
            temp = set(tf.global_variables())

            # Define Global step for training
            global_step = tf.Variable(0, trainable=False, name='global_step')

            # Set up inputs
            with tf.variable_scope(self._input_scope, reuse=True) as scope:
                # Construct training data queues
                targets, \
                q_ids, \
                responses, \
                response_lengths, _, _ = self._construct_dataset_from_tfrecord([train_data],
                                                                               self._parse_func,
                                                                               self._map_func,
                                                                               self._batch_func,
                                                                               batch_size,
                                                                               train=True,
                                                                               capacity_mul=1000,
                                                                               num_threads=8)

                valid_iterator = self._construct_dataset_from_tfrecord([valid_data],
                                                                       self._parse_func,
                                                                       self._map_func,
                                                                       self._batch_func,
                                                                       batch_size,
                                                                       train=False,
                                                                       capacity_mul=100,
                                                                       num_threads=1)
                valid_targets, \
                valid_q_ids, \
                valid_responses, \
                valid_response_lengths, _, _ = valid_iterator.get_next(name='valid_data')

                targets, q_ids = self._sampling_function(targets=targets,
                                                         q_ids=q_ids,
                                                         unigram_path=unigram_path,
                                                         batch_size=batch_size,
                                                         n_samples=n_samples,
                                                         name='train',
                                                         distortion=distortion)
                valid_targets, valid_q_ids = self._sampling_function(targets=valid_targets,
                                                                     q_ids=valid_q_ids,
                                                                     unigram_path=unigram_path,
                                                                     batch_size=batch_size,
                                                                     n_samples=n_samples,
                                                                     name='valid',
                                                                     distortion=1.0)

            topics = tf.convert_to_tensor(topics, dtype=tf.int32)
            topic_lens = tf.convert_to_tensor(topic_lens, dtype=tf.int32)

            prompts = tf.nn.embedding_lookup(topics, q_ids, name='train_prompot_loopkup')
            prompt_lens = tf.gather(topic_lens, q_ids)

            valid_prompts = tf.nn.embedding_lookup(topics, valid_q_ids, name='valid_prompot_loopkup')
            valid_prompt_lens = tf.gather(topic_lens, valid_q_ids)

            # Construct Training & Validation models
            with tf.variable_scope(self._model_scope, reuse=True) as scope:
                trn_predictions, \
                trn_probabilities, \
                trn_logits, _, = self._construct_network(a_input=responses,
                                                         a_seqlens=response_lengths,
                                                         n_samples=n_samples,
                                                         q_input=prompts,
                                                         q_seqlens=prompt_lens,
                                                         maxlen=tf.reduce_max(response_lengths),
                                                         batch_size=batch_size,
                                                         keep_prob=self.dropout)

                valid_predictions, \
                valid_probabilities, \
                valid_logits, \
                valid_attention = self._construct_network(a_input=valid_responses,
                                                          a_seqlens=valid_response_lengths,
                                                          n_samples=n_samples,
                                                          q_input=valid_prompts,
                                                          q_seqlens=valid_prompt_lens,
                                                          maxlen=tf.reduce_max(valid_response_lengths),
                                                          batch_size=batch_size,
                                                          keep_prob=1.0)

            # Construct XEntropy training costs
            trn_cost, total_loss = self._construct_xent_cost(targets=targets,
                                                             logits=trn_logits,
                                                             pos_weight=float(n_samples),
                                                             is_training=True)
            evl_cost = self._construct_xent_cost(targets=valid_targets,
                                                 logits=valid_logits,
                                                 pos_weight=float(n_samples),
                                                 is_training=False)

            train_op = util.create_train_op(total_loss=total_loss,
                                            learning_rate=learning_rate,
                                            optimizer=optimizer,
                                            optimizer_params=optimizer_params,
                                            n_examples=n_examples,
                                            batch_size=batch_size,
                                            learning_rate_decay=lr_decay,
                                            global_step=global_step,
                                            clip_gradient_norm=10.0,
                                            summarize_gradients=False)

            # Intialize only newly created variables, as opposed to reused - allows for finetuning and transfer learning :)
            init = tf.variables_initializer(set(tf.global_variables()) - temp)
            self.sess.run(init)

            if load_path != None:
                self._load_variables(load_scope='model/Embeddings/word_embedding',
                                     new_scope='atm/Embeddings/word_embedding', load_path=load_path)

            # Update Log with training details
            with open(os.path.join(self._save_path, 'LOG.txt'), 'a') as f:
                format_str = (
                    'Learning Rate: %f\nLearning Rate Decay: %f\nBatch Size: %d\nValid Size: %d\nOptimizer: %s\nDropout: %f\nSEED: %i\n')
                f.write(format_str % (
                    learning_rate, lr_decay, batch_size, valid_size, str(optimizer), dropout, self._seed) + '\n\n')

            format_str = (
                'Epoch %d, Train Loss = %.2f, Valid Loss = %.2f, Valid ROC = %.2f, (%.1f examples/sec; %.3f ' 'sec/batch)')
            print("Starting Training!\n-----------------------------")
            start_time = time.time()
            for epoch in xrange(epoch + 1, epoch + n_epochs + 1):
                # Run Training Loop
                loss = 0.0
                batch_time = time.time()
                for batch in xrange(n_batches):
                    _, loss_value = self.sess.run([train_op, trn_cost], feed_dict={self.dropout: dropout})
                    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                    loss += loss_value

                duration = time.time() - batch_time
                loss /= n_batches
                examples_per_sec = batch_size / duration
                sec_per_epoch = float(duration)

                # Run Validation Loop
                eval_loss = 0.0
                valid_probs = None
                vld_targets = None
                total_size = 0
                self.sess.run(valid_iterator.initializer)
                while True:
                    try:
                        batch_eval_loss, \
                        batch_valid_preds, \
                        batch_valid_probs, \
                        batch_attention, \
                        batch_valid_targets = self.sess.run([evl_cost,
                                                             valid_predictions,
                                                             valid_probabilities,
                                                             valid_attention,
                                                             valid_targets])
                        size = batch_valid_probs.shape[0]
                        eval_loss += float(size) * batch_eval_loss
                        if valid_probs is None:
                            valid_probs = batch_valid_probs
                            vld_targets = batch_valid_targets
                        else:
                            valid_probs = np.concatenate((valid_probs, batch_valid_probs), axis=0)
                            vld_targets = np.concatenate((vld_targets, batch_valid_targets), axis=0)
                        total_size += size
                    except:  # tf.errors.OutOfRangeError:
                        break

                eval_loss = eval_loss / float(total_size)
                roc_score = roc(np.squeeze(vld_targets), np.squeeze(valid_probs))

                # Summarize Epoch
                with open(os.path.join(self._save_path, 'LOG.txt'), 'a') as f:
                    f.write(format_str % (epoch, loss, eval_loss, roc_score, examples_per_sec, sec_per_epoch) + '\n')
                print(format_str % (epoch, loss, eval_loss, roc_score, examples_per_sec, sec_per_epoch))
                self.save(step=epoch)

            # Finish Training
            duration = time.time() - start_time
            with open(os.path.join(self._save_path, 'LOG.txt'), 'a') as f:
                format_str = ('Training took %.3f sec')
                f.write('\n' + format_str % (duration) + '\n')
                f.write('----------------------------------------------------------\n')
            print(format_str % (duration))
            self.save()

    def predict(self, test_pattern, batch_size=20, cache_inputs=False, apply_bucketing=True):
        """
        Run inference on a trained model on a dataset.
        :param test_pattern: filepath to dataset to run inference/evaluation on
        :param batch_size: int
        :param cache_inputs: Whether to save the response, prompts, response lengths, and prompt lengths in
        text form together with the predictions. Useful, since bucketing changes the order of the files and this allows
        to investigate which prediction corresponds to which prompt/response pair
        :param apply_bucketing: bool, whether to apply bucketing, i.e. group examples by their response length to
        minimise the overhead associated with zero-padding. If False, the examples will be evaluated in the original
        order as read from the file.

        :return: Depends on whether the inputs are being cached. If cache_inputs=False:
        returns test_loss, test_probabilities_array, test_true_labels_array
        If cache_inputs=True:
        returns test_loss, test_probabilities_array, test_true_labels_array, test_response_lengths,
                test_prompt_lengths, test_responses_list, test_prompts_list
        """
        with self._graph.as_default():
            test_files = tf.gfile.Glob(test_pattern)
            if apply_bucketing:
                batching_function = self._batch_func
            else:
                batching_function = self._batch_func_without_bucket
            test_iterator = self._construct_dataset_from_tfrecord(test_files,
                                                                  self._parse_func,
                                                                  self._map_func,
                                                                  batching_function,
                                                                  batch_size=batch_size,
                                                                  train=False,
                                                                  capacity_mul=100,
                                                                  num_threads=1)
            test_targets, \
            test_q_ids, \
            test_responses, \
            test_response_lengths, test_prompts, test_prompt_lens = test_iterator.get_next(name='valid_data')

            with tf.variable_scope(self._model_scope, reuse=True) as scope:
                test_predictions, \
                test_probabilities, \
                test_logits, \
                test_attention = self._construct_network(a_input=test_responses,
                                                         a_seqlens=test_response_lengths,
                                                         n_samples=0,
                                                         q_input=test_prompts,
                                                         q_seqlens=test_prompt_lens,
                                                         maxlen=tf.reduce_max(test_response_lengths),
                                                         batch_size=batch_size,
                                                         keep_prob=1.0)

            loss = self._construct_xent_cost(targets=test_targets, logits=tf.squeeze(test_logits), pos_weight=1.0,
                                             is_training=False)
            self.sess.run(test_iterator.initializer)
            if cache_inputs:
                return self._predict_loop_with_caching(loss, test_probabilities, test_targets,
                                                       test_responses, test_response_lengths, test_prompts,
                                                       test_prompt_lens)
            else:
                return self._predict_loop(loss, test_probabilities, test_targets)

    def _predict_loop_with_caching(self, loss, test_probabilities, test_targets, test_responses, test_response_lengths,
                                   test_prompts, test_prompt_lens):
        test_loss = 0.0
        total_size = 0
        count = 0

        # Variables for storing the batch_ordered data
        test_responses_list = []
        test_prompts_list = []
        while True:
            try:
                batch_eval_loss, \
                batch_test_probs, \
                batch_test_targets, \
                batch_responses, \
                batch_response_lengths, \
                batch_prompts, \
                batch_prompt_lens = self.sess.run([loss,
                                                   test_probabilities,
                                                   test_targets,
                                                   test_responses,
                                                   test_response_lengths,
                                                   test_prompts,
                                                   test_prompt_lens])

                size = batch_test_probs.shape[0]
                test_loss += float(size) * batch_eval_loss
                if count == 0:
                    test_probs_arr = batch_test_probs  # shape: (num_batches, 1)
                    test_labels_arr = batch_test_targets[:, np.newaxis]  # becomes shape: (num_batches, 1)
                    test_response_lens_arr = batch_response_lengths[:, np.newaxis]  # becomes shape: (num_batches, 1)
                    test_prompt_lens_arr = batch_prompt_lens[:, np.newaxis]  # becomes shape: (num_batches, 1)
                else:
                    test_probs_arr = np.concatenate((test_probs_arr, batch_test_probs), axis=0)
                    test_labels_arr = np.concatenate((test_labels_arr, batch_test_targets[:, np.newaxis]), axis=0)
                    test_response_lens_arr = np.concatenate(
                        (test_response_lens_arr, batch_response_lengths[:, np.newaxis]), axis=0)
                    test_prompt_lens_arr = np.concatenate((test_prompt_lens_arr, batch_prompt_lens[:, np.newaxis]),
                                                          axis=0)
                test_responses_list.extend(list(batch_responses))  # List of numpy arrays!
                test_prompts_list.extend(list(batch_prompts))  # List of numpy arrays!

                total_size += size
                count += 1
            except:  # todo: tf.errors.OutOfRangeError:
                break

        test_loss = test_loss / float(total_size)

        return (test_loss,
                test_probs_arr,
                test_labels_arr.astype(np.int32),
                test_response_lens_arr.astype(np.int32),
                test_prompt_lens_arr.astype(np.int32),
                test_responses_list,
                test_prompts_list)

    def _predict_loop(self, loss, test_probabilities, test_targets):
        test_loss = 0.0
        total_size = 0
        count = 0

        # Variables for storing the batch_ordered data
        while True:
            try:
                batch_eval_loss, \
                batch_test_probs, \
                batch_test_targets = self.sess.run([loss,
                                                    test_probabilities,
                                                    test_targets])

                size = batch_test_probs.shape[0]
                test_loss += float(size) * batch_eval_loss
                if count == 0:
                    test_probs_arr = batch_test_probs  # shape: (num_batches, 1)
                    test_labels_arr = batch_test_targets[:, np.newaxis]  # becomes shape: (num_batches, 1)
                else:
                    test_probs_arr = np.concatenate((test_probs_arr, batch_test_probs), axis=0)
                    test_labels_arr = np.concatenate((test_labels_arr, batch_test_targets[:, np.newaxis]), axis=0)

                total_size += size
                count += 1
            except:  # todo: tf.errors.OutOfRangeError:
                break

        test_loss = test_loss / float(total_size)

        return (test_loss,
                test_probs_arr,
                test_labels_arr.astype(np.int32))

        # def rank(self, X, topics, name=None):
        #     with self._graph.as_default():
        #         test_probs = None
        #         batch_size = len(topics[1])
        #         test_probs = np.zeros(shape=(len(X[0]), batch_size), dtype=np.float32)
        #         for i in xrange(len(X[1])):
        #             if i % 10 == 0: print i
        #             batch_test_probs = self.sess.run(self._probabilities,
        #                                              feed_dict={self.x_a: np.asarray([X[0][i]] * batch_size),
        #                                                         self.alens: np.asarray([X[1][i]] * batch_size),
        #                                                         self.q_ids: np.arange(batch_size),
        #                                                         self.x_q: topics[0],
        #                                                         self.qlens: topics[1],
        #                                                         self.maxlen: np.max(X[1]),
        #                                                         self.batch_size: batch_size})
        #             test_probs[i, :] = np.squeeze(batch_test_probs)
        #         np.savetxt(name + '_probabilities_topics.txt', test_probs)
        #         test_probs = np.reshape(test_probs, newshape=(batch_size * len(X[0])))
        #         hist = np.histogram(test_probs, bins=100, range=[0.0, 1.0], density=True)
        #
        #         plt.plot(hist[0])
        #         plt.xticks(np.arange(0, 101, 20), [str(i / 100.0) for i in xrange(0, 101, 20)])
        #         plt.ylim(0, 50)
        #         plt.ylabel('Density')
        #         plt.xlabel('Relevance Probability')
        #         plt.title('Empirical PDF of Relevance Probabilities')
        #         # plt.show()
        #         plt.savefig('histogram_LINSKneg02.png')
        #         plt.close()


class AttentionTopicModelStudent(AttentionTopicModel):
    def __init__(self, network_architecture=None, name=None, save_path='./', load_path=None, debug_mode=0, seed=100,
                 epoch=None, num_teachers=None):
        AttentionTopicModel.__init__(network_architecture=None, name=None, save_path='./', load_path=None, debug_mode=0,
                                     seed=100, epoch=None)
        self.num_teachers = num_teachers

    def _parse_func(self, example_proto):
        contexts, features = tf.parse_single_sequence_example(
            serialized=example_proto,
            context_features={"targets": tf.FixedLenFeature([], tf.float32),
                              "grade": tf.FixedLenFeature([], tf.float32),
                              "spkr": tf.FixedLenFeature([], tf.string),
                              "q_id": tf.FixedLenFeature([], tf.int64),
                              "teacher_preds": tf.FixedLenFeature([self.num_teachers], tf.float32)},
            sequence_features={'response': tf.FixedLenSequenceFeature([], dtype=tf.int64),
                               'prompt': tf.FixedLenSequenceFeature([], dtype=tf.int64)})

        return contexts['targets'], contexts['teacher_preds'], contexts['q_id'], features['response'], features['prompt']

    def _map_func(self, dataset, num_threads, capacity, augment=None):
        dataset = dataset.map(lambda targets, teacher_preds, q_id, resp, prompt: (targets,
                                                                                  teacher_preds,
                                                                                  tf.cast(q_id, dtype=tf.int32),
                                                                                  tf.cast(resp, dtype=tf.int32),
                                                                                  tf.cast(prompt, dtype=tf.int32)),
                              num_parallel_calls=num_threads).prefetch(capacity)

        return dataset.map(lambda targets, teacher_preds, q_id, resp, prompt: (
        targets, teacher_preds, q_id, resp, tf.size(resp), prompt, tf.size(prompt)),
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
                    tf.TensorShape([]),  # targets -- unused
                    tf.TensorShape([]),  # predictions -- unused
                    tf.TensorShape([]),  # q_id -- unused
                    tf.TensorShape([None]),  # resp
                    tf.TensorShape([]),  # resp len -- unused
                    tf.TensorShape([None]),  # prompt
                    tf.TensorShape([])),  # prompt len -- unused
                padding_values=(
                    0.0,  # targets -- unused
                    0.0,  # ensemble predictions - unused
                    np.int32(0),  # q_id -- unused
                    np.int32(0),  # resp
                    np.int32(0),  # resp len -- unused
                    np.int32(0),  # resp len -- unused
                    np.int32(
                        0)))

        def key_func(unused_1, unused_2, unused_3, unused_4, resp_len, unused_5, unused_6):
            # Calculate bucket_width by maximum source sequence length.
            # Pairs with length [0, bucket_width) go to bucket 0, length
            # [bucket_width, 2 * bucket_width) go to bucket 1, etc.  Pairs with length
            # over ((num_bucket-1) * bucket_width) words all go into the last bucket.

            # Bucket sentence pairs by the length of their response
            bucket_id = resp_len // bucket_width
            return tf.to_int64(tf.minimum(num_buckets, bucket_id))

        def reduce_func(unused_key, windowed_data):
            return batching_func(windowed_data)

        batched_dataset = dataset.apply(tf.contrib.data.group_by_window(key_func=key_func,
                                                                        reduce_func=reduce_func,
                                                                        window_size=batch_size))

        return batched_dataset

    def fit(self,
            train_data,
            valid_data,
            load_path,
            topics,
            topic_lens,
            unigram_path,
            train_size=100,
            valid_size=100,
            learning_rate=1e-2,
            lr_decay=0.8,
            dropout=1.0,
            batch_size=50,
            distortion=1.0,
            optimizer=tf.train.AdamOptimizer,
            optimizer_params={},
            n_epochs=30,
            n_samples=1,  # Number of negative samples to generate per positive sample
            epoch=1):
        with self._graph.as_default():
            # Compute number of training examples and batch size
            n_examples = train_size * (1 + n_samples)
            n_batches = n_examples / (batch_size * (1 + n_samples))

            # If some variables have been initialized - get them into a set
            temp = set(tf.global_variables())

            # Define Global step for training
            global_step = tf.Variable(0, trainable=False, name='global_step')

            # Set up inputs
            with tf.variable_scope(self._input_scope, reuse=True) as scope:
                # Construct training data queues
                targets, \
                teacher_predictions, \
                q_ids, \
                responses, \
                response_lengths, _, _ = self._construct_dataset_from_tfrecord([train_data],
                                                                               self._parse_func,
                                                                               self._map_func,
                                                                               self._batch_func,
                                                                               batch_size,
                                                                               train=True,
                                                                               capacity_mul=1000,
                                                                               num_threads=8)

                valid_iterator = self._construct_dataset_from_tfrecord([valid_data],
                                                                       self._parse_func,
                                                                       self._map_func,
                                                                       self._batch_func,
                                                                       batch_size,
                                                                       train=False,
                                                                       capacity_mul=100,
                                                                       num_threads=1)
                valid_targets, \
                valid_teacher_predictions, \
                valid_q_ids, \
                valid_responses, \
                valid_response_lengths, _, _ = valid_iterator.get_next(name='valid_data')

                # targets, q_ids = self._sampling_function(targets=targets,
                #                                          q_ids=q_ids,
                #                                          unigram_path=unigram_path,
                #                                          batch_size=batch_size,
                #                                          n_samples=n_samples,
                #                                          name='train',
                #                                          distortion=distortion)
                # valid_targets, valid_q_ids = self._sampling_function(targets=valid_targets,
                #                                                      q_ids=valid_q_ids,
                #                                                      unigram_path=unigram_path,
                #                                                      batch_size=batch_size,
                #                                                      n_samples=n_samples,
                #                                                      name='valid',
                #                                                      distortion=1.0)


            # Preprocess the teacher predictions to create targets

            # Calculate student targets as the mean teacher ensemble prediction
            trn_teacher_targets = tf.reduce_mean(teacher_predictions, axis=1)
            valid_teacher_targets = tf.reduce_mean(valid_teacher_predictions, axis=1)
            trn_teacher_targets = tf.Print(trn_teacher_targets, tf.shape(trn_teacher_targets), 'Shape of teacher targets')  # todo: remove once known


            topics = tf.convert_to_tensor(topics, dtype=tf.int32)
            topic_lens = tf.convert_to_tensor(topic_lens, dtype=tf.int32)

            prompts = tf.nn.embedding_lookup(topics, q_ids, name='train_prompt_loopkup')
            prompt_lens = tf.gather(topic_lens, q_ids)

            valid_prompts = tf.nn.embedding_lookup(topics, valid_q_ids, name='valid_prompt_loopkup')
            valid_prompt_lens = tf.gather(topic_lens, valid_q_ids)

            # Construct Training & Validation models
            with tf.variable_scope(self._model_scope, reuse=True) as scope:
                trn_predictions, \
                trn_probabilities, \
                trn_logits, _, = self._construct_network(a_input=responses,
                                                         a_seqlens=response_lengths,
                                                         n_samples=n_samples,
                                                         q_input=prompts,
                                                         q_seqlens=prompt_lens,
                                                         maxlen=tf.reduce_max(response_lengths),
                                                         batch_size=batch_size,
                                                         keep_prob=self.dropout)

                valid_predictions, \
                valid_probabilities, \
                valid_logits, \
                valid_attention = self._construct_network(a_input=valid_responses,
                                                          a_seqlens=valid_response_lengths,
                                                          n_samples=n_samples,
                                                          q_input=valid_prompts,
                                                          q_seqlens=valid_prompt_lens,
                                                          maxlen=tf.reduce_max(valid_response_lengths),
                                                          batch_size=batch_size,
                                                          keep_prob=1.0)

            # Construct XEntropy training costs
            trn_cost, total_loss = self._construct_xent_cost(targets=trn_teacher_targets,
                                                             logits=trn_logits,
                                                             pos_weight=float(n_samples),
                                                             is_training=True)
            evl_cost = self._construct_xent_cost(targets=valid_targets,
                                                 logits=valid_logits,
                                                 pos_weight=float(n_samples),
                                                 is_training=False)

            train_op = util.create_train_op(total_loss=total_loss,
                                            learning_rate=learning_rate,
                                            optimizer=optimizer,
                                            optimizer_params=optimizer_params,
                                            n_examples=n_examples,
                                            batch_size=batch_size,
                                            learning_rate_decay=lr_decay,
                                            global_step=global_step,
                                            clip_gradient_norm=10.0,
                                            summarize_gradients=False)

            # Intialize only newly created variables, as opposed to reused - allows for finetuning and transfer learning :)
            init = tf.variables_initializer(set(tf.global_variables()) - temp)
            self.sess.run(init)

            if load_path != None:
                self._load_variables(load_scope='model/Embeddings/word_embedding',
                                     new_scope='atm/Embeddings/word_embedding', load_path=load_path)

            # Update Log with training details
            with open(os.path.join(self._save_path, 'LOG.txt'), 'a') as f:
                format_str = (
                    'Learning Rate: %f\nLearning Rate Decay: %f\nBatch Size: %d\nValid Size: %d\nOptimizer: %s\nDropout: %f\nSEED: %i\n')
                f.write(format_str % (
                    learning_rate, lr_decay, batch_size, valid_size, str(optimizer), dropout, self._seed) + '\n\n')

            format_str = (
                'Epoch %d, Train Loss = %.2f, Valid Loss = %.2f, Valid ROC = %.2f, (%.1f examples/sec; %.3f ' 'sec/batch)')
            print("Starting Training!\n-----------------------------")
            start_time = time.time()
            for epoch in xrange(epoch + 1, epoch + n_epochs + 1):
                # Run Training Loop
                loss = 0.0
                batch_time = time.time()
                for batch in xrange(n_batches):
                    _, loss_value = self.sess.run([train_op, trn_cost], feed_dict={self.dropout: dropout})
                    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                    loss += loss_value

                duration = time.time() - batch_time
                loss /= n_batches
                examples_per_sec = batch_size / duration
                sec_per_epoch = float(duration)

                # Run Validation Loop
                eval_loss = 0.0
                valid_probs = None
                vld_targets = None
                total_size = 0
                self.sess.run(valid_iterator.initializer)
                while True:
                    try:
                        batch_eval_loss, \
                        batch_valid_preds, \
                        batch_valid_probs, \
                        batch_attention, \
                        batch_valid_targets = self.sess.run([evl_cost,
                                                             valid_predictions,
                                                             valid_probabilities,
                                                             valid_attention,
                                                             valid_targets])
                        size = batch_valid_probs.shape[0]
                        eval_loss += float(size) * batch_eval_loss
                        if valid_probs is None:
                            valid_probs = batch_valid_probs
                            vld_targets = batch_valid_targets
                        else:
                            valid_probs = np.concatenate((valid_probs, batch_valid_probs), axis=0)
                            vld_targets = np.concatenate((vld_targets, batch_valid_targets), axis=0)
                        total_size += size
                    except:  # tf.errors.OutOfRangeError:
                        break

                eval_loss = eval_loss / float(total_size)
                roc_score = roc(np.squeeze(vld_targets), np.squeeze(valid_probs))

                # Summarize Epoch
                with open(os.path.join(self._save_path, 'LOG.txt'), 'a') as f:
                    f.write(format_str % (epoch, loss, eval_loss, roc_score, examples_per_sec, sec_per_epoch) + '\n')
                print(format_str % (epoch, loss, eval_loss, roc_score, examples_per_sec, sec_per_epoch))
                self.save(step=epoch)

            # Finish Training
            duration = time.time() - start_time
            with open(os.path.join(self._save_path, 'LOG.txt'), 'a') as f:
                format_str = ('Training took %.3f sec')
                f.write('\n' + format_str % (duration) + '\n')
                f.write('----------------------------------------------------------\n')
            print(format_str % (duration))
            self.save()
