import os
import time

import matplotlib
import numpy as np

matplotlib.use('agg')
from matplotlib import pyplot as plt

from sklearn.metrics import roc_auc_score as roc

import tensorflow as tf
import tensorflow.contrib.slim as slim

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
                self.attention,  = self._construct_network(a_input=self.x_a,
                                                           a_seqlens=self.alens,
                                                           q_input=self.x_q,
                                                           q_seqlens=self.qlens,
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

    def _construct_network(self, a_input, a_seqlens, q_input, q_seqlens, maxlen,
                           batch_size, keep_prob=1.0):
        """ Construct RNNLM network
        Args:
          ?
        Returns:
          predictions, logits
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
                                     keep_prob=keep_prob, seed=self._seed+1)
            q_inputs = tf.nn.dropout(tf.nn.embedding_lookup(embedding, q_input, name='embedded_data'),
                                     keep_prob=keep_prob, seed=self._seed+2)

        with tf.variable_scope('RNN_Q', initializer=initializer(self._seed)) as scope:
            cell_fw = tf.contrib.rnn.BasicLSTMCell(num_units=self.network_architecture['n_phid'],
                                                   forget_bias=1.0,
                                                   activation=self.network_architecture['r_activation_fn'],
                                                   state_is_tuple=True)
            cell_bw = tf.contrib.rnn.BasicLSTMCell(num_units=self.network_architecture['n_phid'],
                                                   forget_bias=1.0,
                                                   activation=self.network_architecture['r_activation_fn'],
                                                   state_is_tuple=True)

            cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=keep_prob)
            cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=keep_prob)

            initial_state_fw = cell_fw.zero_state(batch_size=batch_size, dtype=tf.float32)
            initial_state_bw = cell_bw.zero_state(batch_size=batch_size, dtype=tf.float32)

            _, state = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                                       cell_bw=cell_bw,
                                                       inputs=q_inputs,
                                                       sequence_length=q_seqlens,
                                                       initial_state_fw=initial_state_fw,
                                                       initial_state_bw=initial_state_bw,
                                                       dtype=tf.float32,
                                                       parallel_iterations=32,
                                                       scope=scope)

            question_embeddings = tf.concat([state[0][1], state[1][1]], axis=1)
            question_embeddings = tf.nn.dropout(question_embeddings, keep_prob=keep_prob, seed=self._seed)

        # Response Encoder RNN
        with tf.variable_scope('RNN_A', initializer=initializer(self._seed)) as scope:
            cell_fw = tf.contrib.rnn.BasicLSTMCell(num_units=self.network_architecture['n_rhid'],
                                                   forget_bias=1.0,
                                                   activation=self.network_architecture['r_activation_fn'],
                                                   state_is_tuple=True)
            cell_bw = tf.contrib.rnn.BasicLSTMCell(num_units=self.network_architecture['n_rhid'],
                                                   forget_bias=1.0,
                                                   activation=self.network_architecture['r_activation_fn'],
                                                   state_is_tuple=True)

            initial_state_fw = cell_fw.zero_state(batch_size=batch_size, dtype=tf.float32)
            initial_state_bw = cell_bw.zero_state(batch_size=batch_size, dtype=tf.float32)

            cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=keep_prob)
            cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=keep_prob)

            outputs, state = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                                             cell_bw=cell_bw,
                                                             inputs=a_inputs,
                                                             sequence_length=a_seqlens,
                                                             initial_state_fw=initial_state_fw,
                                                             initial_state_bw=initial_state_bw,
                                                             dtype=tf.float32,
                                                             parallel_iterations=32,
                                                             scope=scope)

            outputs = tf.concat([outputs[0], outputs[1]], axis=2)

        hidden, attention = self._bahdanau_attention(memory=outputs, seq_lens=a_seqlens, maxlen=maxlen,
                                                     query=question_embeddings,
                                                     size=2 * self.network_architecture['n_rhid'],
                                                     batch_size=batch_size)

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

    def _map_func(self, dataset, num_threads, capacity):

        dataset =  dataset.map(lambda targets, q_id, resp, prompt: (targets,
                                                                    tf.cast(q_id, dtype=tf.int32),
                                                                    tf.cast(resp, dtype=tf.int32),
                                                                    tf.cast(prompt, dtype=tf.int32)),
                              num_threads=num_threads,
                              output_buffer_size=capacity)

        return dataset.map(lambda targets, q_id, resp, prompt: (targets, q_id, resp, tf.size(resp), prompt),
                              num_threads=num_threads,
                              output_buffer_size=capacity)

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
                    tf.TensorShape([None])),  # prompt
                padding_values=(
                    0.0, # targets -- unused
                    np.int32(0), # q_id -- unused
                    np.int32(0), # resp
                    np.int32(0), # resp len -- unused
                    np.int32(0)))#.filter(lambda targets, q_id, resp, size, prompt: tf.equal(tf.size(size), batch_size))  # prompt

        def key_func(unused_1, unused_2, unused_3, resp_len, unused_4):
            # Calculate bucket_width by maximum source sequence length.
            # Pairs with length [0, bucket_width) go to bucket 0, length
            # [bucket_width, 2 * bucket_width) go to bucket 1, etc.  Pairs with length
            # over ((num_bucket-1) * bucket_width) words all go into the last bucket.

            # Bucket sentence pairs by the length of their response
            bucket_id = resp_len // bucket_width
            return tf.to_int64(tf.minimum(num_buckets, bucket_id))

        def reduce_func(unused_key, windowed_data):
            return batching_func(windowed_data)

        batched_dataset = dataset.group_by_window(key_func=key_func,
                                                  reduce_func=reduce_func,
                                                  window_size=batch_size)

        return batched_dataset

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
            n_samples=1,
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
                response_lengths, _ = self._construct_dataset_from_tfrecord([train_data],
                                                                            self._parse_func,
                                                                            self._map_func,
                                                                            self._batch_func,
                                                                            batch_size,
                                                                            train=True,
                                                                            capacity_mul=1000,
                                                                            num_threads=8)
                responses = tf.tile(responses, [1 + n_samples, 1])
                response_lengths = tf.tile(response_lengths, [1 + n_samples])

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
                valid_response_lengths, _ = valid_iterator.get_next(name='valid_data')

                valid_responses = tf.tile(valid_responses, [1 + n_samples, 1])
                valid_response_lengths = tf.tile(valid_response_lengths, [1 + n_samples])


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

            prompts=tf.nn.embedding_lookup(topics, q_ids, name='train_prompot_loopkup')
            prompt_lens = tf.gather(topic_lens, q_ids)

            valid_prompts=tf.nn.embedding_lookup(topics, valid_q_ids, name='valid_prompot_loopkup')
            valid_prompt_lens=tf.gather(topic_lens, valid_q_ids)
            # Construct Training & Validation models
            with tf.variable_scope(self._model_scope, reuse=True) as scope:
                trn_predictions, \
                trn_probabilities, \
                trn_logits, _,  = self._construct_network(a_input=responses,
                                                          a_seqlens=response_lengths,
                                                          q_input=prompts,
                                                          q_seqlens=prompt_lens,
                                                          maxlen=tf.reduce_max(response_lengths),
                                                          batch_size=batch_size * (1 + n_samples),
                                                          keep_prob=self.dropout)

                valid_predictions, \
                valid_probabilities, \
                valid_logits, \
                valid_attention  = self._construct_network(a_input=valid_responses,
                                                           a_seqlens=valid_response_lengths,
                                                           q_input=valid_prompts,
                                                           q_seqlens=valid_prompt_lens,
                                                           maxlen=tf.reduce_max(valid_response_lengths),
                                                           batch_size=batch_size * (1 + n_samples),
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
                self._load_variables(load_scope='model/Embeddings/word_embedding', new_scope='atm/Embeddings/word_embedding', load_path=load_path)

            # Update Log with training details
            with open(os.path.join(self._save_path, 'LOG.txt'), 'a') as f:
                format_str = (
                'Learning Rate: %f\nLearning Rate Decay: %f\nBatch Size: %d\nValid Size: %d\nOptimizer: %s\nDropout: %f\nSEED: %i\n')
                f.write(format_str % (learning_rate, lr_decay, batch_size, valid_size, str(optimizer), dropout, self._seed) + '\n\n')

            format_str = ('Epoch %d, Train Loss = %.2f, Valid Loss = %.2f, Valid ROC = %.2f, (%.1f examples/sec; %.3f ' 'sec/batch)')
            print "Starting Training!\n-----------------------------"
            start_time = time.time()
            for epoch in xrange(epoch + 1, epoch + n_epochs + 1):
                #Run Training Loop
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

                #Run Validation Loop
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
                        batch_attention,\
                        batch_valid_targets     = self.sess.run([evl_cost,
                                                           valid_predictions,
                                                           valid_probabilities,
                                                           valid_attention,
                                                           valid_targets])
                        size=batch_valid_probs.shape[0]
                        eval_loss += float(size) * batch_eval_loss
                        if valid_probs is None:
                            valid_probs=batch_valid_probs
                            vld_targets=batch_valid_targets
                        else:
                            valid_probs = np.concatenate((valid_probs, batch_valid_probs), axis=0)
                            vld_targets=np.concatenate((vld_targets, batch_valid_targets), axis=0)
                        total_size+=size
                    except:# tf.errors.OutOfRangeError:
                        break

                eval_loss = eval_loss / float(total_size)
                roc_score = roc(np.squeeze(vld_targets), np.squeeze(valid_probs))

                #Summarize Epoch
                with open(os.path.join(self._save_path, 'LOG.txt'), 'a') as f:
                    f.write(format_str % (epoch, loss, eval_loss, roc_score, examples_per_sec, sec_per_epoch) + '\n')
                print (format_str % (epoch, loss, eval_loss, roc_score, examples_per_sec, sec_per_epoch))
                self.save(step=epoch)

            #Finish Training
            duration = time.time() - start_time
            with open(os.path.join(self._save_path, 'LOG.txt'), 'a') as f:
                format_str = ('Training took %.3f sec')
                f.write('\n' + format_str % (duration) + '\n')
                f.write('----------------------------------------------------------\n')
            print (format_str % (duration))
            self.save()

    def predict(self, data):
        with self._graph.as_default():
            loss = self._construct_xent_cost(targets=self.y, logits=self._logits, pos_weight=1.0, is_training=False)
            test_size = len(data[0])
            test_loss = 0.0
            test_preds = np.zeros(shape=(test_size), dtype=np.int32)
            test_probs = np.zeros(shape=(test_size), dtype=np.float32)
            attention = np.zeros(shape=(test_size, np.max(data[2])), dtype=np.float32)
            batch_size = 424

            for i in xrange(test_size / batch_size +1 ):
                size = len(data[1][i * batch_size:(i + 1) * batch_size])
                batch_test_loss, batch_test_preds, batch_test_probs, batch_attention = self.sess.run(
                    [loss,
                     self._predictions,
                     self._probabilities,
                     self.attention],
                    feed_dict={self.y: np.squeeze(np.round(data[0][i * batch_size:(i + 1) * batch_size]))[:, np.newaxis],
                            self.x_a: data[1][i * batch_size:(i + 1) * batch_size],
                            self.alens: data[2][i * batch_size:(i + 1) * batch_size],
                            self.x_q: data[3][i * batch_size:(i + 1) * batch_size],
                            self.qlens: data[4][i * batch_size:(i + 1) * batch_size],
                            self.maxlen: np.max(data[2]),
                            self.batch_size: size,
                            self.dropout : 1.0})
                test_loss += float(size) * batch_test_loss
                test_preds[i * batch_size:(i+1)*batch_size] = np.squeeze(batch_test_preds)
                test_probs[i * batch_size:(i + 1) * batch_size] = np.squeeze(batch_test_probs)
                attention[i * batch_size:(i + 1) * batch_size,:] = batch_attention

            test_loss = test_loss / float(test_size)

        return test_probs, test_preds, attention, test_loss

    def rank(self, X, topics, name=None):
        with self._graph.as_default():
            test_probs = None
            batch_size = len(topics[1])
            test_probs = np.zeros(shape=(len(X[0]), batch_size), dtype=np.float32)
            for i in xrange(len(X[1])):
                if i % 10 == 0: print i
                batch_test_probs = self.sess.run(self._probabilities,
                                                 feed_dict={self.x_a: np.asarray([X[0][i]] * batch_size),
                                                            self.alens: np.asarray([X[1][i]] * batch_size),
                                                            self.q_ids: np.arange(batch_size),
                                                            self.x_q: topics[0],
                                                            self.qlens: topics[1],
                                                            self.maxlen: np.max(X[1]),
                                                            self.batch_size: batch_size})
                test_probs[i, :] = np.squeeze(batch_test_probs)
            np.savetxt(name + '_probabilities_topics.txt', test_probs)
            test_probs = np.reshape(test_probs, newshape=(batch_size * len(X[0])))
            hist = np.histogram(test_probs, bins=100, range=[0.0, 1.0], density=True)

            plt.plot(hist[0])
            plt.xticks(np.arange(0, 101, 20), [str(i / 100.0) for i in xrange(0, 101, 20)])
            plt.ylim(0, 50)
            plt.ylabel('Density')
            plt.xlabel('Relevance Probability')
            plt.title('Empirical PDF of Relevance Probabilities')
            # plt.show()
            plt.savefig('histogram_LINSKneg02.png')
            plt.close()
