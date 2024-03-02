import pickle

import numpy as np
import tensorflow as tf
import random

from tensorflow.python.ops.nn import dynamic_rnn
from tensorflow.contrib.rnn import GRUCell, LSTMCell, MultiRNNCell
from tensorflow.contrib.lookup.lookup_ops import HashTable, KeyValueTensorInitializer
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.python.ops import variable_scope
from user_model.utils import gen_batched_data, compute_acc
from user_model.utils import FLAGS, PAD_ID, UNK_ID, GO_ID, EOS_ID, _START_VOCAB

np.random.seed(15)
random.seed(15)

class EnvModel(object):
    def __init__(self,
            num_items,
            num_embed_units,
            num_units,
            num_layers,
            vocab=None,
            embed=None,
            learning_rate=5e-4,
            learning_rate_decay_factor=0.95,
            max_gradient_norm=5.0,
            use_lstm=True):

        self.epoch = tf.Variable(0, trainable=False, name='env/epoch')
        self.epoch_add_op = self.epoch.assign(self.epoch + 1)

        self.sessions_input = tf.placeholder(tf.int32, shape=(None, None))
        self.rec_lists = tf.placeholder(tf.int32, shape=(None, None, None))
        self.rec_mask = tf.placeholder(tf.float32, shape=(None, None, None))
        self.aims_idx = tf.placeholder(tf.int32, shape=(None, None))
        self.sessions_length = tf.placeholder(tf.int32, shape=(None))
        self.purchase = tf.placeholder(tf.int32, shape=(None, None))

        if embed is None:
            self.embed = tf.get_variable('env/embed', [num_items, num_embed_units], tf.float32, initializer=tf.truncated_normal_initializer(0,1))
        else:
            self.embed = tf.get_variable('env/embed', dtype=tf.float32, initializer=embed)

        batch_size, encoder_length, rec_length = tf.shape(self.sessions_input)[0], tf.shape(self.sessions_input)[1], tf.shape(self.rec_lists)[2]

        encoder_mask = tf.reshape(tf.cumsum(tf.one_hot(self.sessions_length - 2,
            encoder_length), reverse=True, axis=1), [-1, encoder_length])

        self.encoder_input = tf.nn.embedding_lookup(self.embed, self.sessions_input) #batch*len*unit
        self.aims = tf.one_hot(self.aims_idx, rec_length)
        if use_lstm:
            cell = MultiRNNCell([LSTMCell(num_units) for _ in range(num_layers)])
        else:
            cell = MultiRNNCell([GRUCell(num_units) for _ in range(num_layers)])

        # Training
        with tf.variable_scope("env"):
            # [batch_size, length, num_units]
            encoder_output, _ = dynamic_rnn(cell, self.encoder_input,
                    self.sessions_length, dtype=tf.float32, scope="encoder")

            # [batch_size, length, embed_units]
            preference = tf.layers.dense(encoder_output, num_embed_units, name="pref_output")
            # [batch_size, length, rec_length, embed_units]
            self.candidate = tf.reshape(
                tf.gather_nd(self.embed, tf.expand_dims(self.rec_lists, 3)),
                [batch_size, encoder_length, rec_length, num_embed_units])

            # [batch_size, length, rec_length]
            logits = tf.reduce_mean(tf.multiply(tf.expand_dims(preference,2), self.candidate), 3)
            mul_prob = tf.nn.softmax(logits) * self.rec_mask

            # [batch_size, length, rec_length]
            self.norm_prob = mul_prob / (tf.expand_dims(tf.reduce_sum(mul_prob, 2), 2) + 1e-20)
            # [batch_size, length, metric_num]
            _, self.argmax_index = tf.nn.top_k(self.norm_prob, k=FLAGS.metric+1)
            local_predict_loss = tf.reduce_sum(-self.aims * tf.log(self.norm_prob + 1e-20), 2) * encoder_mask
            self.predict_loss = tf.reduce_sum(local_predict_loss) / tf.reduce_sum(encoder_mask)

            # [batch_size, length, embed_units]
            aim_embed = tf.reduce_sum(tf.expand_dims(self.aims, 3) * self.candidate, 2)
            if FLAGS.use_simulated_data:
                self.purchase_prob, local_purchase_loss, self.purchase_loss = tf.zeros([batch_size,encoder_length,2], dtype=tf.float32), tf.zeros([batch_size,encoder_length], dtype=tf.float32), tf.constant(0., dtype=tf.float32)
            else:
                # [batch_size, length, 2]
                self.purchase_prob = tf.nn.softmax(tf.layers.dense(tf.multiply(
                    tf.layers.dense(tf.stop_gradient(encoder_output), num_units, name="purchase_layer"),
                    tf.layers.dense(tf.stop_gradient(aim_embed), num_units, name="purchase_aim")), 2, name="purchase_projection"))
                local_purchase_loss = tf.reduce_sum(-tf.one_hot(self.purchase, 2) * tf.log(self.purchase_prob + 1e-20), 2) * encoder_mask * tf.pow(tf.cast(self.purchase, tf.float32)+1, 5.3)
                self.purchase_loss = tf.reduce_sum(local_purchase_loss) / tf.reduce_sum(encoder_mask)
            self.decoder_loss = self.predict_loss + self.purchase_loss

            self.score = tf.placeholder(tf.float32, (None, None))
            self.score_loss = tf.reduce_sum(self.score * (local_predict_loss + local_purchase_loss)) / tf.reduce_sum(encoder_mask)

        # Inference
        with tf.variable_scope("env", reuse=True):
            # tf.get_variable_scope().reuse_variables()
            # [batch_size, 1, embed_units]
            inf_preference = tf.expand_dims(tf.layers.dense(encoder_output[:,-1,:], num_embed_units, name="pref_output"), 1)
            # [batch_size, 1, rec_length, embed_units]
            self.inf_candidate = tf.reshape(
                tf.gather_nd(self.embed, tf.expand_dims(self.rec_lists, 3)),
                [batch_size, 1, rec_length, num_embed_units])

            # [batch_size, 1, rec_length]
            inf_logits = tf.reduce_mean(tf.multiply(tf.expand_dims(inf_preference,2), self.inf_candidate), 3)
            inf_mul_prob = tf.nn.softmax(inf_logits) * self.rec_mask

            self.inf_norm_prob = inf_mul_prob / (tf.expand_dims(tf.reduce_sum(inf_mul_prob, 2), 2) + 1e-20)
            # [batch_size, 1, metric_num]
            _, self.inf_argmax_index = tf.nn.top_k(self.inf_norm_prob, k=FLAGS.metric)
            _, self.inf_all_argmax_index = tf.nn.top_k(self.inf_norm_prob, k=tf.shape(self.inf_norm_prob)[-1])

            def gumbel_max(inp, alpha, beta):
                # assert len(tf.shape(inp)) == 2
                g = tf.random_uniform(tf.shape(inp),0.0001,0.9999)
                g = -tf.log(-tf.log(g))
                inp_g = tf.nn.softmax((tf.nn.log_softmax(inp/1.0) + g * alpha) * beta)
                return inp_g
            # [batch_size, action_num]
            _, self.inf_random_index = tf.nn.top_k(gumbel_max(tf.log(self.inf_norm_prob+1e-20), 1, 1), k=FLAGS.metric)
            _, self.inf_all_random_index = tf.nn.top_k(gumbel_max(tf.log(self.inf_norm_prob+1e-20), 1, 1), k=tf.shape(self.inf_norm_prob)[-1])

            inf_aim_embed = tf.reduce_sum(tf.cast(tf.reshape(tf.one_hot(self.inf_argmax_index[:,:,0], rec_length), [batch_size,1,rec_length,1]), tf.float32) * self.inf_candidate, 2)

            if FLAGS.use_simulated_data:
                self.inf_purchase_prob = tf.zeros([batch_size,1,2], dtype=tf.float32)
            else:
                # [batch_size, 1, 2]
                self.inf_purchase_prob = tf.nn.softmax(tf.layers.dense(tf.multiply(
                    tf.layers.dense(tf.stop_gradient(encoder_output), num_units, name="purchase_layer"),
                    tf.layers.dense(tf.stop_gradient(inf_aim_embed), num_units, name="purchase_aim")), 2, name="purchase_projection"))

        self.learning_rate = tf.Variable(float(learning_rate), trainable=False,
                dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(
                self.learning_rate * learning_rate_decay_factor)

        self.global_step = tf.Variable(0, trainable=False)
        opt = tf.train.AdamOptimizer(self.learning_rate)
        self.params = tf.trainable_variables()

        # For pretraining
        gradients = tf.gradients(self.decoder_loss, self.params)
        clipped_gradients, self.gradient_norm = tf.clip_by_global_norm(gradients,
                max_gradient_norm)
        self.update = opt.apply_gradients(zip(clipped_gradients, self.params),
                global_step=self.global_step)

        # For adversarial training
        score_gradients = tf.gradients(self.score_loss, self.params)
        score_clipped_gradients, self.score_gradient_norm = tf.clip_by_global_norm(score_gradients,
                max_gradient_norm)
        self.score_update = opt.apply_gradients(zip(score_clipped_gradients, self.params),
                global_step=self.global_step)

        if tf.train.get_checkpoint_state(FLAGS.env_train_dir):
            self.saver = tf.train.import_meta_graph(tf.train.latest_checkpoint(FLAGS.env_train_dir)+".meta")
        else:
            self.saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2,
                max_to_keep=100, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)

    def print_parameters(self):
        for item in self.params:
            print('%s: %s' % (item.name, item.get_shape()))

    def step_decoder(self, session, data, forward_only=False):
        input_feed = {self.sessions_input: data['sessions'],
                self.rec_lists: data['rec_lists'],
                self.aims_idx: data['aims'],
                self.rec_mask: data['rec_mask'],
                self.purchase: data['purchase'],
                self.sessions_length: data['sessions_length']}
        if forward_only:
            output_feed = [self.decoder_loss, self.argmax_index, self.predict_loss, self.purchase_loss, self.purchase_prob]
        else:
            output_feed = [self.decoder_loss, self.argmax_index, self.predict_loss, self.purchase_loss, self.purchase_prob, self.gradient_norm, self.update]
        return session.run(output_feed, input_feed)


    def pg_step_decoder(self, session, data, forward_only=False):
        input_feed = {self.sessions_input:data['sessions'],
                self.rec_lists:data['rec_lists'],
                self.aims_idx: data['aims'],
                self.rec_mask: data['rec_mask'],
                self.score: data['dis_reward'],
                self.purchase: data['purchase'],
                self.sessions_length: data['sessions_length']}
        if forward_only:
            output_feed = [self.score_loss, self.norm_prob]
        else:
            output_feed = [self.score_loss, self.norm_prob, self.score_gradient_norm, self.score_update]

        return session.run(output_feed, input_feed)


    def train(self, sess, dataset, is_train=True, ftest_name=FLAGS.env_output_file):
        st, ed, loss, acc, acc_1, pr_loss, pu_loss = 0, 0, [], [], [], [], []
        tp, tn, fp, fn = [], [], [], []
        if is_train:
            print("Get %s data:len(dataset) is %d " % ("training" if is_train else "testing", len(dataset)))
        if not is_train:
            fout = open(ftest_name, "w")
            fout.close()
        while ed < len(dataset):
            st, ed = ed, ed + FLAGS.batch_size if ed + \
                FLAGS.batch_size < len(dataset) else len(dataset)
            batch_data = gen_batched_data(dataset[st:ed])
            outputs = self.step_decoder(sess, batch_data, forward_only=False if is_train else True)
            loss.append(outputs[0])
            predict_index = outputs[1]  # [batch_size, length, 10]
            pr_loss.append(outputs[2])
            pu_loss.append(outputs[3])
            purchase_prob = outputs[4][:,:,1]
            prob_list = purchase_prob.tolist()[0]
            while 0.5 in prob_list:
                prob_list.remove(0.5)

            if not prob_list:
                prob_mean = 0
            else:
                # prob_mean = max(prob_list)
                prob_mean = random.choice(prob_list)
            return prob_mean


    def pg_train(self, sess, dataset):
        st, ed, loss = 0, 0, []
        print("Get %s data:len(dataset) is %d " % ("training", len(dataset)))
        while ed < len(dataset):
            st, ed = ed, ed + FLAGS.batch_size if ed + \
                FLAGS.batch_size < len(dataset) else len(dataset)
            batch_data = gen_batched_data(dataset[st:ed])
            outputs = self.pg_step_decoder(sess, batch_data, forward_only=False)
            loss.append(outputs[0])
        sess.run(self.epoch_add_op)
        return np.mean(loss)
