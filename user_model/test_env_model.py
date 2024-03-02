# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
import sys
import time
import random
import copy
from user_model.environment import EnvModel
from user_model.utils import FLAGS, load_data, build_vocab, gen_batched_data, PAD_ID, UNK_ID, GO_ID, EOS_ID, \
    _START_VOCAB
import os
import pickle

# **********************************************************************************


generate_session, gen_session, gen_rec_list, gen_aims_idx, gen_purchase, session_no, next_session = [], [], [], [], [], 0, True
ini_state = [[[[0.] * FLAGS.units]] * 2] * FLAGS.layers
gen_state = ini_state

# **********************************************************************************
# **********************************************************************************
# **********************************************************************************

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

env_graph = tf.Graph()
env_sess = tf.Session(config=config, graph=env_graph)

data = load_data(FLAGS.data_dir, FLAGS.data_name)
data = np.random.permutation(data)  

vocab, embed = build_vocab(data)
aid2index = {}
index2aid = {}
for i, a in enumerate(vocab):  
    aid2index[a] = i
    index2aid[i] = a


def filter(d):
    new_d = []
    for i, s in enumerate(d):
        tmps = []
        for c in s:  
            c["click"] = aid2index[c["click"]] if c["click"] in aid2index else UNK_ID
            c["rec_list"] = list(set([aid2index[rl] if rl in aid2index else UNK_ID for rl in c["rec_list"]])) + [EOS_ID]
            tmps.append(c)
        new_d.append(tmps)
    d = copy.deepcopy(new_d)
    return d


f = open('user_model/data_train.pkl', 'rb')
data_train = pickle.load(f)  

print("Get training data: number is %d, average length is %.4f" % (
    len(data_train), np.mean([len(s) for s in data_train])))

with env_graph.as_default():
    env_model = EnvModel(
        num_items=len(embed),
        num_embed_units=FLAGS.embed_units,
        num_units=FLAGS.units,
        num_layers=FLAGS.layers,
        vocab=vocab,
        embed=embed)
    env_model.print_parameters()
    if tf.train.get_checkpoint_state(FLAGS.env_train_dir):  
        print("Reading environment model parameters from %s" % FLAGS.env_train_dir)
        env_sess.run(tf.global_variables_initializer())
        env_model.saver.restore(env_sess, tf.train.latest_checkpoint(FLAGS.env_train_dir))
    else:
        print("Creating environment model with fresh parameters.")
        env_sess.run(tf.global_variables_initializer())

best_env_train_acc, best_env_train_acc_1 = 0., 0.


def get_config_score(users_session_list):
    user_test_data = filter(users_session_list)
    config_score = env_model.train(env_sess, user_test_data, is_train=False)
    return config_score
