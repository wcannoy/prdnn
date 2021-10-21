import os
import sys
import random
import math
import numpy as np
import pandas as pd
from six import b
import yaml
import pickle

import sklearn as sk
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score

import tensorflow as tf
from util import get_mask_for_variable
from prune import isnip, igrasp, synflow, proposed_method


random.seed(2021)
np.random.seed(2021)
tf.random.set_seed(2021)


class DNNModel():

    def __init__(self, num_cols, cat_nuniq, emb_size = 32, hidden_units = [300, 300, 300]):
        super(DNNModel, self).__init__()
        self.trainable_variables = []
        
        self.num_cols = num_cols[:]
        self.cat_nuniq = cat_nuniq
        
        self.embedding_size = emb_size

        self.hidden_units = hidden_units[:]
        self.weights = {}
        self.mask = {}
        self.bias = {}
        
        W_init = tf.keras.initializers.GlorotUniform()
        bias_init = tf.zeros_initializer()
        
        prev_unit = len(self.cat_nuniq) * self.embedding_size + len(self.num_cols) * self.embedding_size
        # print(f'prev_unit:{prev_unit}')
        
        for i, unit in enumerate(self.hidden_units + [1]):
            self.weights[i] = tf.Variable(name = f'weight_{i}', 
                                          initial_value = W_init(shape = [prev_unit, unit]), trainable = True)
    
            self.mask[i] = get_mask_for_variable(self.weights[i])

            self.trainable_variables.append(self.weights[i])
            
            self.bias[i] = tf.Variable(name = f'bias_{i}', 
                                       initial_value = bias_init(shape = [unit]), trainable = True)
            self.trainable_variables.append(self.bias[i])
            prev_unit = unit
        
        emb_init = tf.keras.initializers.TruncatedNormal(mean = 0.0, stddev = 1/math.sqrt(self.embedding_size))
        self.cat_embeddings = {}
        for cat_col, cat_num in self.cat_nuniq.items():
            self.cat_embeddings[cat_col] = tf.Variable(name = f'emb_table_{cat_col}', 
                                    initial_value = emb_init(shape = [cat_num + 1, self.embedding_size]),
                                    trainable = True)
            self.trainable_variables.append(self.cat_embeddings[cat_col])
    
        self.num_trans_weights = {}
        for col in self.num_cols:
            self.num_trans_weights[col] = tf.Variable(name = f'trans_weight_{col}', 
                                          initial_value = W_init(shape = [1, self.embedding_size]), trainable = True)

        self.loss = tf.keras.losses.BinaryCrossentropy()
        self._reset()

    def get_pruneable_variables_with_masks(self):
        return list(self.weights.values()), list(self.mask.values())

    def get_embeddings(self):
        return list(self.cat_embeddings.values()) + list(self.num_trans_weights.values())

    def _reset(self):
        self.cat_id_weights = []
        self.cat_id_wrappers = []
        self.num_values = []
        
        
    def _make_input(self, batch):
        self._reset()
        y_true = batch['label']
        for cat_col in self.cat_nuniq:
            cat_ids = batch[cat_col]
            cat_id_weight = tf.nn.embedding_lookup(self.cat_embeddings[cat_col], cat_ids) 
            self.cat_id_weights.append(cat_id_weight)
        
        for num_col in self.num_cols:
            num_value = tf.cast(tf.expand_dims(batch[num_col], -1), tf.float32)
            
            # project numeric field to the same dimension with categorical embeddings
            num_value = tf.matmul(num_value, self.num_trans_weights[num_col])
            self.num_values.append(num_value)
        
        inps = tf.concat(self.cat_id_weights + self.num_values, axis=1)
        # print(inps.get_shape())
        
        return inps, y_true

    def dump_nn_input(self, outfile, dataset, num_batch=1):
        inps = []
        for (i, batch) in enumerate(dataset):
            if i >= num_batch:
                break
            inp, _ = self._make_input(batch)
            inps.append(inp.numpy())
        inps = np.stack(inps, axis=1)
        with open(outfile, "wb") as f:
            pickle.dump(inps, f)

    def dump_nn_input_connection(self, outfile):
        mask = self.mask[0].numpy()
        connect = mask.sum(axis=1)
        with open(outfile, "wb") as f:
            pickle.dump(connect, f)

    def _forward(self, inp):
        num_hidden = len(self.hidden_units)
        for i in range(num_hidden):
            inp = tf.nn.bias_add(tf.matmul(inp, tf.math.multiply(self.weights[i], self.mask[i])), self.bias[i])
            inp = tf.nn.relu(inp)
        out = tf.nn.bias_add(tf.matmul(inp, self.weights[num_hidden]), self.bias[num_hidden])
        out = tf.sigmoid(out)
        out = tf.reshape(out, shape=[-1])
        return out

    
    def do_train(self, batch):
        inp, y_true = self._make_input(batch)
        out = self._forward(inp)
        loss = self.loss(y_true, out)
        return loss
    
    
    def do_predict(self, batch):
        inp, y_true = self._make_input(batch)
        out = self._forward(inp)
        return out, y_true

    def evaluate_logit(self, batch):
        inp, y_true = self._make_input(batch)
        num_hidden = len(self.hidden_units)
        for i in range(num_hidden):
            inp = tf.nn.bias_add(tf.matmul(inp, tf.math.multiply(self.weights[i], self.mask[i])), self.bias[i])
            inp = tf.nn.relu(inp)
        out = tf.nn.bias_add(tf.matmul(inp, self.weights[num_hidden]), self.bias[num_hidden])
        out = tf.reshape(out, shape=[-1])
        return out

def train(model, num_epoch = 1):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    for epoch in range(num_epoch):
        total_loss = np.array([])
        for (_, batch) in enumerate(dataset_train):
            with tf.GradientTape() as tape:
                loss = model.do_train(batch)
                total_loss = np.append(total_loss, loss)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        print(f'[TRAIN] Epoch: {epoch}, loss: {np.mean(total_loss):.6f}')
    return model


def predict(model, dataset):
    y_pred = []
    y_true = []
    for (_, batch) in enumerate(dataset):
        out, true = model.do_predict(batch)
        out = tf.reshape(out, shape=[-1])
        
        y_pred += list(out.numpy())
        y_true += list(true.numpy())
        
    return y_pred, y_true



if __name__ == '__main__':
    
    pruning_method = sys.argv[1]
    pruning_ratio = float(sys.argv[2])
    
    assert pruning_method in  [
        "dense",
        "isnip","igrasp","synflow",
        "fsnip","fgrasp","fsynflow"
    ]
    assert 0 < pruning_ratio and pruning_ratio <= 1.0
    
    dataset_dir = '../data'
    
    num_cols = ['num_{}'.format(i) for i in range(1, 14)]
    cat_cols = ['cat_{}'.format(i) for i in range(1, 27)]
    header = ['label'] + num_cols + cat_cols
    df_train = pd.read_csv(os.path.join(dataset_dir, 'train.tsv'), sep = '\t', names = header).head(10000)
    df_valid = pd.read_csv(os.path.join(dataset_dir, 'valid.tsv'), sep = '\t', names = header).head(10000)
    
    print(f'#train:{df_train.shape}, #valid:{df_valid.shape}')
    
    le_dic = {}
    for cat_col in cat_cols:
        le_dic[cat_col] = preprocessing.LabelEncoder()
        df_train[cat_col] = le_dic[cat_col].fit_transform(df_train[cat_col]) + 1

        dic = dict(zip(le_dic[cat_col].classes_, le_dic[cat_col].transform(le_dic[cat_col].classes_)))
        df_valid[cat_col] = df_valid[cat_col].apply(lambda x: dic[x] + 1 if x in dic else 0)
    
    cat_nuniq = dict(df_train[cat_cols].nunique())
    ds_train = tf.data.Dataset.from_tensor_slices(dict(df_train))
    shuffled = ds_train.shuffle(100_000, seed=2021, reshuffle_each_iteration=False)
    dataset_train = shuffled.take(100_000).batch(256)

    ds_valid = tf.data.Dataset.from_tensor_slices(dict(df_valid))
    dataset_valid = ds_valid.take(100_000).batch(256)

    model = DNNModel(num_cols, cat_nuniq)

    # prune the model
    if pruning_method == "dense":
        pass
    elif pruning_method in ["fsnip","fgrasp","fsynflow"]:
        with open("../data/feat_map.yaml", "r") as f:
            fea_maps = yaml.load(f, yaml.SafeLoader)
        proposed_method("i" + pruning_method[1:], fea_maps, model, dataset_train, 1, pruning_ratio)
        model.dump_nn_input_connection("../plots/cache/{}_{}.pkl".format(pruning_method, pruning_ratio))
    else:
        pruning_method_maps = {
            "isnip" : isnip,
            "igrasp" : igrasp,
            'synflow' : synflow
        }
        pruning_method_maps[pruning_method](
            model, dataset_train, 1, pruning_ratio
        )
        if pruning_method == "isnip" and pruning_ratio == 0.01:
            model.dump_nn_input("../plots/cache/nn_input_{}_{}.pkl".format(pruning_method, pruning_ratio), dataset_train, 1)
        model.dump_nn_input_connection("../plots/cache/{}_{}.pkl".format(pruning_method, pruning_ratio))

    num_epoch = 2
    model = train(model, num_epoch)
    y_pred, y_true = predict(model, dataset_valid)
    auc = roc_auc_score(y_true, y_pred)
    print('[METHOD]: {} [RATIO] {} [VALID] AUC: {:.4f}'.format(pruning_method, pruning_ratio, auc))



