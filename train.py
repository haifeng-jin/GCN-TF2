import os
import pickle as pkl
import sys
import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras import layers
import numpy as np
import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh


tf.executing_eagerly()

hidden_dim = 16
dropout_rate = 0.5
weight_decay = 5e-4


def masked_softmax_cross_entropy(preds, labels, mask):
    """
    Softmax cross-entropy loss with masking.
    """
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask):
    """
    Accuracy with masking.
    """
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)


def dot(x, y, sparse=False):
    """
    Wrapper for tf.matmul (sparse vs dense).
    """
    if sparse:
        res = tf.sparse.sparse_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class GraphConvolution(layers.Layer):
    """
    Graph convolution layer.
    """
    def __init__(self, input_dim, output_dim, num_features_nonzero,
                 dropout=0.,
                 is_sparse_inputs=False,
                 activation=tf.nn.relu,
                 bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        self.dropout = dropout
        self.activation = activation
        self.is_sparse_inputs = is_sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.num_features_nonzero = num_features_nonzero

        self.weights_ = []
        for i in range(1):
            w = self.add_variable('weight' + str(i), [input_dim, output_dim])
            self.weights_.append(w)
        if self.bias:
            self.bias = self.add_variable('bias', [output_dim])

    def sparse_dropout(self, x):
        """
        Dropout for sparse tensors.
        """
        random_tensor = 1 - self.dropout
        random_tensor += tf.random.uniform(self.num_features_nonzero)
        dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
        pre_out = tf.sparse.retain(x, dropout_mask)
        return pre_out * (1./(1 - self.dropout))

    def call(self, inputs, training=None):
        x, support_ = inputs

        # dropout
        if training is not False and self.is_sparse_inputs:
            x = self.sparse_dropout(x)
        elif training is not False:
            x = tf.nn.dropout(x, self.dropout)


        # convolve
        supports = list()
        for i in range(len(support_)):
            if not self.featureless: # if it has features x
                pre_sup = dot(x, self.weights_[i], sparse=self.is_sparse_inputs)
            else:
                pre_sup = self.weights_[i]

            support = dot(support_[i], pre_sup, sparse=True)
            supports.append(support)

        output = tf.add_n(supports)

        # bias
        if self.bias:
            output += self.bias

        return self.activation(output)


class GCN(keras.Model):

    def __init__(self, input_dim, output_dim, num_features_nonzero, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.input_dim = input_dim # 1433
        self.output_dim = output_dim

        print('input dim:', input_dim)
        print('output dim:', output_dim)
        print('num_features_nonzero:', num_features_nonzero)

        self.layers_ = []
        self.layers_.append(GraphConvolution(input_dim=self.input_dim, # 1433
                                             output_dim=hidden_dim, # 16
                                             num_features_nonzero=num_features_nonzero,
                                             activation=tf.nn.relu,
                                             dropout=dropout_rate,
                                             is_sparse_inputs=True))

        self.layers_.append(GraphConvolution(input_dim=hidden_dim, # 16
                                             output_dim=self.output_dim, # 7
                                             num_features_nonzero=num_features_nonzero,
                                             activation=lambda x: x,
                                             dropout=dropout_rate))


        for p in self.trainable_variables:
            print(p.name, p.shape)

    def call(self, inputs, training=None):
        x, support = inputs

        outputs = [x]

        for layer in self.layers:
            hidden = layer((outputs[-1], support), training)
            outputs.append(hidden)
            output = outputs[-1]

        return output

        # # Weight decay loss
        # for var in self.layers_[0].trainable_variables:
        #  loss += weight_decay * tf.nn.l2_loss(var)

    def predict(self):
        return tf.nn.softmax(self.outputs)


def parse_index_file(filename):
    """
    Parse index file.
    """
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """
    Create mask.
    """
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data():
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
    (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
    object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    dataset_str = 'cora'
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def sparse_to_tuple(sparse_mx):
    """
    Convert sparse matrix to tuple representation.
    """
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """
    Row-normalize feature matrix and convert to tuple representation
    """
    rowsum = np.array(features.sum(1)) # get sum of each row, [2708, 1]
    r_inv = np.power(rowsum, -1).flatten() # 1/rowsum, [2708]
    r_inv[np.isinf(r_inv)] = 0. # zero inf data
    r_mat_inv = sp.diags(r_inv) # sparse diagonal matrix, [2708, 2708]
    features = r_mat_inv.dot(features) # D^-1:[2708, 2708]@X:[2708, 2708]
    return sparse_to_tuple(features) # [coordinates, data, shape], []


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)) # D
    d_inv_sqrt = np.power(rowsum, -0.5).flatten() # D^-0.5
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt) # D^-0.5
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo() # D^-0.5AD^0.5


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


# set random seed
seed = 123
np.random.seed(seed)
tf.random.set_seed(seed)



# load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data()
print('adj:', adj.shape)
print('features:', features.shape)
print('y:', y_train.shape, y_val.shape, y_test.shape)
print('mask:', train_mask.shape, val_mask.shape, test_mask.shape)



# D^-1@X
features = preprocess_features(features) # [49216, 2], [49216], [2708, 1433]
print('features coordinates::', features[0].shape)
print('features data::', features[1].shape)
print('features shape::', features[2])

# D^-0.5 A D^-0.5
support = [preprocess_adj(adj)]
num_supports = 1
model_func = GCN



# Create model
model = GCN(input_dim=features[2][1], output_dim=y_train.shape[1], num_features_nonzero=features[1].shape) # [1433]


train_label = tf.convert_to_tensor(y_train)
train_mask = tf.convert_to_tensor(train_mask)
val_label = tf.convert_to_tensor(y_val)
val_mask = tf.convert_to_tensor(val_mask)
test_label = tf.convert_to_tensor(y_test)
test_mask = tf.convert_to_tensor(test_mask)
features = tf.sparse.reorder(tf.SparseTensor(*features))
support = [tf.sparse.reorder(tf.cast(tf.SparseTensor(*support[0]), dtype=tf.float32))]
# num_features_nonzero = features.values.shape

optimizer = optimizers.Adam(lr=1e-2)
# model.compile(loss=gcn_loss, metrics=[gcn_metric], optimizer=optimizer)
# model.fit(
    # x=(features, train_mask, support),
    # y=y_train,
    # epochs=200,
# )


def gcn_loss(y_true, y_pred, mask):
    label = y_true
    output = y_pred
    return masked_softmax_cross_entropy(output, label, mask)


def gcn_metric(y_true, y_pred, mask):
    label = y_true
    output = y_pred
    return masked_accuracy(output, label, mask)


epochs = 200
for epoch in range(epochs):

    with tf.GradientTape() as tape:
        y_true = train_label 
        y_pred = model((features, support))
        loss = gcn_loss(y_true, y_pred, train_mask)
        acc = gcn_metric(y_true, y_pred, train_mask)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    y_true = val_label 
    y_pred = model((features, support), training=False)
    val_acc = gcn_metric(y_true, y_pred, val_mask)


    if epoch % 20 == 0:

        print(epoch, float(loss), float(acc), '\tval:', float(val_acc))


y_pred = model((features, support), training=False)
y_true = test_label
test_loss = gcn_loss(y_true, y_pred, test_mask)
test_acc = gcn_metric(y_true, y_pred, test_mask)


print('\ttest:', float(test_loss), float(test_acc))
