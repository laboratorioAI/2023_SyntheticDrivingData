"""
Author: Diego Tamayo
Based on Time-series Generative Adversarial Networks (TimeGAN) Codebase.
Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.
"""

import numpy as np
import tensorflow as tf


# train_test_divide: Divide train and test data for both original and synthetic data.
def train_test_divide (data_x, data_x_hat, data_t, data_t_hat, train_rate = 0.7):
    """Divide train and test data for both original and synthetic data.
    Args:
    - data_x: original data
    - data_x_hat: generated data
    - data_t: original time
    - data_t_hat: generated time
    - train_rate: ratio of training data from the original data
    """
    # Divide train/test index (original data)
    no = len(data_x)
    idx = np.random.permutation(no)
    train_idx = idx[:int(no*train_rate)]
    test_idx  = idx[int(no*train_rate):]
    
    train_x = [data_x[i] for i in train_idx]
    test_x  = [data_x[i] for i in test_idx]
    train_t = [data_t[i] for i in train_idx]
    test_t  = [data_t[i] for i in test_idx]
    
    # Divide train/test index (synthetic data)
    no = len(data_x_hat)
    idx = np.random.permutation(no)
    train_idx = idx[:int(no*train_rate)]
    test_idx = idx[int(no*train_rate):]
    
    train_x_hat = [data_x_hat[i] for i in train_idx]
    test_x_hat  = [data_x_hat[i] for i in test_idx]
    train_t_hat = [data_t_hat[i] for i in train_idx]
    test_t_hat  = [data_t_hat[i] for i in test_idx]
    
    return train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat


# extract_time: Returns Maximum sequence length and each sequence length.
def extract_time (data):
    """Returns Maximum sequence length and each sequence length.
    Args:
    - data: original data
    Returns:
    - time: extracted time information
    - max_seq_len: maximum sequence length
    """
    time = list()
    max_seq_len = 0
    for i in range(len(data)):
        max_seq_len = max(max_seq_len, len(data[i][:,0]))
        time.append(len(data[i][:,0]))
    return time, max_seq_len


# rnn_cell: Basic RNN Cell.
def rnn_cell(module_name, hidden_dim):
    """Basic RNN Cell.
    Args:
    - module_name: gru, lstm, or lstmLN
    Returns:
    - rnn_cell: RNN Cell
    """
    assert module_name in ['gru','lstm','lstmLN']
    # GRU
    if (module_name == 'gru'):
        rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_dim, activation=tf.nn.tanh)
        rnn_cell = tf.contrib.rnn.DropoutWrapper(rnn_cell, output_keep_prob=0.80)
    # LSTM
    elif (module_name == 'lstm'):
        #rnn_cell_0 = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, activation=tf.nn.tanh)
        rnn_cell_0 = tf.contrib.rnn.LSTMCell(num_units=hidden_dim, activation=tf.nn.tanh)
        rnn_cell = tf.contrib.rnn.DropoutWrapper(rnn_cell_0, output_keep_prob=0.80)
    # LSTM Layer Normalization
    elif (module_name == 'lstmLN'):
        rnn_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units=hidden_dim, activation=tf.nn.tanh)
    return rnn_cell

# random_generator: random vector generator
def random_generator (batch_size, z_dim, T_mb, max_seq_len):
    """Random vector generation.
    Args:
    - batch_size: size of the random vector
    - z_dim: dimension of random vector
    - T_mb: time information for the random vector
    - max_seq_len: maximum sequence length
    Returns:
    - Z_mb: generated random vector
    """
    Z_mb = list()
    for i in range(batch_size):
        temp = np.zeros([max_seq_len, z_dim])
        temp_Z = np.random.uniform(0., 1, [T_mb[i], z_dim])
        temp[:T_mb[i],:] = temp_Z
        Z_mb.append(temp_Z)
    return Z_mb

# batch_generator: mini-batch generator
def batch_generator(data, time, batch_size): # data: 1177 secuencias de 24 filas, time: 1777 valores de 24.
    """Mini-batch generator.
    Args:
    - data: time-series data
    - time: time information
    - batch_size: the number of samples in each batch
    Returns:
    - X_mb: time-series data in each batch
    - T_mb: time information in each batch
    """
    no = len(data)
    #print('\n* utils.py 128')
    #print('data size: ', data.size)
    #print('length time: ', len(time))
    idx = np.random.permutation(no) # devuelve desde 0
    #print('\n* utils.py 143')
    #print('idx: ', idx)
    train_idx = idx[:batch_size] # selecciona los primeros 32 (batch_size) valores del vector de 1177 valores aleatorios.
    #print('\n* utils.py 146')
    #print('train_idx: ', train_idx)
            
    X_mb = list(data[i] for i in train_idx) # 32 secuencias de 24 filas
    T_mb = list(time[i] for i in train_idx) # 32 valores iguales a 24
    
    return X_mb, T_mb