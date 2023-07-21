"""
Author: Diego Tamayo
Based on Time-series Generative Adversarial Networks (TimeGAN) Codebase.
Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.
"""

import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_absolute_error
from utils import extract_time
import matplotlib.pyplot as plt

def predictive_score_metrics (ori_data, generated_data, number_of_iterations, size_of_batch, print_network_parameters_information, print_network_information, module, hidden_dim_pred):
    """Report the performance of Post-hoc RNN one-step ahead prediction.
    Args:
    - ori_data: original data
    - generated_data: generated synthetic data
    Returns:
    - predictive_score: MAE of the predictions on the original data
    """

    # Predictor function.
    def predictor(x, t):
        """Returns:
        - y_hat: prediction
        - p_vars: predictor variables
        """
        with tf.compat.v1.variable_scope("predictor", reuse=tf.compat.v1.AUTO_REUSE) as vs:
            if module == 'gru':
                cells = [tf.nn.rnn_cell.GRUCell(num_units=hidden_dim) for layer in range(2)]
            if module == 'lstm':
                cells = [tf.nn.rnn_cell.LSTMCell(num_units=hidden_dim) for layer in range(2)]

            cells_drop = [tf.compat.v1.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.80) for cell in cells]
            p_cell = tf.nn.rnn_cell.MultiRNNCell(cells_drop)
            p_outputs, p_last_states = tf.nn.dynamic_rnn(p_cell, x, dtype=tf.float32, sequence_length=t)
            y_hat_logit = tf.contrib.layers.fully_connected(p_outputs, 1, activation_fn=None)
            y_hat = tf.nn.sigmoid(y_hat_logit)
            p_vars = [v for v in tf.compat.v1.global_variables() if v.name.startswith(vs.name)]

        return y_hat, p_vars, p_outputs

    # Initialization of the Graph.
    tf.compat.v1.reset_default_graph()

    # Variables for printing.
    counter_print_shape = 1

    # Get dataset basic parameters.
    no, seq_len, dim = np.asarray(ori_data).shape

    # Sequences and length information.
    ori_time, ori_max_seq_len = extract_time(ori_data)
    generated_time, generated_max_seq_len = extract_time(generated_data)
    max_seq_len = max([ori_max_seq_len, generated_max_seq_len])

    # Print traces.
    if print_network_information:
        # Enabling traces printing.
        if counter_print_shape == 1:
            print_traces_status = True
        else:
            print_traces_status = False
        counter_print_shape += 1

        if print_traces_status:
            print('\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
            print('-> PREDICTIVE NETWORK')
            print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

            print('\n-> Original data time information:')
            print(' - ori_data shape:    ', np.shape(ori_data))
            print(' - ori_data min. val: ', np.min(ori_data))
            print(' - ori_data max. val: ', np.max(ori_data))

            print('\n-> Synthetic data time information:')
            print(' - generated_data:          ', np.shape(generated_data))
            print(' - generated_data min. val: ', np.min(generated_data))
            print(' - generated_data max. val: ', np.max(generated_data))

    # Network parameters.
    hidden_dim = hidden_dim_pred
    iterations = number_of_iterations
    batch_size = size_of_batch

    # Input place holders.
    X_gen = tf.compat.v1.placeholder(tf.float32, [None, max_seq_len-1, dim-1], name = "myinput_x_gen")
    T_gen = tf.compat.v1.placeholder(tf.int32, [None], name = "myinput_t_gen")
    Y_gen = tf.compat.v1.placeholder(tf.float32, [None, max_seq_len-1, 1], name = "myinput_y_gen")

    # Predictor method calls.
    y_pred_gen, p_vars, p_outputs_gen = predictor(X_gen, T_gen)

    # Loss for the predictor.
    p_loss = tf.compat.v1.losses.absolute_difference(Y_gen, y_pred_gen)

    # Optimizer.
    p_solver = tf.compat.v1.train.AdamOptimizer(learning_rate = 0.001,
                                      beta1 = 0.90,
                                      beta2 = 0.999,
                                      epsilon = 1e-08,
                                      use_locking = False,
                                      name = 'Adam').minimize(p_loss, var_list = p_vars)

    # Start session and initialization.
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    # Trainable parameters' information.
    if print_network_parameters_information:
        # Networks names.
        predictive_vars = [v for v in tf.compat.v1.trainable_variables() if v.name.startswith('predictor')]
        variable_names_pred = []
        for variable in predictive_vars:
            variable_names_pred.append(variable.name)

        print("\n****************************************************************************************")
        print('-> PREDICTIVE NETWORK COMPONENTS (PREDICTIVE SCORE):')
        print(f'-> Dataset shape for predictive network: [{no}, {seq_len}, {dim}]')

        # Each network parameters.
        total_parameters_pred = 0
        print("****************************************************************************************\n")

        for name in variable_names_pred:
            print('-> PREDICTIVE NETWORK:', name)
            for variable_pred in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, name):
                # Size of each network component.
                shape_pred = variable_pred.get_shape()
                print("   - Size of the matrix: {}".format(shape_pred))
                variable_parameters_pred = 1

                for dim_pred in shape_pred:
                    variable_parameters_pred *= dim_pred.value

                print("   - Total number of elements in a matrix: {}".format(variable_parameters_pred))
                print("--------------------------------------------------------------------------")
                total_parameters_pred += variable_parameters_pred
        print("-> TOTAL NUMBER OF PARAMETERS IN PREDICTIVE NETWORK: {}".format(total_parameters_pred))
        print("--------------------------------------------------------------------------\n")

    # Training.

    # Variables definition.
    print_traces_status = False
    counter_print_shape = 1


    for _ in range(iterations):
        # Enabling traces printing.
        if print_network_information :
            if counter_print_shape == 1 :
                print_traces_status = True
            else :
                print_traces_status = False
            counter_print_shape += 1

        # Vector train_idx_gen.
        idx_gen = np.random.permutation(len(generated_data))
        train_idx_gen = idx_gen[:batch_size]

        # Mini batches setting.
        X_mb_gen = list(generated_data[i][:-1,:(dim-1)] for i in train_idx_gen)
        T_mb_gen = list(generated_time[i]-1 for i in train_idx_gen)
        Y_mb_gen = list(np.reshape(generated_data[i][1:,(dim-1)],[len(generated_data[i][1:,(dim-1)]),1]) for i in train_idx_gen)

        # Train predictor.
        _, step_p_loss = sess.run([p_solver, p_loss], feed_dict={X_gen: X_mb_gen, T_gen: T_mb_gen, Y_gen: Y_mb_gen})

        # Print batch information.
        if print_traces_status:
            # Print vector train_idx_gen.
            print('\n--------------------------------')
            print('>> Training with generated data:')
            print('--------------------------------')
            print('* idx_gen = np.random.permutation(len(generated_data))')
            print('* train_idx_gen = idx_gen[:batch_size]')
            print('-> train_idx_gen: ', np.shape(train_idx_gen))

            # X_mb_gen, Y_mb_gen.
            print('\n----------------------')
            print('>> X_mb_gen, Y_mb_gen:')
            print('----------------------')
            print(' * X_mb_gen = list(generated_data[i][:-1,:(dim-1)] for i in train_idx)')
            print('-> X_mb_gen: ', np.shape(X_mb_gen))
            print('\n * Y_mb_gen = list(np.reshape(generated_data[i][1:,(dim-1)],[len(generated_data[i][1:,(dim-1)]),1]) for i in train_idx)')
            print('-> Y_mb_gen: ', np.shape(Y_mb_gen))

            # Outputs values.
            print('\n------------------------------')
            print('>> Training data (predictive):')
            print('------------------------------')
            outputs_val_gen = sess.run([p_outputs_gen], feed_dict={X_gen: X_mb_gen, T_gen: T_mb_gen})

            if module == 'gru':
                print('>> GRU real outputs:')
            if module == 'lstm':
                print('>> LSTM real outputs:')

            print('-> outputs_val_gen [batch_size, (num_steps,) neurons_number]:          ', np.shape(outputs_val_gen), '\n')

    # Test the trained model on the original data.

    # Vector train_idx_ori.
    idx_ori = np.random.permutation(len(ori_data))
    train_idx_ori = idx_ori[:no]

    # Mini batch setting.
    X_mb_ori = list(ori_data[i][:-1,:(dim-1)] for i in train_idx_ori)
    T_mb_ori = list(ori_time[i]-1 for i in train_idx_ori)
    Y_mb_ori = list(np.reshape(ori_data[i][1:,(dim-1)], [len(ori_data[i][1:,(dim-1)]),1]) for i in train_idx_ori)

    # Prediction.
    pred_Y_curr = sess.run(y_pred_gen, feed_dict={X_gen: X_mb_ori, T_gen: T_mb_ori})

    # Compute the performance in terms of MAE.
    MAE_temp = 0
    for i in range(no):
        error = mean_absolute_error(Y_mb_ori[i], pred_Y_curr[i,:,:])
        MAE_temp = MAE_temp + error

    predictive_score = MAE_temp / no

    return predictive_score