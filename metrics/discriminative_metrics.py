"""
Author: Diego Tamayo
Based on Time-series Generative Adversarial Networks (TimeGAN) Codebase.
Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.
"""

import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
from utils import train_test_divide, extract_time, batch_generator
import matplotlib.pyplot as plt

def discriminative_score_metrics (ori_data, generated_data, number_of_iterations, size_of_batch, print_network_parameters_information, print_network_information, module, hidden_dim_disc, number_of_experiment):
    """Use post-hoc RNN to classify original data and synthetic data
    Args:
    - ori_data: original data
    - generated_data: generated synthetic data
    Returns:
    - discriminative_score: np.abs(classification accuracy - 0.5)
    """


    # Discriminator function.
    def discriminator(x, t):
        """Returns:
        - y_hat_logit: logits of the discriminator output
        - y_hat: discriminator output
        """
        with tf.compat.v1.variable_scope("discriminator", reuse=tf.compat.v1.AUTO_REUSE) as vs:
            if module == 'gru':
                cells = [tf.nn.rnn_cell.GRUCell(num_units=hidden_dim) for layer in range(1)]
                cells_drop = [tf.compat.v1.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.80) for cell in cells]
                d_cell = tf.nn.rnn_cell.MultiRNNCell(cells_drop)
                d_outputs, d_last_states = tf.nn.dynamic_rnn(d_cell, x, dtype=tf.float32, sequence_length=t)
                disc_outputs = d_outputs[-1]
            if module == 'lstm':
                cells = [tf.nn.rnn_cell.LSTMCell(num_units=hidden_dim) for layer in range(1)]
                cells_drop = [tf.compat.v1.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.80) for cell in cells]
                d_cell = tf.nn.rnn_cell.MultiRNNCell(cells_drop)
                d_outputs, d_last_states = tf.nn.dynamic_rnn(d_cell, x, dtype=tf.float32, sequence_length=t)
                d_outputs_transposed = tf.transpose(d_outputs, [1, 0, 2])
                disc_outputs = tf.gather(d_outputs_transposed, int(d_outputs_transposed.get_shape()[0]) - 1, name="last_lstm_output")

            y_logit = tf.contrib.layers.fully_connected(disc_outputs, 1, activation_fn=None)
            y_hat = tf.nn.sigmoid(y_logit)
            d_outputs_last = d_outputs[-1]
            d_vars = [v for v in tf.compat.v1.global_variables() if v.name.startswith(vs.name)]

        return y_logit, y_hat, d_vars, d_outputs_last, disc_outputs


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

    # Network information.
    if print_network_information:
        # region Enabling traces printing.
        if counter_print_shape == 1:
            print_traces_status = True
        else:
            print_traces_status = False
        counter_print_shape += 1
        # endregion

        if print_traces_status:
            print('\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
            print('-> DISCRIMINATIVE NETWORK')
            print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

            # Original data information.
            print('\n-> Original data time information:')
            print('- No. of sequences:                    ' + str(no))
            print('- No. of timesteps of each sequence:   ' + str(seq_len))
            print('- No. of features of each timestep:    ' + str(dim))

            print('\n-> Synthetic data time information:')
            print(' - generated_data:          ', np.shape(generated_data))
            print(' - generated_data min. val: ', np.min(generated_data))
            print(' - generated_data max. val: ', np.max(generated_data))

    # Network parameters.
    hidden_dim = hidden_dim_disc
    iterations = number_of_iterations
    batch_size = size_of_batch

    # Tensors definition.
    X_ori = tf.compat.v1.placeholder(tf.float32, [None, max_seq_len, dim], name = "myinput_x_ori")
    X_gen = tf.compat.v1.placeholder(tf.float32, [None, max_seq_len, dim], name = "myinput_x_gen")
    T_ori = tf.compat.v1.placeholder(tf.int32,   [None], name = "myinput_t_ori")
    T_gen = tf.compat.v1.placeholder(tf.int32,   [None], name = "myinput_t_gen")

    # Discriminator method calls.
    y_logit_x_ori, y_hat_x_ori, d_vars, d_outputs_last_x_ori, disc_outputs_x_ori = discriminator(X_ori, T_ori)
    y_logit_x_gen, y_hat_x_gen, _, _, _ = discriminator(X_gen, T_gen)

    # Trainable parameters' information.
    if print_network_parameters_information:
        # Networks names.
        discriminative_vars = [v for v in tf.compat.v1.trainable_variables() if v.name.startswith('discriminator')]
        variable_names_disc = []
        for variable_disc in discriminative_vars:
            variable_names_disc.append(variable_disc.name)

        print("\n****************************************************************************************")
        print('-> DISCRIMINATIVE NETWORK COMPONENTS (DISCRIMINATIVE SCORE):')
        print(f'-> Dataset shape for discriminative network: [{no}, {seq_len}, {dim}]')

        # Each network parameters.
        total_parameters_disc = 0
        print("****************************************************************************************\n")

        for name in variable_names_disc:
            print('-> DISCRIMINATIVE NETWORK:', name)
            for variable_disc in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, name):
                # Size of each network component.
                shape_disc = variable_disc.get_shape()
                print("   - Size of the matrix: {}".format(shape_disc))
                variable_parameters_disc = 1
                for dim_disc in shape_disc:
                    variable_parameters_disc *= dim_disc.value
                print("   - Total number of elements in a matrix: {}".format(variable_parameters_disc))
                print("--------------------------------------------------------------------------")
                total_parameters_disc += variable_parameters_disc
        print("-> TOTAL NUMBER OF PARAMETERS IN DISCRIMINATIVE NETWORK: {}".format(total_parameters_disc))
        print("--------------------------------------------------------------------------\n")

    # Loss for the discriminator.
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = y_logit_x_ori,
                                                                         labels = tf.ones_like(y_logit_x_ori)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = y_logit_x_gen,
                                                                         labels = tf.zeros_like(y_logit_x_gen)))
    d_loss = d_loss_real + d_loss_fake

    # Optimizer.
    d_solver = tf.compat.v1.train.AdamOptimizer(learning_rate = 0.001,
                                                beta1 = 0.90,
                                                beta2 = 0.999,
                                                epsilon = 1e-08,
                                                use_locking = False,
                                                name = 'Adam').minimize(d_loss, var_list = d_vars)

    # Start session and initialize.
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    # Train/test division for both original and generated data.
    x_train_ori, x_train_gen, x_test_ori, x_test_gen, t_train_ori, t_train_gen, t_test_ori, t_test_gen = \
    train_test_divide(ori_data, generated_data, ori_time, generated_time)

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

        # Mini batches setting.
        X_mb_ori, T_mb_ori = batch_generator(x_train_ori, t_train_ori, batch_size)
        X_mb_gen, T_mb_gen = batch_generator(x_train_gen, t_train_gen, batch_size)

        # Train discriminator.
        _, step_d_loss = sess.run([d_solver, d_loss], feed_dict={X_ori: X_mb_ori, T_ori: T_mb_ori, X_gen: X_mb_gen, T_gen: T_mb_gen})

        # Print discriminator outputs information.

        if print_traces_status:
            print('\n---------------------------------')
            print('>> Discriminative network shapes:')
            print('---------------------------------')

            print('>> Train/Test dataset division:')
            print('-> x_train_ori: ', np.shape(x_train_ori))
            print('-> t_train_ori: ', np.shape(t_train_ori))
            print('-> x_test_ori:  ', np.shape(x_test_ori))
            print('-> t_test_ori:  ', np.shape(t_test_ori))
            print('-> x_train_gen: ', np.shape(x_train_gen))
            print('-> t_train_gen: ', np.shape(t_train_gen))
            print('-> x_test_gen:  ', np.shape(x_test_gen))
            print('-> t_test_gen:  ', np.shape(t_test_gen))

            y_logit_x_mb_ori, y_hat_x_mb_ori, d_outputs_last_x_mb_ori, disc_outputs_x_mb_ori = sess.run(
                [y_logit_x_ori, y_hat_x_ori, d_outputs_last_x_ori, disc_outputs_x_ori,], feed_dict={X_ori: X_mb_ori, T_ori: T_mb_ori})

            print('\n-------------------------------------')
            print('>> Data description (discriminative):')
            print('-------------------------------------')

            if module == 'gru':
                print('\n>> GRU real outputs:')
            if module == 'lstm':
                print('\n>> LSTM real outputs:')

            print('\n * y_logit, y_hat, _, _, _, _ = discriminator(X, T)')
            print(' * X_mb_ori, T_mb_ori         = batch_generator(x_train_ori, t_train_ori, batch_size)')
            print('-> X_mb_ori:                   ', np.shape(X_mb_ori))
            print('\n-> d_outputs_last_x_mb_ori [(num_steps,) neurons_number]:      ', np.shape(d_outputs_last_x_mb_ori))
            print('-> disc_outputs_x_mb_ori   [batch_size, neurons_number]:       ', np.shape(disc_outputs_x_mb_ori))
            print('\n-> y_logit_x_mb_ori: ', np.shape(y_logit_x_mb_ori))
            print('-> y_hat_x_mb_ori:   ', np.shape(y_hat_x_mb_ori))
            print('\n-------------------------------------')
            print('>> Training process (discriminative):')
            print('-------------------------------------')
            print(' * X_mb_gen, T_mb_gen = batch_generator(x_train_gen, t_train_gen, batch_size)')
            print('-> X_mb_gen:           ', np.shape(X_mb_gen))

    # Test the performance on the testing set.

    # Calculate and print y_hat shape for testing.
    print('\n--------------------------------------')
    print('>> Discriminative network test shapes:')
    print('--------------------------------------')
    y_hat_x_test_ori, y_hat_x_test_gen = sess.run([y_hat_x_ori, y_hat_x_gen], feed_dict={X_ori: x_test_ori, T_ori: t_test_ori, X_gen: x_test_gen, T_gen: t_test_gen})
    print('-> x_test_ori:       ', np.shape(x_test_ori))
    print('-> y_hat_x_test_ori: ', np.shape(y_hat_x_test_ori))
    print('-> y_hat_x_test_gen: ', np.shape(y_hat_x_test_gen), '\n')

    y_pred_final  = np.squeeze(np.concatenate((y_hat_x_test_ori, y_hat_x_test_gen), axis = 0))
    y_label_final = np.concatenate((np.ones([len(y_hat_x_test_ori),]), np.zeros([len(y_hat_x_test_gen),])), axis = 0)

    # Compute accuracy.
    acc = accuracy_score(y_label_final, (y_pred_final > 0.5))
    discriminative_score = np.abs(0.5 - acc)

    return discriminative_score