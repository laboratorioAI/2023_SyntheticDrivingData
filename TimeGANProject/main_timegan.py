'''
Author: Diego Tamayo
Based on Time-series Generative Adversarial Networks (TimeGAN) Codebase.
Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import numpy as np
from timegan import timegan
from data_loading import real_data_loading
from metrics.discriminative_metrics import discriminative_score_metrics
from metrics.predictive_metrics import predictive_score_metrics
from metrics.visualization_metrics import visualization

def main (args):
    """Main function for timeGAN experiments.
    Args:
    - data_name: name of dataset
    - seq_len: sequence length
    - Network parameters (should be optimized for different datasets)
      - module: gru, lstm, or lstmLN
      - hidden_dim: hidden dimensions
      - num_layer: number of layers
      - iteration: number of training iterations
      - batch_size: the number of samples in each batch
      - metric_iteration: number of iterations for metric computation
    Returns:
    - ori_data: original data
    - generated_data: generated synthetic data
    - metric_results: discriminative and predictive scores
    """

    # Data loading.
    initial_data_len, ori_data, min_vals, max_vals = real_data_loading(args.data_name, args.seq_len)
    print('\n-> Initial information:\n')
    print('- No. of records in original dataset: ', initial_data_len)

    # Set network parameters.
    parameters = dict()  
    parameters['module'] = args.module
    parameters['hidden_dim'] = args.hidden_dim
    parameters['num_layer'] = args.num_layer
    parameters['iterations'] = args.iteration
    parameters['batch_size'] = args.batch_size

    # Synthetic data generation.
    generated_data = timegan(ori_data, parameters)
    print('\n-> Synthetic Data Generation finished...')

    # Performance metrics.
    metric_results = dict()

    # Getting dimensions.
    _, _, dim_ori_data = np.asarray(ori_data).shape
    print('\n-> dim_ori_data: ', dim_ori_data)
    print('-> parameters[\'hidden_dim\']: ', parameters['hidden_dim'])

    number_of_experiment = 13
    print('\n-> Number of experiment: ', number_of_experiment)

    # Experiment configuration:
    print('\n-> Experiment configuration:')
    print('- module:     ', parameters['module'])
    print('- hidden_dim: ', parameters['hidden_dim'])
    print('- num_layer:  ', parameters['num_layer'])
    print('- iterations: ', parameters['iterations'])
    print('- batch_size: ', parameters['batch_size'])

    # Discriminative network parameters
    number_of_iterations_disc = 200
    hidden_dim_disc = int(dim_ori_data/2)
    batch_size_disc = 32
    module_disc = 'lstm'

    # Predictive network parameters
    number_of_iterations_pred = 8000
    hidden_dim_pred = dim_ori_data * 2
    batch_size_pred = 512
    module_pred = 'lstm'

    print('\n-> Discriminative network parameters:')
    print('- number_of_iterations_disc: ', number_of_iterations_disc)
    print('- hidden_dim_disc: ', hidden_dim_disc)
    print('- batch_size_disc: ', batch_size_disc)
    print('- module_disc: ', module_disc)

    print('\n-> Predictive network parameters:')
    print('- number_of_iterations_pred: ', number_of_iterations_pred)
    print('- hidden_dim_pred: ', hidden_dim_pred)
    print('- batch_size_pred: ', batch_size_pred)
    print('- module_pred: ', module_pred)

    # Calculate discriminative score.
    discriminative_score = list()
    counter_discriminative = 1

    for _ in range(args.metric_iteration):
        if counter_discriminative == 1:
            print_network_parameters_information = True
            print_network_information = True
        else :
            print_network_parameters_information = False
            print_network_information = False
        counter_discriminative += 1

        temp_disc = discriminative_score_metrics(ori_data, generated_data, number_of_iterations_disc, batch_size_disc, print_network_parameters_information, print_network_information, module_disc, hidden_dim_disc, number_of_experiment)
        discriminative_score.append(temp_disc)

    metric_results['discriminative'] = np.mean(discriminative_score)

    # Calculate predictive score.
    predictive_score = list()
    counter_predictive = 1

    for _ in range(args.metric_iteration):
        if counter_predictive == 1 :
            print_network_parameters_information = True
            print_network_information = True
        else :
            print_network_parameters_information = True
            print_network_information = True
        counter_predictive += 1

        temp_pred = predictive_score_metrics(ori_data, generated_data, number_of_iterations_pred, batch_size_pred, print_network_parameters_information, print_network_information, module_pred, hidden_dim_pred)
        predictive_score.append(temp_pred)

    metric_results['predictive'] = np.mean(predictive_score)

    # Visualization (PCA and tSNE).
    number_of_components = 2
    visualization(ori_data, generated_data, 'pca' , number_of_components)
    visualization(ori_data, generated_data, 'tsne', number_of_components)

    # Discriminative and predictive results.
    print('\n->> Metrics results:\n')
    print(metric_results, '\n')


    # Creation of synthetic data csv file.
    synthetic_data_dir = "synthetic_data/"
    array_sequences = []
    for counter in range(len(generated_data)):
        if counter == 0:
            sequence = np.array(generated_data[counter])
            array_sequences.append(sequence)
            array_squeezed = np.squeeze(array_sequences)
        else:
            next_sequence = generated_data[counter]
            next_sequence = np.squeeze(next_sequence)
            array_squeezed = np.concatenate((array_squeezed, next_sequence), axis=0)

    generated_data_csv = np.array(array_squeezed)

    # Data denormalization
    generated_data_csv_denormalized = generated_data_csv * (max_vals - min_vals) + min_vals
    print('\n-> Reconstructed synthetic data generated for csv: ', np.shape(generated_data_csv_denormalized))
    file_synthetic_norm_csv = synthetic_data_dir + 'data_synthetic_' + str(number_of_experiment) + '.csv'
    np.savetxt(file_synthetic_norm_csv, generated_data_csv_denormalized, delimiter=',', fmt='%f',
               header='steering_angle,speed,rpm,acceleration,throttle_position,engine_temperature,voltage,distance,latitude,longitude,heart_rate,current_weather,temperature,precipitation,accidents_onsite',
               comments='')
    print('-> CSV file created successfully...')


    return ori_data, generated_data, metric_results


if __name__ == '__main__':  
    
    # Inputs for the main function
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_name',
        choices=['sine', 'stock', 'energy', 'serie_temporal'],
        default='serie_temporal',
        type=str)
    parser.add_argument(
        '--seq_len',
        help='sequence length',
        default=24,
        type=int)
    parser.add_argument(
        '--module',
        choices=['gru','lstm','lstmLN'],
        default='gru',
        type=str)
    parser.add_argument(
        '--hidden_dim',
        help='hidden state dimensions (should be optimized)',
        default=24,
        type=int)
    parser.add_argument(
        '--num_layer',
        help='number of layers (should be optimized)',
        default=3,
        type=int)
    parser.add_argument(
        '--iteration',
        help='Training iterations (should be optimized)',
        default=50000,
        type=int)
    parser.add_argument(
        '--batch_size',
        help='the number of samples in mini-batch (should be optimized)',
        default=128,
        type=int)
    parser.add_argument(
        '--metric_iteration',
        help='iterations of the metric computation',
        default=10,
        type=int)
  
    args = parser.parse_args() 
  
# Calls main function  
ori_data, generated_data, metrics = main(args)

