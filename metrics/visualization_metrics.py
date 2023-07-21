"""
Author: Diego Tamayo
Based on Time-series Generative Adversarial Networks (TimeGAN) Codebase.
Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.
"""

import time

# Necessary packages
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_colwidth', None)
   
def visualization (ori_data, generated_data, analysis, number_of_components):
    """Using PCA or tSNE for generated and original data visualization.
    Args:
    - ori_data: original data
    - generated_data: generated synthetic data
    - analysis: tsne or pca
    """

    # Analysis sample size (for faster computation)
    anal_sample_no = min([5000, len(ori_data)])
    idx = np.random.permutation(len(ori_data))[:anal_sample_no]
    
    # Data preprocessing
    ori_data = np.asarray(ori_data)
    generated_data = np.asarray(generated_data)
    
    ori_data = ori_data[idx]
    generated_data = generated_data[idx]
    
    no, seq_len, dim = ori_data.shape

    for i in range(anal_sample_no):
        if (i == 0):
            prep_data = np.reshape(np.mean(ori_data[0,:,:], 1), [1,seq_len])
            prep_data_hat = np.reshape(np.mean(generated_data[0,:,:],1), [1,seq_len])
        else:
            prep_data = np.concatenate((prep_data, 
                                        np.reshape(np.mean(ori_data[i,:,:],1), [1,seq_len])))
            prep_data_hat = np.concatenate((prep_data_hat,
                                        np.reshape(np.mean(generated_data[i,:,:],1), [1,seq_len])))

    # Visualization parameter
    colors = ["orangered" for i in range(anal_sample_no)] + ["darkblue" for i in range(anal_sample_no)]
    
    if analysis == 'pca':
        # PCA Analysis
        pca = PCA(n_components = number_of_components)
        pca.fit(prep_data)
        pca_results = pca.transform(prep_data)
        pca_hat_results = pca.transform(prep_data_hat)

        # Plotting
        plt.figure(figsize=(10, 8))
        plt.scatter(pca_results[:,0], pca_results[:,1], c = colors[:anal_sample_no], alpha = 0.4, s=5, label = "Original")
        plt.scatter(pca_hat_results[:,0], pca_hat_results[:,1], c = colors[anal_sample_no:], alpha = 0.4, s=5, label = "Synthetic")
        leg = plt.legend()
        leg.get_texts()[0].set_color("orangered")
        leg.get_texts()[1].set_color("darkblue")
        plt.title("PC1 and PC2 components", fontsize=20, pad=15, color='black', fontweight='bold')
        plt.xlabel('PC1', fontsize=16, labelpad=10)
        plt.ylabel('PC2', fontsize=16, labelpad=10)
        plt.xticks(fontsize=13, color='#605e5c')
        plt.yticks(fontsize=13, color='#605e5c')
        plt.grid(False)
        plt.show()
        
    elif analysis == 'tsne':
        
        # Do t-SNE Analysis together       
        prep_data_final = np.concatenate((prep_data, prep_data_hat), axis = 0)

        # TSNE analysis
        time_start = time.time()
        tsne = TSNE(n_components = 2, verbose = 2, perplexity = 40, n_iter = 3000, learning_rate=200)
        # Default perplexity: 30
        # Default learning rate: 200
        # Default maximum number of iterations: 1000
        tsne_results = tsne.fit_transform(prep_data_final)
        print('\nTime elapsed for T-SNE: {} seconds'.format(time.time() - time_start))
        
        # Plotting
        plt.figure(figsize=(10, 8))
        plt.scatter(tsne_results[:anal_sample_no,0], tsne_results[:anal_sample_no,1], c = colors[:anal_sample_no], alpha = 0.4, s=5, label = "Original")
        plt.scatter(tsne_results[anal_sample_no:,0], tsne_results[anal_sample_no:,1], c = colors[anal_sample_no:], alpha = 0.4, s=5, label = "Synthetic")
        leg = plt.legend()
        leg.get_texts()[0].set_color("orangered")
        leg.get_texts()[1].set_color("darkblue")
        plt.title("T-SNE plot", fontsize=20, pad=15, color='black', fontweight='bold')
        plt.xlabel('x-tsne', fontsize=16, labelpad=10)
        plt.ylabel('y-tsne', fontsize=16, labelpad=10)
        plt.xticks(fontsize=13, color='#605e5c')
        plt.yticks(fontsize=13, color='#605e5c')
        plt.grid(False)
        plt.show()