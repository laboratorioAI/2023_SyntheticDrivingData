"""
Author: Diego Tamayo
Based on Time-series Generative Adversarial Networks (TimeGAN) Codebase.
Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks,"
Neural Information Processing Systems (NeurIPS), 2019.
"""

import numpy as np


def MinMaxScaler(data):
    """Min Max normalizer.
    Args:
    - data: original data
    Returns:
    - norm_data: normalized data
    """
    min_vals = np.min(data, 0)
    numerator = data - np.min(data, 0)
    max_vals = np.max(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    norm_data = numerator / (denominator + 1e-7)
    return norm_data, min_vals, max_vals


def real_data_loading(data_name, seq_len):
    """Load and preprocess real-world datasets.
    Args:
    - data_name: stock or energy
    - seq_len: sequence length
    Returns:
    - data: preprocessed data.
    """
    assert data_name in ['serie_temporal']
    ori_data = np.loadtxt('data/driving_event_dataset.csv', delimiter=",", skiprows=1)
    ori_data = ori_data[::-1]

    # 2D Data escalation.
    ori_data, min_vals, max_vals = MinMaxScaler(ori_data)

    # Reshape 2D data into 3D tensor.
    temp_data = []
    for i in range(0, len(ori_data) - seq_len + 1):
        _x = ori_data[i:i + seq_len]
        temp_data.append(_x)

    # Mix the datasets (to make it similar to i.i.d)
    idx = np.random.permutation(len(temp_data))

    data = []
    for i in range(len(temp_data)):
        data.append(temp_data[idx[i]])

    return len(ori_data), data, min_vals, max_vals