import numpy as np


def make_batches(data1, data2=None, batch_size=1, axis=0):
    """
    Function to split training data into batches given a numpy array, batch size
    and the axis for splitting.
    It can take two numpy arrays as inputs -
        A primary array which is a non-default argument
        A secondary array which can be considered as a default argument
    In case of two arguments, data is shuffled uniformly.
    :param data1: Primary array of input data to be split into batches
    :param data2:Secondary  Array of input data to be split into batches
    :param axis: An integer value giving the Axis to split the data on
    :param batch_size: An integer value giving the size of each batch
    :return:
        if data2 is None : A list of arrays
        else : A tuple containing two lists of arrays, for corresponding
         training and label sets
    """

    data_size = data1.shape[axis]
    idx = np.random.permutation(data_size)

    data1 = data1[idx]
    num_batches = data_size // batch_size
    residue = data_size % batch_size
    data1_batches = np.split(data1[:data_size - residue], num_batches)
    if residue != 0:
        data1_batches = data1_batches + [data1[data_size - residue:]]

    if data2 is None :
        return data1_batches

    data2 = data2[idx]
    num_batches = data_size // batch_size
    residue = data_size % batch_size
    data2_batches = np.split(data2[:data_size - residue], num_batches)
    if residue != 0:
        data2_batches = data2_batches + [data2[data_size - residue:]]

    return data1_batches, data2_batches