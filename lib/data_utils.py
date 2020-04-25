import numpy as np
import os
import pickle


def unpickle(file):
    # dict = pickle.load(file, encoding='bytes')
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict


def load_batch_file(file):
    """ load single batch of cifar """
    batch_dict = unpickle(file)
    data = batch_dict['data']
    labels = np.array(batch_dict['labels'])
    data = data.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
    return data, labels


def load_cifar10(dir):
    """ load all of cifar """
    x_list = []
    y_list = []

    for b in range(1, 6):
        # f = os.path.join(dir, 'data_batch_%d' % (b,))
        f = dir+"/data_batch_%d"%(b)
        X, Y = load_batch_file(f)
        x_list.append(X)
        y_list.append(Y)
    X_train = np.concatenate(x_list)
    X_train = X_train/255
    Y_train = np.concatenate(y_list)

    X_test, Y_test = load_batch_file(os.path.join(dir, 'test_batch'))
    X_test = X_test/255

    return X_train, Y_train, X_test, Y_test
