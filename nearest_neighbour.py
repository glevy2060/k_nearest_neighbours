import sys

import numpy as np
import scipy
from classifier import Classifier
from scipy.spatial import distance
import math
import operator
import matplotlib.pyplot as plt
import random
def gensmallm(x_list: list, y_list: list, m: int):
    """
    gensmallm generates a random sample of size m along side its labels.

    :param x_list: a list of numpy arrays, one array for each one of the labels
    :param y_list: a list of the corresponding labels, in the same order as x_list
    :param m: the size of the sample
    :return: a tuple (X, y) where X contains the examples and y contains the labels
    """
    assert len(x_list) == len(y_list), 'The length of x_list and y_list should be equal'

    x = np.vstack(x_list)
    y = np.concatenate([y_list[j] * np.ones(x_list[j].shape[0]) for j in range(len(y_list))])

    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)

    rearranged_x = x[indices]
    rearranged_y = y[indices]

    return rearranged_x[:m], rearranged_y[:m]


def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)


def getKClosestIndexes(x_train, x, k):
    distances = []
    for i in range(len(x_train)):
        dist = np.linalg.norm(x - x_train[i])
        distances.append((x_train[i], dist, i))
    distances.sort(key=operator.itemgetter(1))
    neighbors_indexes = []
    for i in range(k):
        neighbors_indexes.append(distances[i][2])
    return neighbors_indexes


def getMostCommonLabel(indexes, y_train, labels):
    counters = np.zeros((len(labels)))
    for i in indexes:
       index = np.where(labels == y_train[i])[0]
       counters[index] += 1

    max_value = np.max(counters)
    return labels[np.where(counters == max_value)][0]


def learnknn(k: int, x_train: np.array, y_train: np.array):
    x, y = map(np.asarray, (x_train, y_train))
    return Classifier(k, x, y)


def predictknn(classifier, x_test: np.array):
    x_train = classifier.x
    k = classifier.k
    y_train = classifier.y
    y_test = []
    labels = np.unique(y_train)
    for x in x_test:
        indexes = getKClosestIndexes(x_train, x, k)
        y = getMostCommonLabel(indexes, y_train, labels)
        y_test.append(y)

    return np.asarray(y_test).reshape((len(y_test), 1))

def test(examplesNum, k, is_corrupted_labels = False):
    data = np.load('mnist_all.npz')

    train0 = data['train0']
    train1 = data['train1']
    train2 = data['train2']
    train3 = data['train3']

    test0 = data['test0']
    test1 = data['test1']
    test2 = data['test2']
    test3 = data['test3']

    # x_train, y_train = gensmallm([train0, train1, train2, train3], [1, 3, 4, 6], 100)
    x_train, y_train = gensmallm([train0, train1, train2, train3], [1, 3, 4, 6], examplesNum)

    if is_corrupted_labels: #section 2f
        y_train = changeLabelsRandomlly(y_train)

    # x_test, y_test = gensmallm([test0, test1, test2, test3], [1, 3, 4, 6], 100)
    x_test, y_test = gensmallm([test0, test1, test2, test3], [1, 3, 4, 6], len(test0) + len(test1) + len(test2) + len(test3))

    classifier1 = learnknn(k, x_train, y_train)

    preds = predictknn(classifier1, x_test)
    preds = preds.flatten()

    return np.mean(preds != y_test)


def simple_test():
    data = np.load('mnist_all.npz')

    train0 = data['train0']
    train1 = data['train1']
    train2 = data['train2']
    train3 = data['train3']

    test0 = data['test0']
    test1 = data['test1']
    test2 = data['test2']
    test3 = data['test3']

    x_train, y_train = gensmallm([train0, train1, train2, train3], [1, 3, 4, 6], 100)

    x_test, y_test = gensmallm([test0, test1, test2, test3], [1, 3, 4, 6], 50)

    classifier1 = learnknn(1, x_train, y_train)

    preds = predictknn(classifier1, x_test)

    # tests to make sure the output is of the intended class and shape
    assert isinstance(preds, np.ndarray), "The output of the function predictknn should be a numpy array    " 
    assert preds.shape[0] == x_test.shape[0] and preds.shape[1] == 1, f"The shape of the output should be ({x_test.shape[0]}, 1)"

    # get a random example from the test set
    i = np.random.randint(0, x_test.shape[0])

    # this line should print the classification of the i'th test sample.
    print(f"The {i}'th test sample was classified as {preds[i]}")

def q2a():
    sample_size = [20, 30, 50, 80, 100]
    errors = []
    min_errors = []
    max_errors = []
    for s in sample_size:
        err = 0
        min_err = sys.maxsize
        max_err = 0
        for i in range(10):
            curr_err = test(s, 1)
            min_err = min(min_err, curr_err)
            max_err = max(max_err, curr_err)
            err += curr_err
        err /= 10
        errors.append(err)
        min_errors.append(min_err)
        max_errors.append(max_err)

    subtract = [x1 - x2 for (x1, x2) in zip(max_errors, min_errors)]
    plt.plot(sample_size, errors)
    plt.bar(sample_size, min_errors, 1, label = "min", color=['cyan', 'cyan', 'cyan', 'cyan', 'cyan'])
    plt.bar(sample_size, subtract, 1, bottom = min_errors, label = "max", color = ['blue', 'blue', 'blue', 'blue', 'blue'])
    plt.title("average test error")
    plt.xlabel("sample size")
    plt.ylabel("average error")
    plt.legend()
    plt.show()

def q1():
    k = 1
    x_train = np.array([[1, 2], [3, 4], [5, 6]])
    y_train = np.array([1, 0, 1])
    classifier = learnknn(k, x_train, y_train)
    x_test = np.array([[10, 11], [3.1, 4.2], [2.9, 4.2], [5, 6]])
    y_testprediction = predictknn(classifier, x_test)
    print(y_testprediction)


def q2e():
    errors = []
    for k in range(1, 12):
        err = 0
        for i in range(0, 10):
            curr_err = test(100, k)
            err += curr_err
        errors.append(err / 10)

    plt.plot(np.arange(1, 12), errors)
    plt.title("average test error")
    plt.xlabel("k")
    plt.ylabel("average error")
    plt.legend()
    plt.show()



def changeLabelsRandomlly(y_train):
    labels = [1, 3, 4, 6]
    indexes = np.arange(0, 100)
    random20_indexes = random.choices(indexes, k=20)
    for i in random20_indexes:
        i_label = y_train[i]
        new_label = random.choice(labels)
        while i_label == new_label:
            new_label = random.choice(labels)
        y_train[i] = new_label
    return y_train

def q2f():
    errors = []
    for k in range(1, 12):
        err = 0
        for i in range(0, 10):
            curr_err = test(100, k, True)
            err += curr_err
        errors.append(err / 10)

    plt.plot(np.arange(1, 12), errors)
    plt.title("average test error")
    plt.xlabel("k")
    plt.ylabel("average error")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    # q1()
    # q2a()
    #q2e()
    q2f()

    #keep checking the graph of 2e
    # keep doing f


