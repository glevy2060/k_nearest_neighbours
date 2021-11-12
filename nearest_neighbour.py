import numpy as np
import scipy
from classifier import Classifier
from scipy.spatial import distance
import math
import operator
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
    return labels[np.where(counters == max_value)[0]]


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


if __name__ == '__main__':
    k = 1
    x_train = np.array([[1, 2], [3, 4], [5, 6]])
    y_train = np.array([1, 0, 1])
    classifier = learnknn(k, x_train, y_train)
    x_test = np.array([[10, 11], [3.1, 4.2], [2.9, 4.2], [5, 6]])
    y_testprediction = predictknn(classifier, x_test)
    print(y_testprediction)
    # before submitting, make sure that the function simple_test runs without errors
    simple_test()


