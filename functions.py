import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


def import_dataset(name, separator=',', header=None):
    dataset = pd.read_csv("datasets/" + name + ".csv", sep=separator, header=header)
    # dataset = dataset.sample(frac=1)  # shuffle
    return dataset


def get_attributes_and_classes_from_csv(name, class_column, columns_to_cut=None, separator=','):
    def get_attributes_and_classes(dataset, class_column, columns_to_cut=None):
        Y = dataset.loc[:, class_column].values
        X = dataset.loc[:, dataset.columns != class_column].values
        if columns_to_cut:
            X = np.hstack((X[:, :columns_to_cut], X[:, columns_to_cut + 1:]))
        return X, Y

    dataset = import_dataset(name, separator)
    X, Y = get_attributes_and_classes(dataset, class_column, columns_to_cut)
    return X, Y


def get_dataset(name):
    X, Y = [], []
    if name == 'iris':
        X, Y = get_attributes_and_classes_from_csv(name, 4, separator=';')
    if name == 'wine':
        X, Y = get_attributes_and_classes_from_csv(name, 0)
    if name == 'glass':
        X, Y = get_attributes_and_classes_from_csv(name, 10, columns_to_cut=0)
    if name == 'diabetes':
        X, Y = get_attributes_and_classes_from_csv(name, 8)
    return X, Y


def normalize_data(x):
    x = preprocessing.normalize(x)
    return x


def standardize_data(x):
    scaler = StandardScaler()
    scaler.fit(x)
    x = scaler.transform(x)
    return x


def split_data(k, x, y, cross_validation_type):
    cross_val = cross_validation_type(n_splits=k, shuffle=True)
    cross_val.get_n_splits(x, y)
    return cross_val


def fit_values(k, x, y, train_ind, test_ind, f='uniform', metric='euclidean'):
    classifier = KNeighborsClassifier(n_neighbors=k, weights=f, metric=metric)
    classifier.fit(x[train_ind], y[train_ind])
    y_pred = classifier.predict(x[test_ind])
    return y_pred, classifier


def get_metrics(y_true, y_pred, avg='macro'):
    # print(classification_report(y_true, y_pred))
    f1 = f1_score(y_true, y_pred, average=avg, labels=np.unique(y_pred))
    acc = accuracy_score(y_true, y_pred)
    return acc, f1


def weight_function(weights):
    # print('Weights:', weights)
    n, k = weights.shape
    w = np.indices((n, k))[1]
    w = (w.T + np.ones((n,))).T
    w = np.cbrt(np.reciprocal(w))
    return w
