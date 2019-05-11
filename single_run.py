from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier

from functions import *
from plots import *

import warnings
warnings.filterwarnings('ignore')


cross_validation_types = [KFold, StratifiedKFold]
folds_values = [2, 3, 5, 10]
k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30]
weights = ['uniform', 'distance', weight_function]
metrics = ['euclidean', 'manhattan']
datasets = ['wine', 'glass', 'diabetes']
data_standarized = [True, False]


if __name__ == '__main__':
    x, y = get_dataset('diabetes')
    x = standardize_data(x)  # changing very much

    folds = 2
    k = 7
    Folds = split_data(folds, x, y, cross_validation_types[0])
    for train, test in Folds.split(x, y):
        y_pred, classifier = fit_values(k, x, y, train, test, f=weights[0], metric=metrics[0])

        acc, f1 = get_metrics(y[test], y_pred)
        print(f1, acc)

