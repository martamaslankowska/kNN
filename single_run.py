from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier

from functions import *

cross_validation_types = [KFold, StratifiedKFold]


if __name__ == '__main__':
    x, y = get_dataset('wine')
    x = standardize_data(x)  # changing very much

    Folds = split_data(2, x, y, cross_validation_types[1])
    for train, test in Folds.split(x, y):
        classifier = KNeighborsClassifier(n_neighbors=10)
        classifier.fit(x[train], y[train])

        y_pred = classifier.predict(x[test])

        print(confusion_matrix(y[test], y_pred))
        print(classification_report(y[test], y_pred))
