import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler


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
    cross_val = cross_validation_type(n_splits=k)
    cross_val.get_n_splits(x, y)
    return cross_val






'''

LOAD GLASS DATASET
```{r}
glass <- read.csv(file="./datasets/glass.csv", head=TRUE, sep=",", fileEncoding="UTF-8-BOM")
glass <- glass[, -1]
colnames(glass)[10] <- "class"
```

LOAD WINE DATASET
```{r}
wine <- read.csv(file="./datasets/wine.csv", head=FALSE)
colnames(wine) <- c("class","alcohol","acid","ash","alcalinity","magnesium","phenols","flavanoids","nonphenols","procyanins","color","hue","OD","proline")

head(wine)
```

LOAD DIABETES DATASET
```{r}
diabetes <- read.csv(file="./datasets/diabetes.csv", head=TRUE, sep=",", fileEncoding="UTF-8-BOM")
colnames(diabetes)[9] <- "class"
```

LOAD WHOLESALES DATASET
```{r}
sales <- read.csv(file="./datasets/wholesale.csv", head=TRUE, sep=",", fileEncoding="UTF-8-BOM")
```


'''