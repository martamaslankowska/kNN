import statistics
from single_run import *

iterations = 10
columns = ['cross_val', 'is_stand', 'fold', 'k', 'weight', 'metric', 'acc', 'f1']


if __name__ == '__main__':
    for dataset in datasets:
        file_name = f'{dataset}_values'
        df = pd.DataFrame(columns=columns)
        x, y = get_dataset(dataset)

        for standarized in data_standarized:
            if standarized:
                x = standardize_data(x)  # changing very much

            for cross_val_type in cross_validation_types:
                for folds in folds_values:
                    print(f'Testing for {dataset.upper()} dataset with standarized = {standarized} '
                          f'| using {cross_val_type.__name__} with {folds} folds')
                    for k in k_values:
                        for weight in weights:
                            for metric in metrics:
                                for i in range(iterations):  # taking mean of all iterations
                                    f1_values, acc_values = [], []
                                    Folds = split_data(folds, x, y, cross_val_type)
                                    for train, test in Folds.split(x, y):  # not a parameter
                                        y_pred, classifier = fit_values(k, x, y, train, test,
                                                                        f=weight, metric=metric)
                                        acc, f1 = get_metrics(y[test], y_pred)
                                        acc_values.append(acc)
                                        f1_values.append(f1)
                                acc = statistics.mean(acc_values)
                                f1 = statistics.mean(f1_values)

                                df = df.append({'cross_val': cross_val_type.__name__,
                                                'is_stand': standarized, 'fold': folds, 'k': k,
                                                'weight': weight if isinstance(weight, str) else 'mma',
                                                'metric': metric, 'acc': acc, 'f1':f1},
                                               ignore_index=True)

        df.to_csv(file_name + '.csv')