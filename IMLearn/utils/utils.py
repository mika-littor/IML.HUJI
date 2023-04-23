# CSE: mika.li 322851593
from typing import Tuple
import numpy as np
import pandas as pd


def split_train_test(X: pd.DataFrame, y: pd.Series, train_proportion: float = .75) \
        -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Randomly split given sample to a training- and testing sample

    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Data frame of samples and feature values.

    y : Series of shape (n_samples, )
        Responses corresponding samples in data frame.

    train_proportion: Fraction of samples to be split as training set

    Returns
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples

    """
    # create shuffled list of the indexes of the rows that represent samples in X
    shuffled_indexes = np.random.choice(X.index, size=len(X), replace=False)
    # get the shuffled X and y
    X_shuffled = X.reindex(shuffled_indexes)
    y_shuffled = y.reindex(shuffled_indexes)

    # get the number of rows in the training set according to the train_proportion
    train_samples_num = int(X.shape[0] * train_proportion)

    train_x = pd.DataFrame(X_shuffled.iloc[:train_samples_num])
    test_x = pd.DataFrame(X_shuffled.iloc[train_samples_num:])
    train_y = pd.Series(y_shuffled.iloc[:train_samples_num])
    test_y = pd.Series(y_shuffled.iloc[train_samples_num:])
    return train_x, train_y, test_x, test_y



def confusion_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute a confusion matrix between two sets of integer vectors

    Parameters
    ----------
    a: ndarray of shape (n_samples,)
        First vector of integers

    b: ndarray of shape (n_samples,)
        Second vector of integers

    Returns
    -------
    confusion_matrix: ndarray of shape (a_unique_values, b_unique_values)
        A confusion matrix where the value of the i,j index shows the number of times value `i` was found in vector `a`
        while value `j` vas found in vector `b`
    """
    raise NotImplementedError()
