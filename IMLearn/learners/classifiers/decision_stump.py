from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """

    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit a decision stump to the given data. That is, finds the best feature and threshold by which to split

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        best_err = float('inf')
        classification_signs = [-1, 1]
        # iterate over the possible features
        num_features = X.shape[1]
        for j in range(num_features):
            for sign in classification_signs:
                # find the threshold over the j feature
                j_feature = X[:, j]
                threshold, threshold_err = self._find_threshold(j_feature, y, sign)
                # check if the threshold feature is better than what was already found
                if threshold_err < best_err:
                    best_err = threshold_err
                    self.threshold_ = threshold
                    self.j_ = j
                    self.sign_ = sign
                    if best_err == 0:
                        # achieved the ultimate error
                        return


def _predict(self, X: np.ndarray) -> np.ndarray:
    """
    Predict sign responses for given samples using fitted estimator

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data to predict responses for

    y : ndarray of shape (n_samples, )
        Responses of input data to fit to

    Returns
    -------
    responses : ndarray of shape (n_samples, )
        Predicted responses of given samples

    Notes
    -----
    Feature values strictly below threshold are predicted as `-sign` whereas values which equal
    to or above the threshold are predicted as `sign`
    """
    predictions = np.empty(X.shape[0], dtype=int)
    below_th = X[:, self.j_] < self.threshold_
    # features below the threshold
    predictions[below_th] = -self.sign_
    # features equal or above the threshold
    above_th = ~below_th
    predictions[above_th] = self.sign_
    return predictions


def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
    """
    Given a feature vector and labels, find a threshold by which to perform a split
    The threshold is found according to the value minimizing the misclassification
    error along this feature

    Parameters
    ----------
    values: ndarray of shape (n_samples,)
        A feature vector to find a splitting threshold for

    labels: ndarray of shape (n_samples,)
        The labels to compare against

    sign: int
        Predicted label assigned to values equal to or above threshold

    Returns
    -------
    thr: float
        Threshold by which to perform split

    thr_err: float between 0 and 1
        Misclassificaiton error of returned threshold

    Notes
    -----
    For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
    which equal to or above the threshold are predicted as `sign`
    """
    raise NotImplementedError()


def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
    """
    Evaluate performance under misclassification loss function

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Test samples

    y : ndarray of shape (n_samples, )
        True labels of test samples

    Returns
    -------
    loss : float
        Performance under missclassification loss function
    """
    from ...metrics import misclassification_error
    to_predict = self._predict(X)
    return misclassification_error(y, to_predict)
