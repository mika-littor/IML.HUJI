from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """
    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        # find the different classes of the y's using set - data structure in which every different value appears
        # only once
        self.classes_ = np.array(sorted(list(set(y))))
        num_classes = len(self.classes_)

        # calculate pi and mu by iterating on each class
        self.pi_ = np.zeros(num_classes)
        self.mu_ = []
        for i, class_ in enumerate(self.classes_):
            # pi is the number of appearances in each class divided by the number of samples
            count = list(y).count(class_)
            self.pi_[i] = count / len(y)
            # mu is the mean of examples per class
            mean_equal = np.mean(X[y == class_], axis=0)
            self.mu_.append(mean_equal)
        self.mu_ = np.array(self.mu_)

        # calculate the cov matrix
        # getting the mean vector per simple
        # The purpose of subtracting the minimum value is to ensure that the calculated indices
        # are aligned with the indices of self.mu_ array.
        normalized_mu_ = self.mu_[y.astype(int)]
        X_diff = X - normalized_mu_
        self.cov_ = np.matmul(X_diff.T, X_diff)
        # calc the unbiased estimator
        self.cov_ = self.cov_ / (len(X) - len(self.classes_))

        # calculate the inv matrix to the cov matrix
        self.cov_inv_ = inv(self.cov_)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        likelihood = self.likelihood(X)
        # for each sample find the indexes with the maximal likelihood
        max_indices = np.argmax(likelihood, axis=1)
        # choose the classes in that indexes
        return self.classes_[max_indices]

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        features_num = X.shape[1]
        # the determinant of the cov matrix
        det_cov = np.linalg.det(self.cov_)
        # normalization factor or z
        normalization_factor = np.sqrt((2 * np.pi) ** features_num * det_cov)
        # changing the dimensions of the x to match mu
        normalized_x = np.expand_dims(X, axis=1)
        # Calculate the difference between each sample and the mean for each class
        diff_x_mu = normalized_x - self.mu_

        # inside the exp - the np.einsum does as follows:
        # 1. Multiply element-wise the diff_x_mu with the self.cov_inv_ -> matrix of shape (N, M, D)
        # 2. Sum the elements along axes with shared dimensions between the input arrays -> matrix of shape (N, M)
        # 3. The '...' notation indicates that the output has the same shape as the remaining axes after contraction,
        # which is (N, M)
        inside_exp = - np.einsum('...j,...jk,...k->...', diff_x_mu, self.cov_inv_, diff_x_mu) / 2

        # return the final result of the likelihood funciton
        return self.pi_ * np.exp(inside_exp) / normalization_factor

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
        predicted = self._predict(X)
        return misclassification_error(y, predicted)
