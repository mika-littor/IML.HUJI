from typing import NoReturn
from ...base import BaseEstimator
import numpy as np

class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.fitted_ = True
        # find the different classes of the y's using set - data structure in which every different value appears
        # only once
        self.classes_ = np.array(sorted(list(set(y))))
        num_classes = len(self.classes_)

        # calculate pi, mu and vars by iterating on each class
        self.pi_ = np.zeros(num_classes)
        self.mu_ = []
        self.vars_ = []
        for i, class_ in enumerate(self.classes_):
            # pi is the number of appearances in each class divided by the number of samples
            count = list(y).count(class_)
            self.pi_[i] = count / len(y)
            # mu is the mean of examples per class
            X_in_class = X[y == class_]
            self.mu_.append(np.mean(X_in_class, axis=0))
            # vars is the variance of examples per class
            self.vars_.append(np.var(X_in_class, axis=0, ddof=1))
        self.mu_ = np.array(self.mu_)
        self.vars_ = np.array(self.vars_)

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

        # changing the dimensions of the x to match mu
        normalized_x = np.expand_dims(X, axis=1)
        # Calculate the difference between each sample and the mean for each class
        diff_x_mu = normalized_x - self.mu_
        inside_exp = np.power(diff_x_mu, 2) / (-2 * self.vars_)
        denominator = np.sqrt(2 * np.pi * self.vars_)
        # element-wise division
        div_in_exp = np.exp(inside_exp) / denominator
        # applying the np.prod function along the 2nd axis of div_in_exp, in order
        # to calculate the joint probabilities of the features for each class.
        likelihood = np.apply_along_axis(np.prod, axis=2, arr=div_in_exp)
        return self.pi_ * likelihood


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
        return misclassification_error(y_true=y, y_pred=self._predict(X))
