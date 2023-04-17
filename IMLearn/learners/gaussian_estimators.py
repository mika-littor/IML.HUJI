from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm, multivariate_normal
from numpy.linalg import inv, det, slogdet


# CSE: mika.li 322851593


class UnivariateGaussian:
    """
    Class for univariate Gaussian Distribution Estimator
    """

    def __init__(self, biased_var: bool = False) -> UnivariateGaussian:
        """
        Estimator for univariate Gaussian mean and variance parameters

        Parameters
        ----------
        biased_var : bool, default=False
            Should fitted estimator of variance be a biased or unbiased estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `UnivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `UnivariateGaussian.fit`
            function.

        var_: float
            Estimated variance initialized as None. To be set in `UnivariateGaussian.fit`
            function.
        """
        self.biased_ = biased_var
        self.fitted_, self.mu_, self.var_ = False, None, None

    def fit(self, X: np.ndarray) -> UnivariateGaussian:
        """
        Estimate Gaussian expectation and variance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.var_` attributes according to calculated estimation (where
        estimator is either biased or unbiased). Then sets `self.fitted_` attribute to `True`
        """
        self.mu_ = X.mean(axis=0)
        if self.biased_:
            self.var_ = X.var(ddof=0, axis=0)
        else:
            self.var_ = X.var(ddof=1, axis=0)
        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, var_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")
        return norm.pdf(X, self.mu_, math.sqrt(self.var_))

    @staticmethod
    def log_likelihood(mu: float, sigma: float, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        sigma : float
            Variance of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        len_X = len(X)
        mult_val = 1 / math.pow(2 * sigma * math.pi, len_X / 2)
        in_exp_mult = - 1 / (2 * sigma)
        vec_mu = np.repeat(-1 * mu, len_X)
        sum_in_exp = np.sum(np.power(np.add(X, vec_mu), 2))
        exp_val = math.exp(in_exp_mult * sum_in_exp)
        return np.emath.log(mult_val * exp_val)


class MultivariateGaussian:
    """
    Class for multivariate Gaussian Distribution Estimator
    """

    def __init__(self):
        """
        Initialize an instance of multivariate Gaussian estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `MultivariateGaussian.fit` function.

        mu_: ndarray of shape (n_features,)
            Estimated expectation initialized as None. To be set in `MultivariateGaussian.fit`
            function.

        cov_: ndarray of shape (n_features, n_features)
            Estimated covariance initialized as None. To be set in `MultivariateGaussian.fit`
            function.
        """
        self.mu_, self.cov_ = None, None
        self.fitted_ = False

    def fit(self, X: np.ndarray) -> MultivariateGaussian:
        """
        Estimate Gaussian expectation and covariance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Training data

        Returns
        -------
        self : returns an instance of self

        Notes
        -----
        Sets `self.mu_`, `self.cov_` attributes according to calculated estimation.
        Then sets `self.fitted_` attribute to `True`
        """
        # get mean of each column
        self.mu_ = X.mean(axis=0)
        self.cov_ = np.cov(X.T, bias=False)
        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray):
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, cov_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")
        # return multivariate_normal.pdf(X, self.mu_, self.cov_, allow_singular=True)

        # if only one dimension
        if (isinstance(self.mu_, np.floating)):
            instance = UnivariateGaussian()
            instance.fit(X)
            return instance.pdf(X)
        # otherwise more than one dimension
        d = len(self.mu_)
        first_mul = 1 / math.sqrt(math.pow(2 * math.pi, d) * det(self.cov_))
        inside_exp = -0.5 * (np.dot(np.dot((X - self.mu_), inv(self.cov_)), (X - self.mu_).T))
        # we use einsum because the pdf for each vector is on the main diagonal as X is a metrix of the vectors
        return first_mul * np.exp(np.einsum("aa->a", inside_exp))

    @staticmethod
    def log_likelihood(mu: np.ndarray, cov: np.ndarray, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : ndarray of shape (n_features,)
            Expectation of Gaussian
        cov : ndarray of shape (n_features, n_features)
            covariance matrix of Gaussian
        X : ndarray of shape (n_samples, n_features)
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated over all input data and under given parameters of Gaussian
        """
        # if only one dimension
        if (isinstance(mu, np.floating)):
            instance = UnivariateGaussian()
            instance.fit(X)
            return instance.log_likelihood(instance.mu_, instance.var_, X)
        # otherwise more than one dimension
        d = len(mu)
        m = len(X)
        first_add = - (m / 2) * np.emath.log(math.pow(2 * math.pi, d))
        seconds_add = - (m / 2) * np.emath.log(np.linalg.det(cov))
        third_add = - (1 / 2) * np.einsum("ab,bk,ka", (X - mu), inv(cov), (X - mu).T)
        return first_add + seconds_add + third_add
