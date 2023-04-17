import gaussian_estimators
import numpy as np

from gaussian_estimators import UnivariateGaussian as ug1
from gaussian_estimators_answers import UnivariateGaussian as ug2

from gaussian_estimators import MultivariateGaussian as mg1
from gaussian_estimators_answers import MultivariateGaussian as mg2


def print_test_uni(instance, arr):
    instance.fit(arr)
    print(instance.mu_)
    print(instance.var_)
    print(instance.pdf(arr))
    print(instance.log_likelihood(instance.mu_, instance.var_, arr))


def test_univariate():
    print("test univariate\n")
    instance1 = ug1()
    instance2 = ug2()
    arr = np.array([1.5, 2.6, 3.3])
    print("first answers")
    print_test_uni(instance1, arr)
    print("\nsecond answers")
    print_test_uni(instance2, arr)


def print_test_mult(instance, arr):
    instance.fit(arr)
    print(instance.mu_)
    print(instance.cov_)
    print(instance.pdf(arr))
    print(instance.log_likelihood(instance.mu_, instance.cov_, arr))


def test_multivariant():
    print("test Multivariant\n")
    instance1 = mg1()
    instance2 = mg2()
    arr = np.array([[1, 2, 3], [ 4.1, 5.6, 7.8], [8, 0, 2]])
    # arr = np.array([1, 2, 3])
    print("first answers")
    print_test_mult(instance1, arr)
    print("\nsecond answers")
    print_test_mult(instance2, arr)


def main():
    test_univariate()


if __name__ == "__main__":
    main()
