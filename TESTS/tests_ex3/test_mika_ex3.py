import numpy as np

from TESTS.tests_ex3 import loss_functions_ans as lf_ans
from IMLearn.metrics import loss_functions as lf


def test_misclassification_error():
    arr1 = np.array([1, 2, 3, 4, 5])
    arr2 = np.array([1, 2, 3, 6, 6])
    print(lf.misclassification_error(arr1, arr2, False))
    print(lf_ans.misclassification_error(arr1, arr2, False))

def test_accuracy():
    arr1 = np.array([1, 2, 3, 4, 5, 900])
    arr2 = np.array([1, 2, 3, 6, 6, 5])
    print(lf.accuracy(arr1, arr2))
    print(lf_ans.accuracy(arr1, arr2))

def main():
    test_accuracy()

if __name__ == "__main__":
    main()
