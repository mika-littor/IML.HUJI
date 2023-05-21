import numpy as np
from numpy.linalg import det, inv

from TESTS.tests_ex3 import loss_functions_ans as lf_ans
from IMLearn.metrics import loss_functions as lf
from TESTS.tests_ex3 import perceptron_ans as per_ans
from IMLearn.learners.classifiers import perceptron as per

from TESTS.tests_ex3 import linear_discriminant_analysis_ans as lda_ans
from IMLearn.learners.classifiers import linear_discriminant_analysis as lda

from TESTS.tests_ex3 import gaussian_naive_bayes_ans as gnb_ans
from IMLearn.learners.classifiers import gaussian_naive_bayes as gnb


def f_test_misclassification_error():
    arr1 = np.array([1, 2, 3, 4, 5])
    arr2 = np.array([1, 2, 3, 6, 6])
    print(lf.misclassification_error(arr1, arr2, False))
    print(lf_ans.misclassification_error(arr1, arr2, False))


def test_accuracy():
    arr1 = np.array([1, 2, 3, 4, 5, 900])
    arr2 = np.array([1, 2, 3, 6, 6, 5])
    print(lf.accuracy(arr1, arr2))
    print(lf_ans.accuracy(arr1, arr2))


def f_test_perceptron():
    # Generate some random data for testing
    X = np.array([[1, 2], [0.3, 4], [5, 6], [-7, 528]])
    y = np.array([-1, 1, -1, 1])

    # Create an instance of the Perceptron classifier
    perceptron1 = per.Perceptron(include_intercept=True, max_iter=1000)
    perceptron2 = per_ans.Perceptron(include_intercept=True, max_iter=1000)

    # Define a custom callback function for testing
    def custom_callback1(fit: per, x: np.ndarray, y: int):
        pass
        # print(f"mine: Sample: {x}, Label: {y}")

    def custom_callback2(fit: per_ans, x: np.ndarray, y: int):
        pass
        # print(f"answers: Sample: {x}, Label: {y}")

    # Set the callback function in the Perceptron instance
    perceptron1.callback_ = custom_callback1
    perceptron2.callback_ = custom_callback2

    # Fit the classifier to the data
    perceptron1.fit(X, y)
    perceptron2.fit(X, y)

    # # Print the coefficients of the fitted model
    print("mine: Fitted Coefficients", perceptron1.coefs_)
    print("answer: Fitted Coefficients", perceptron2.coefs_)

    # Predict the labels for new data
    new_X = np.array([[7.9, 89999999999], [-0.189, 1000]])
    predictions1 = perceptron1.predict(new_X)
    predictions2 = perceptron2.predict(new_X)
    print("mine: Predictions:", predictions1)
    print("answer: Predictions:", predictions2)

    # # Calculate the misclassification loss on the test data
    test_X = np.array([[1.55, 224.02], [-443, 94], [5, 6]])
    test_y = np.array([-1, 1, 1])
    loss = perceptron1.loss(test_X, test_y)
    print("mine: Misclassification Loss:", loss)
    loss = perceptron2.loss(test_X, test_y)
    print("answer:  Misclassification Loss:", loss)


def f_test_run_perceptron():
    losses = [1, 3, 4, 5, 9]
    print(list(range(len(losses))))
    print(np.arange(len(losses)).tolist())


def f_test_lda_fit():
    X = np.array([[10, -2], [32, 4], [-905, 6], [7.5, 8], [9, 140], [151, 12], [13, 14], [15, 16], [17, 18], [19, 20]])
    y = np.array([0, 5, -47, 1, 1, 144, 20, 2, 3, 3])
    classes_ = np.array(sorted(list(set(y))))
    num_classes = len(classes_)
    # the pi is the number of appearances in each class divided by the number of total classes
    pi_ = np.zeros(num_classes)
    for i, class_ in enumerate(classes_):
        # go through each class and insert calculate its pi value 
        count = list(y).count(class_)
        pi_[i] = count / len(y)
    # print(classes_)
    # print(pi_)

    classes_, pi_ = np.unique(y, return_counts=True)
    pi_ = pi_ / len(y)
    # print(classes_)
    # print(pi_)

    mu_ = np.array([np.mean(X[y == c], axis=0) for c in classes_])
    # print(mu_)

    mu_ = []
    for c in classes_:
        # calculate the mean along the features
        mean_equal = np.mean(X[y == c], axis=0)
        mu_.append(mean_equal)
    mu_ = np.array(mu_)
    # print(mu_)

    c = X - mu_[y.astype(int)]
    cov_ = np.einsum("ki,kj->kij", c, c).sum(axis=0) / (len(X) - len(classes_))
    print(cov_)

def f_test_cov_lda():
    X = np.array([[1, 100], [3, 4], [50, 600], [7, 8]])
    y = np.array([1, 0, 0, 1.5])
    classes_, pi_ = np.unique(y, return_counts=True)
    pi_ = pi_ / len(y)

    mu_ = np.array([np.mean(X[y == c], axis=0) for c in classes_])
    c = X - mu_[y.astype(int)]
    cov_ = np.einsum("ki,kj->kij", c, c).sum(axis=0) / (len(X) - len(classes_))
    print(cov_)

    normalized_mu_ = mu_[y.astype(int)]
    X_diff = X - normalized_mu_
    cov_ = np.matmul(X_diff.T, X_diff)
    # calc the unbiased estimator
    cov_ = cov_ / (len(X) - len(classes_))
    print(cov_)

    y_mu = np.array([mu_[np.where(classes_ == int(y_val))][0] for y_val in y])
    x_mu = X - y_mu
    cov_ = np.einsum("ab,ac->abc", x_mu, x_mu).sum(axis=0) / (len(X) - len(classes_))
    print(cov_)


def f_test_lda():
    X = np.array([[1, 4], [3, 4], [1, 6], [7, 8], [5, 8], [1, 6]])
    y = np.array([1, 1, 0, 1.5, 2, 2])

    # Instantiate and fit the LDA classifier
    lda1 = lda_ans.LDA()
    lda1.fit(X, y)

    lda2 = lda.LDA()
    lda2.fit(X, y)

    print("Ans like:\n", lda1.likelihood(X))
    print("Mine like:\n", lda2.likelihood(X))

    print("Ans loss:\n", lda1.loss(X, y))
    print("Mine loss:\n", lda2.loss(X, y))


def compare_fit_naive_bayes():
    X = np.array([[1, 2]])
    y = np.array([0, 1])
    
    ans_classes_, ans_pi_ = np.unique(y, return_counts=True)
    ans_pi_ = ans_pi_ / len(y)
    ans_mu_ = np.array([np.mean(X[y == c], axis=0) for c in ans_classes_])
    ans_vars_ = np.array([np.var(X[y == c], axis=0, ddof=1) for c in ans_classes_])


    min_classes_ = np.array(sorted(list(set(y))))
    num_classes = len(min_classes_)

    # calculate pi, mu and vars by iterating on each class
    min_pi_ = np.zeros(num_classes)
    min_mu_ = []
    min_vars_ = []
    for i, class_ in enumerate(min_classes_):
        # pi is the number of appearances in each class divided by the number of samples
        count = list(y).count(class_)
        min_pi_[i] = count / len(y)
        # mu is the mean of examples per class
        X_in_class = X[y == class_]
        min_mu_.append(np.mean(X_in_class, axis=0))
        # vars is the variance of examples per class
        min_vars_.append(np.var(X_in_class, axis=0, ddof=1))
    min_mu_ = np.array(min_mu_)
    min_vars_ = np.array(min_vars_)
    
    print("ans classes:\n", ans_classes_)
    print("mine classes:\n", min_classes_)

    print("ans pi:\n", ans_pi_)
    print("mine pi:\n", min_pi_)

    print("ans mu:\n", ans_mu_)
    print("mine mu:\n", min_mu_)

    print("ans vars:\n", ans_vars_)
    print("mine vars:\n", min_vars_)

def f_test_gaussian():
    X = np.array([[1, 10], [3, 1], [0, 3], [0, 8]])
    y = np.array([2, 1, 0, 1])

    # Instantiate and fit the LDA classifier
    gnb1 = gnb_ans.GaussianNaiveBayes()
    gnb1.fit(X, y)

    gnb2 = gnb.GaussianNaiveBayes()
    gnb2.fit(X, y)

    print("Ans like:\n", gnb1.likelihood(X))
    print("Mine like:\n", gnb2.likelihood(X))

    arr_loss = np.array([0,0,0,0])
    print("Ans loss:\n", gnb1.loss(X, arr_loss))
    print("Mine loss:\n", gnb2.loss(X, arr_loss))


def main():
    f_test_lda()


if __name__ == "__main__":
    main()
