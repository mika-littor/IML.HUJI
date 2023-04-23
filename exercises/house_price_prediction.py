# CSE: mika.li 322851593
from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


# global variable for the zipcodes in the train set
LST_ZIPCODES = []

#todo: in train:
# - delete data that is not suitable as pricing that is < 0, make sure to delete the same line on the x
# part.
# - do hotsopt for the zipcode column
# - date irrelevant (sale)


def preprocess_on_invalid_pricing(X, y):
    """
    removing from the test (and the train) data the samples with missing price as instructed on the forum (because
    then the calculation of the MSE isn't possible)

    Parameters
    ----------
    X the df of the test without pricing
    y the pricing dt of the test

    Returns
    -------
    the X after removal
    """
    pricing_values = y.values
    # get indexes of non nan values
    non_nan_indexes_y = set(y.notnull().index)
    # get indexes of positive values
    positive_indexes = set(y.index[pricing_values > 0].tolist())
    clean_x = X.loc[list(non_nan_indexes_y & positive_indexes)]
    clean_y = y.loc[list(non_nan_indexes_y & positive_indexes)]
    return clean_x, clean_y


def preprocess_remove_nan_or_neg(X, y, col_to_edit, remove_zero):
    """
    removes from the train set negative values that are invalid or values that are np.nan
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector corresponding given samples

    col_to_edit: the column name in the df to delete the values from

    remove_zero: indicator if to remove also the value of zero

    Returns
    -------
    X and y after the removal
    """
    # the values in the column
    pricing_values = X[col_to_edit].values
    # get indexes of non nan values
    non_nan_indexes_x = set(X.notnull().index)
    if remove_zero:
        # get indexes of positive values
        positive_indexes = set(y.index[pricing_values > 0].tolist())
    else:
        # get the indexes of the non-negative values
        positive_indexes = set(y.index[pricing_values >= 0].tolist())
    clean_x = X.loc[list(non_nan_indexes_x & positive_indexes)]
    clean_y = y.loc[list(non_nan_indexes_x & positive_indexes)]
    return clean_x, clean_y

def preprocess_only_on_train_data(X, y):
    """
    preprocessing of the train data by deleting data which is forbidden over the test data,
    implementing "one hot encoding" on the zipcodes

    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector corresponding given samples

    Returns
    -------
    half way pre-processed train data
    """

    # remove invalid pricing data
    X, y = preprocess_on_invalid_pricing(X, y)

    # remove values from the train set that are negative / nan
    to_drop_neg = ["sqft_living", "sqft_lot", "sqft_above", "yr_built"]
    to_drop_non_positive = ["bathrooms", "floors", "sqft_basement", "yr_renovated"]
    for col in to_drop_neg:
        X, y = preprocess_remove_nan_or_neg(X, y, col, remove_zero=True)
    for col in to_drop_non_positive:
        X, y = preprocess_remove_nan_or_neg(X, y, col, remove_zero=False)

    # save the zipcodes for the goal of using hotspot also on the test set
    return X, y

def preprocess_only_on_test_data(X):
    """
    preprocessing of the test data by using the LST_ZIPCODES created by the train part to encode the zipcodes

    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector corresponding given samples

    Returns
    -------
    half way pre-processed train data
    """
    return X



def preprocess_data(X: pd.DataFrame, y: Optional[pd.Series] = None):
    """
    preprocess data
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector corresponding given samples

    Returns
    -------
    Post-processed design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    # remove irrelevant columns or columns that have negative correlation with the price
    X = X.drop(["id", "date", "sqft_lot15", "sqft_living15"], axis=1)

    # do some pre-processing separately as deleting data for example can only be done on the train set
    if y is not None:
        X, y = preprocess_only_on_train_data(X, y)
    else:
        X = preprocess_only_on_test_data(X)

    return X, y


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    df = pd.read_csv("../datasets/house_prices.csv")

    #todo: remove command
    df = pd.read_csv(r"C:\Users\mikal\Documents\CS4\IML\IML.HUJI\TESTS\house_prices_test.csv").head(10)

    # Question 1 - split data into train and test sets
    train_proportion = 0.75
    df_y = df["price"]
    df_x = df.drop(columns="price", axis=1)
    train_x, train_y, test_x, test_y = split_train_test(df_x, df_y, train_proportion)

    # Question 2 - Preprocessing of housing prices dataset
    # removing from the test data the samples with missing price as instructed on the forum (because then the
    # calculation of the MSE isn't possible)
    test_x, test_y = preprocess_on_invalid_pricing(test_x, test_y)
    train_x, train_y = preprocess_data(train_x, train_y)
    print("train x:\n", train_x)
    print("train y:\n", train_y)

    test_x = preprocess_data(test_x)
    print("test x:\n", test_x)

    # Question 3 - Feature evaluation with respect to response

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)