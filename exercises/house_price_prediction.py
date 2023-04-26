# CSE: mika.li 322851593
import numpy

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
LST_ZIPCODES_COL = []


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


def preprocess_only_on_train_data(df):
    """
    preprocessing of the train data by deleting data which is forbidden over the test data,
    implementing "one hot encoding" on the zipcodes

    Parameters
    ----------
    X : DataFrame of the train set that contains both the x and the y values

    Returns
    -------
    half way pre-processed train data
    """
    # remove rows with nan values in the dataframe
    df.replace(['NA', 'N/A', None], np.nan, inplace=True)
    df = df.dropna(axis=0)
    # dropping duplicates
    df = df.drop_duplicates()
    # change the values in the columns of the dataframe to int
    # convert the columns in the dataframe to type float
    df["price"] = pd.to_numeric(df["price"], errors='coerce').astype(float)

    # filter the database values according to the column
    df = df.query("0<price and 0<sqft_above and 0<sqft_living15 "
                  "and 15<sqft_living and 15<sqft_lot "
                  "and 0<=floors and 0<=sqft_basement "
                  "and 1900<=yr_built<=2023 "
                  "and 0<=yr_renovated<=2023 "
                  "and 0<=bedrooms<=30 "
                  "and 0<=bathrooms<=10 "
                  "and waterfront in (0,  1) "
                  "and 0<=view<=4 "
                  "and 1<=condition<=5 "
                  "and 1<=grade<=13 ")

    df = pd.get_dummies(df, prefix='zipcode_', columns=['zipcode'], dummy_na=False)
    global LST_ZIPCODES_COL
    LST_ZIPCODES_COL = df.filter(regex="^zipcode_").columns.tolist()
    return df


def preprocess_only_on_test_data(X):
    """
    using the LST_ZIPCODES created by the train set to encode the zipcodes

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
    for col in LST_ZIPCODES_COL:
        zipcode_num = col.split("_")[-1]
        X[col] = ((X["zipcode"]) == int(zipcode_num)).astype(int)
    X = X.drop(columns="zipcode", axis=1)
    return X

def convert_columns_float(df):
    """
    convert the columns of df into int
    Parameters
    ----------
    df dataframe

    Returns new df
    -------

    """
    for col in list(df.columns):
        df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
        if col == "zipcode":
            df["zipcode"] = pd.to_numeric(df["zipcode"], errors='coerce').astype(int)
    return df


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
    X = X.drop(["id", "date", "lat", "long", "sqft_lot15"], axis=1)
    # create a column indicates if the building was built or renovated in the past 30 years
    # convert the columns in the dataframe to type float
    X = convert_columns_float(X)
    X["renewed_this_century"] = ((X["yr_renovated"] >= 2000) | (X["yr_built"] >= 2000))

    # do some pre-processing separately as deleting data for example can only be done on the train set
    if y is not None:
        # concatenate the X and y data of the train set for processing purpose
        y.name = "price"
        df_train = pd.concat([X, y], axis=1)
        df_train = preprocess_only_on_train_data(df_train)
        # split again between x and y
        y = df_train["price"]
        X = df_train.drop(columns="price", axis=1)
        return X, y
    else:
        return preprocess_only_on_test_data(X)


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
    # drop the zipcode hot coding as every column matches a few samples
    X = X.drop(X.filter(regex='^zipcode_'), axis=1)

    for feature in X:
        # calculate the correlation
        cov_matrix_feature_response = np.cov(X[feature], y)
        cov_val_feature_response = cov_matrix_feature_response[1, 0]
        denominator = np.std(X[feature]) * np.std(y)
        corr = cov_val_feature_response / denominator

        px.scatter(pd.DataFrame({'x_axis': X[feature], 'y_axis': y}), x="x_axis", y="y_axis", trendline="ols",
                   title=f"{feature} vs Response of Pearson Correlation={corr}",
                   labels={"x_axis": f"feature: {feature}", "y_axis": "Response"},
                   color_discrete_sequence=["blue"],
                   trendline_color_override="green").write_image(output_path + f"/pearson_corr_{f}.png")


if __name__ == '__main__':
    np.random.seed(0)
    df = pd.read_csv("../datasets/house_prices.csv")

    # Question 1 - split data into train and test sets
    train_proportion = 0.75
    df_y = df["price"]
    df_x = df.drop(columns="price", axis=1)
    train_x, train_y, test_x, test_y = split_train_test(df_x, df_y, train_proportion)

    # Question 2 - Preprocessing of housing prices dataset
    # removing from the test data the samples with missing price as instructed on the forum (because then the
    # calculation of the MSE isn't possible)
    test_x, test_y = preprocess_on_invalid_pricing(test_x, test_y)
    # preprocess on the train set
    train_x, train_y = preprocess_data(train_x, train_y)
    # preprocess on the test set
    test_x = preprocess_data(test_x)

    # Question 3 - Feature evaluation with respect to response
    # feature_evaluation(train_x, train_y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    lst_p_val = list(range(10, 101))
    lst_mean_val = []
    lst_sd_val = []
    for p in lst_p_val:
        # insert into a list 10 times the loss calculation over the current p fraction of samples of the data
        lst_samples_loss = []
        for i in range(10):
            X_sample = train_x.sample(frac=p / 100.0)
            y_sample = train_y.loc[X_sample.index]
            regression_val = LinearRegression(include_intercept=True).fit(X_sample, y_sample)
            loss_val = regression_val.loss(test_x, test_y)
            lst_samples_loss.append(loss_val)
        # calculate the mean and sd of the loss
        lst_mean_val.append(np.mean(lst_samples_loss))
        lst_sd_val.append(np.std(lst_samples_loss))

    means = numpy.array(lst_mean_val)
    sds = numpy.array(lst_sd_val)

    upper_border = means + 2 * sds
    lower_border = means - 2 * sds
    go.Figure([go.Scatter(x=lst_p_val, y=upper_border, fill=None, mode="lines", line=dict(color="lightblue")),
                     go.Scatter(x=lst_p_val, y=lower_border, fill='tonexty', mode="lines", line=dict(color="lightblue")),
                     go.Scatter(x=lst_p_val, y=lst_mean_val, mode="markers+lines", marker=dict(color="blue"))],
                    layout=go.Layout(title="MSE of test set over different sample sizes of the train set",
                                     xaxis=dict(title="Sample Percentage (Training Set)"),
                                     yaxis=dict(title="MSE (Test Set)"),
                                     showlegend=False))
    fig.show()
    # fig.write_image("mse.over.training.percentage.png")