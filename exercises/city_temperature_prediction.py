import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    # load data
    df = pd.read_csv(filename, parse_dates=["Date"]).dropna().drop_duplicates()
    # drop null data that is invalid
    df.replace(['NA', 'N/A', None], np.nan, inplace=True)
    df = df.dropna().drop_duplicates()
    # add dayofyear column
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df["Year"] = df["Year"].astype(str)

    # deal with invalid data - delete the columns in which the year and the temp is invalid
    df = df[df["Temp"].astype(int) >= 0]
    df = df[df["Year"].astype(int) >= 0]
    df = df[df["Year"].astype(int) <= 2023]

    # replace the day and month with average were it is invalid
    mean_month = int(df[df["Month"].astype(int).between(1, 12)]["Month"].mean())
    mean_day = int(df[df["Day"].astype(int).between(1, 31)]["Day"].mean())
    df["Month"] = df["Month"].astype(int).apply(lambda x: x if x in range(1, 13) else mean_month)
    df["Day"] = df["Day"].astype(int).apply(lambda x: x if x in range(1, 32) else mean_day)

    # change the type of the columns for the other functions
    return df


def explore_israel_data(data):
    """
    explores the data of israel
    Parameters
    ----------
    data of the countries

    Returns
    -------

    """
    # Plot a scatter plot showing this relation, and color code the dots by the different years
    px.scatter(israel_df, x="DayOfYear", y="Temp", color="Year", title="Avg Daily Temp in Israel vs DayOfYear",
               labels={"x": "DayOfYear", "y": "Averaged Temperature (fahrenheit)"}).write_image("avg_temp_israel.png")

    # plot a bar plot showing for each month the standard deviation of the daily temperatures
    israel_data_sd = pd.DataFrame({'Month': israel_df['Month'].unique(),
                                   'std': israel_df.groupby('Month')['Temp'].std().values})
    px.bar(israel_data_sd,
           title="SD Temp vs Years", x="Month", y="std").write_image("month_avg_temp_israel.png")


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data("../datasets/city_temperature.csv")

    # Question 2 - Exploring data for specific country
    israel_df = df[df["Country"] == "Israel"]
    explore_israel_data(israel_df)

    # Question 3 - Exploring differences between countries
    df_country_month = df.groupby(["Country", "Month"], as_index=False).agg(mean=("Temp", "mean"), std=("Temp", "std"))
    px.line(df_country_month,
            x="Month", y="mean", error_y="std", color="Country") \
        .update_layout(title="Avg Temp per Month",
                       xaxis_title="Month",
                       yaxis_title="Averaged Temperature (fahrenheit)").write_image("avg_temp_month.png")

    # Question 4 - Fitting model for different values of `k`
    train_X, train_y, test_X, test_y = \
        split_train_test(israel_df["DayOfYear"], pd.Series(israel_df["Temp"]), train_proportion=0.75)
    lst_loss_val = []
    for k in range(1, 11):
        pf = PolynomialFitting(k)
        fitted = pf.fit(train_X.values.flatten(), train_y.values)
        lst_loss_val.append(round(fitted.loss(test_X.values.flatten(), test_y.values), 2))
    lst_loss_val = pd.DataFrame({'k': list(range(1, 11)), 'loss': lst_loss_val})
    print(lst_loss_val)
    # bar plot showing the test error recorded for each value of k.
    px.bar(lst_loss_val, x="k", y="loss", text="loss",
           title="Bar Plot or Test Error Per K", labels={"x": "k", "y": "test error"}).write_image("loss_per_k.png")

    # Question 5 - Evaluating fitted model on different countries
    k = 5
    pf = PolynomialFitting(k)
    fitted = pf.fit(israel_df["DayOfYear"].values, israel_df["Temp"].values)
    countries = ["Jordan", "South Africa", "The Netherlands"]
    loss_countries = []
    for country in countries:
        loss_country = round(fitted.loss(df[df["Country"] == country].DayOfYear, df[df["Country"] == country].Temp), 2)
        loss_countries.append(loss_country)
    df_country_loss = pd.DataFrame({"country": countries, "loss": loss_countries})

    px.bar(df_country_loss, x="country", y="loss", text="loss", color="country",
           title="Loss Per Country Trained Over Israel Model", labels={"x": "country", "y": "loss"}) \
        .write_image("test.other.countries.png")
