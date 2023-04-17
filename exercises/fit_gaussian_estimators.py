from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"

# CSE: mika.li 322851593

# QUESTION 1
Q1_MU = 10
Q1_SD = 1
Q1_SAMPLES_NUM = 1000

# QUESTION 2
Q2_FIRST_SAMPLE_SIZE = 10
Q2_LAST_SAMPLE_SIZE = 1010
Q2_JUMP_RANGE = 10

# QUESTION 4
Q4_MU = [0, 0, 4, 0]
Q4_COV = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
Q4_SAMPLE_SIZE = 1000

# QUESTION 5
Q5_START_LINESPACE = -10
Q5_END_LINESPACE = 10
Q5_SIZE_LINESPACE = 200


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    samples = np.random.normal(loc=Q1_MU, scale=Q1_SD, size=Q1_SAMPLES_NUM)
    ug_q1 = UnivariateGaussian()
    fit_samples = ug_q1.fit(samples)
    print((fit_samples.mu_, fit_samples.var_))

    # Question 2 - Empirically showing sample mean is consistent
    lst_distance = []
    lst_sample_size = []
    for sub_samples_size in range(Q2_FIRST_SAMPLE_SIZE, Q2_LAST_SAMPLE_SIZE, Q2_JUMP_RANGE):
        lst_sample_size.append(sub_samples_size)
        fit_sub_samples = UnivariateGaussian().fit(samples[:sub_samples_size])
        lst_distance.append(abs(fit_sub_samples.mu_ - Q1_MU))
    go.Figure(go.Scatter(x=lst_sample_size, y=lst_distance, mode='markers'),
              layout=dict(title=r"Deference Between Mean Estimator To True Value As Function "
                                r"Of Size Of Samples",
                          xaxis_title="size of samples",
                          yaxis_title="r$|\hat\mu - \mu|$")).write_image("diff_mean_vs_size_of_samples.png")

    # Question 3 - Plotting Empirical PDF of fitted model
    samples_sort = np.sort(samples)
    pdf_samples = ug_q1.pdf(samples_sort)
    go.Figure(go.Scatter(x=samples_sort, y=pdf_samples, mode='markers'),
              layout=dict(title=r"Empirical PDF Function Under Fitted Model",
                          xaxis_title="X",
                          yaxis_title="r$N(\hat\mu, \hat\sigma)$")).write_image("empirical_pdf_fit.png")


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    samples = np.random.multivariate_normal(mean=Q4_MU, cov=Q4_COV, size=Q4_SAMPLE_SIZE)
    mg_q4 = MultivariateGaussian()
    fit_samples = mg_q4.fit(samples)
    print(fit_samples.mu_)
    print(fit_samples.cov_)

    # Question 5 - Likelihood evaluation
    max_likelihood = -10000
    max_f1 = 0
    max_f3 = 0
    f_values = np.linspace(Q5_START_LINESPACE, Q5_END_LINESPACE, Q5_SIZE_LINESPACE)
    log_likelihood_matrix = np.zeros((Q5_SIZE_LINESPACE, Q5_SIZE_LINESPACE))
    for i in range(Q5_SIZE_LINESPACE):
        for j in range(Q5_SIZE_LINESPACE):
            f1 = f_values[i]
            f3 = f_values[j]
            likelihood_val = MultivariateGaussian.log_likelihood(np.array([f1, 0, f3, 0]), Q4_COV, samples)
            log_likelihood_matrix[i, j] = likelihood_val
            # update to save the maximum likelihood value
            if (likelihood_val > max_likelihood):
                max_likelihood = likelihood_val
                max_f1 = f1
                max_f3 = f3
    go.Figure(go.Heatmap(x=f_values, y=f_values, z=log_likelihood_matrix),
              layout=dict(title="Log Likelihood Heatmap For Multivariant Gaussian",
                          xaxis_title=r"f3 values",
                          yaxis_title=r"f1 values")).write_image("log_likelihood_heatmap_mv.png")

    # Question 6 - Maximum likelihood
    print("The f1 and f3 values that achieve the maximal likelihood value are:",
          np.round(max_f1, 3), np.round(max_f3, 3))


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
