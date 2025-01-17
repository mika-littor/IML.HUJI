# mika.li 322851593
from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        x_data, y_data = load_dataset(f"../datasets/{f}")

        # Fit Perceptron and record loss in each fit iteration
        losses = []
        Perceptron(callback=lambda fit, a, b: losses.append(fit._loss(x_data, y_data))).fit(x_data, y_data)

        # Plot figure of loss as function of fitting iteration
        dots = np.arange(len(losses)).tolist()
        fig = go.Figure(data=go.Scatter(x=dots, y=losses, mode="lines", marker=dict(color="blue")),
                  layout=go.Layout(
                      title=dict(text="Perceptron Training Error as " + n + " Dataset\n"),
                      xaxis=dict(title="\nFitting Iteration"),
                      yaxis=dict(title="Loss as Misclassification Error\n")))
        fig.update_layout(title_font_size=25, font_size=20)
        fig.write_image("Perceptron_err_iter.png")


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        data_x, data_y = load_dataset(f"../datasets/{f}")

        # Fit models and predict over training set
        gnb = GaussianNaiveBayes().fit(data_x, data_y)
        lda = LDA().fit(data_x, data_y)
        gnb_predict = gnb.predict(data_x)
        lda_predict = lda.predict(data_x)

        # Plot a figure with two subplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        accuracy_gnb = round(accuracy(data_y, gnb_predict) * 100, 2)
        accuracy_lda = round(accuracy(data_y, lda_predict) * 100, 2)
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=(
                                "\n Gaussian Prediction Accuracy=" + str(accuracy_gnb) + "%",
                                "\n LDA Accuracy=" + str(accuracy_lda) + "%"))

        # Add traces for data-points setting symbols and colors
        scatter = []
        for i, classifier in enumerate(["Gaussian Naive Bayes", "LDA"]):
            scatter.append(go.Scatter(
                x=data_x[:, 0],
                y=data_x[:, 1],
                mode='markers',
                marker=dict(
                    color=gnb_predict if classifier == "Gaussian Naive Bayes" else lda_predict,
                    symbol=class_symbols[data_y],
                    colorscale=class_colors(1)
                ),
                name=classifier
            ))
        fig.add_traces(scatter, rows=[1, 1], cols=[1, 2])

        # Add `X` dots specifying fitted Gaussians' means
        markers = []
        for classifier, model in [("Gaussian Naive Bayes", gnb), ("LDA", lda)]:
            scatter = go.Scatter(
                x=model.mu_[:, 0],
                y=model.mu_[:, 1],
                mode="markers",
                marker=dict(symbol="x", color="black", size=15),
                name=classifier
            )
            markers.append(scatter)
        fig.add_traces(markers, rows=[1, 1], cols=[1, 2])

        # Add ellipses depicting the covariances of the fitted Gaussians
        for i in range(3):
            gnb_ellipse = get_ellipse(gnb.mu_[i], np.diag(gnb.vars_[i]))
            lda_ellipse = get_ellipse(lda.mu_[i], lda.cov_)

            fig.add_trace(gnb_ellipse, row=1, col=1)
            fig.add_trace(lda_ellipse, row=1, col=2)

        # update the layout to change the size and color
        fig.update_yaxes(matches="x")
        fig.update_layout(title_text="<span style='color:blue'>Gaussian Naive Bayes Compared To LDA (data=" + f + ")\n",
                          width=700, height=300, showlegend=False)
        fig.write_image(f"gnb_lda_compare.png")


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
