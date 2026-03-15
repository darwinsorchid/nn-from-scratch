from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd
import numpy as np

def generate_data():

    x, y = make_moons(n_samples=1000, noise=0.2, random_state=42)

    df = pd.DataFrame({
        'feature_0' : x[:, 0],
        'feature_1' : x[:, 1],
        'labels' : y
    })

    df.to_csv('data/datasets/moon.csv', index=False)

    return x, y



def plot_data(X, y, plot_title, file_name=None, XX=None, YY=None, preds=None):
    '''Plot decision boundary of the keras classifier model on the moons dataset feature space.
    :param X: Feature matrix
    :param y: Labels/Dependent variable vector
    :param plot_title: Plot title
    :param file_name: Name for saving plot
    :param XX: x-coordinates of feature space grid
    :param YY: y-coordinates of feature space grid
    :param preds: Model's predictions on test set
    '''
    sns.set_style("whitegrid")

    plt.figure(figsize=(16,12))

    axes = plt.gca()
    axes.set(xlabel="$X_1$", ylabel="$X_2$")
    plt.title(plot_title, fontsize=30)
    plt.subplots_adjust(left=0.20)
    plt.subplots_adjust(right=0.80)

    if (XX is not None and YY is not None and preds is not None):
        plt.contourf(XX, YY, preds.reshape(XX.shape), 25, alpha = 1, cmap='Spectral')
        plt.contour(XX, YY, preds.reshape(XX.shape), levels=[.5], cmap="Greys")
    
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), s=40, cmap='Spectral', edgecolors='black')

    if(file_name):
        plt.savefig(file_name, dpi=600)
        plt.close()


if __name__ == "__main__":

    X, y = generate_data()

    plot_data(X, y, "Dataset", file_name="moon_data.png")