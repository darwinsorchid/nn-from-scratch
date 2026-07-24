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


if __name__ == "__main__":

    X, y = generate_data()

