import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def process_data(file_name):

    # Read dataset
    data = pd.read_csv(file_name)

    features = ['feature_0', "feature_1"]
    target = ['labels']

    # Features matrix
    x = data[features].values

    # Dependent variable vector
    y = data[target].values

    # Split data into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, stratify=y, random_state=42)

    # Apply feature scaling - although the moons dataset's features are roughly on the same scale
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    return x_train, x_test, y_train, y_test
