import math

import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets as skl_datasets
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier, plot_tree
import seaborn as sns


def plot_linear(x_data, y_data):
    plt.scatter(x_data, y_data)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

def pre_process_linear(x, y):
    # sklearn requires a 2D array, so lets reshape our 1D arrays.
    x_data = np.array(x).reshape(-1, 1)
    y_data = np.array(y).reshape(-1, 1)

    return x_data, y_data

def fit_a_linear_model(x_data, y_data):
    # Define our estimator/model
    model = LinearRegression()

    # train our estimator/model using our data
    lin_regress = model.fit(x_data, y_data)

    # inspect the trained estimator/model parameters
    m = lin_regress.coef_
    c = lin_regress.intercept_
    print("linear coefs=", m, c)

    return lin_regress

def predict_linear_model(lin_regress, x_data, y_data):
    # predict some values using our trained estimator/model
    # (in this case we predict our input data!)
    linear_data = lin_regress.predict(x_data)

    # calculated a RMS error as a quality of fit metric
    error = math.sqrt(mean_squared_error(y_data, linear_data))
    print("linear error=", error)

    # return our trained model so that we can use it later
    return linear_data

def plot_linear_model(x_data, y_data, predicted_data):
    # visualise!
    # Don't call .show() here so that we can add extra stuff to the figure later
    plt.scatter(x_data, y_data, label="input")
    plt.plot(x_data, predicted_data, "-", label="fit")
    plt.plot(x_data, predicted_data, "rx", label="predictions")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()

def fit_predict_plot_linear(x, y):
    x_data, y_data = pre_process_linear(x, y)
    lin_regress = fit_a_linear_model(x_data, y_data)
    linear_data = predict_linear_model(lin_regress, x_data, y_data)
    plot_linear_model(x_data, y_data, linear_data)

    return lin_regress

def pre_process_poly(x, y):
    # sklearn requires a 2D array, so lets reshape our 1D arrays.
    x_data = np.array(x).reshape(-1, 1)
    y_data = np.array(y).reshape(-1, 1)

    # create a polynomial representation of our data
    poly_features = PolynomialFeatures(degree=2)
    x_poly = poly_features.fit_transform(x_data)

    return x_poly, x_data, y_data


def fit_poly_model(x_poly, y_data):
    # Define our estimator/model(s)
    poly_regress = LinearRegression()

    # define and train our model
    poly_regress.fit(x_poly, y_data)

    # inspect trained model parameters
    poly_m = poly_regress.coef_
    poly_c = poly_regress.intercept_
    print("poly_coefs", poly_m, poly_c)

    return poly_regress


def predict_poly_model(poly_regress, x_poly, y_data):
    # predict some values using our trained estimator/model
    # (in this case - our input data)
    poly_data = poly_regress.predict(x_poly)

    poly_error = math.sqrt(mean_squared_error(y_data, poly_data))
    print("poly error=", poly_error)

    return poly_data


def plot_poly_model(x_data, poly_data):
    # visualise!
    plt.plot(x_data, poly_data, label="poly fit")
    plt.legend()


def fit_predict_plot_poly(x, y):
    # Combine all of the steps
    x_poly, x_data, y_data = pre_process_poly(x, y)
    poly_regress = fit_poly_model(x_poly, y_data)
    poly_data = predict_poly_model(poly_regress, x_poly, y_data)
    plot_poly_model(x_data, poly_data)

    return poly_regress

def get_penguin_data():
    dataset = sns.load_dataset("penguins")
    dataset.dropna(inplace=True)
    return dataset
    
def get_penguin_classification_data():
    dataset = get_penguin_data()

    feature_names = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
    dataset.dropna(subset=feature_names, inplace=True)
    
    class_names = dataset['species'].unique()
    
    X = dataset[feature_names]
    y = dataset['species']

    return X, y

def view_decision_tree_classifier(clf):
    dataset = get_penguin_data()

    class_names = dataset['species'].unique()
    feature_names = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
    
    fig = plt.figure(figsize=(12, 10))
    plot_tree(clf, class_names=class_names, feature_names=feature_names, filled=True, ax=fig.gca())
    plt.show()

def plot_decision_tree_decision_boundaries(clf, X_train, y_train, feature1="bill_length_mm", feature2="body_mass_g"):
    d = DecisionBoundaryDisplay.from_estimator(clf, X_train[[feature1, feature2]])
    sns.scatterplot(X_train, x=feature1, y=feature2, hue=y_train, palette="husl")
    plt.show()

def get_cluster_data(num_clusters=4, cluster_std=0.75):
    data, cluster_id = skl_datasets.make_blobs(n_samples=400, cluster_std=cluster_std, centers=num_clusters, random_state=1)
    return data, cluster_id

def plot_clusters(data, labels=None, centers=None):
    tx = data[:, 0]
    ty = data[:, 1]
    fig = plt.figure(1, figsize=(4, 4))
    if labels is not None:
        plt.scatter(tx, ty, edgecolor='k', c=labels)
    else:
        plt.scatter(tx, ty, edgecolor='k', c=labels)
    if centers is not None:
        for cluster_x, cluster_y in centers:
            plt.scatter(cluster_x, cluster_y, s=150, c='white', edgecolor='k', linewidths=1.5, marker='X')
    plt.show()

def get_moons_data():
    data, cluster_id = skl_datasets.make_moons(n_samples=400, noise=0.1, random_state=1)
    return data, cluster_id

def view_random_forest_tree_classifiers(clf):
    fig, axes = plt.subplots(nrows=1, ncols=5 ,figsize=(12,6))
    dataset = get_penguin_data()

    class_names = dataset['species'].unique()
    feature_names = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
    
    # plot first 5 trees in forest
    for index in range(0, 5):
        plot_tree(clf.estimators_[index],
            class_names=class_names,
            feature_names=feature_names,
            filled=True,
            ax=axes[index])
    
        axes[index].set_title(f'Tree: {index}')
