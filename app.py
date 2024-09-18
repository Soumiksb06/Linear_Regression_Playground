import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression, Perceptron
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_classification, make_blobs

# Function to plot 2D data
def plot_data(X, Y=None, labels=None, Y_pred=None, title="Data Plot", xlabel="X", ylabel="Y"):
    fig = go.Figure()

    if labels is not None:
        fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', marker=dict(color=labels, colorscale='Viridis')))
    else:
        fig.add_trace(go.Scatter(x=X.ravel(), y=Y.ravel(), mode='markers', name="Data Points"))

    if Y_pred is not None:
        fig.add_trace(go.Scatter(x=X.ravel(), y=Y_pred.ravel(), mode='lines', name="Predicted Line"))

    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel)
    st.plotly_chart(fig)

# Function to plot decision boundary for classification models
def plot_decision_boundary(model, X, Y, title):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig = px.scatter(x=X[:, 0], y=X[:, 1], color=Y.astype(str), title=title)
    fig.add_trace(go.Contour(x=np.arange(x_min, x_max, h), y=np.arange(y_min, y_max, h), z=Z, showscale=False, opacity=0.3, colorscale='RdBu', hoverinfo='skip'))
    st.plotly_chart(fig)

# Function to plot 3D data
def plot_3d_data(X, labels=None, title="3D Data Plot", xlabel="X", ylabel="Y", zlabel="Z"):
    fig = go.Figure()

    if labels is not None:
        fig.add_trace(go.Scatter3d(x=X[:, 0], y=X[:, 1], z=X[:, 2], mode='markers',
                                   marker=dict(color=labels, size=5, colorscale='Viridis', opacity=0.8)))
    else:
        fig.add_trace(go.Scatter3d(x=X[:, 0], y=X[:, 1], z=X[:, 2], mode='markers',
                                   marker=dict(size=5, opacity=0.8)))

    fig.update_layout(scene=dict(xaxis_title=xlabel, yaxis_title=ylabel, zaxis_title=zlabel), title=title)
    st.plotly_chart(fig)

# Title and introduction
st.title('Interactive Machine Learning Models with 3D Clustering Visualization')

# Sidebar for model selection
st.sidebar.header("Model and Data Controls")
model_type = st.sidebar.selectbox('Choose Model', ('Linear Regression', 'Polynomial Regression', 'SVM', 'K-Means Clustering (3D)', 'Perceptron'))

# Number of data points and noise
n_points = st.sidebar.slider('Number of Data Points', 50, 300, 100)
noise = st.sidebar.slider('Noise Level', 0, 5, 2)

# Generate dataset based on selected model
if model_type in ['Linear Regression', 'Polynomial Regression']:
    # For Polynomial Regression, use a polynomial data generator
    X = np.random.rand(n_points, 1) * 10
    degree = st.sidebar.slider('Degree of Polynomial', 2, 5, 1)
    
    # Generate polynomial data
    coefficients = np.random.randn(degree + 1)
    Y = np.polyval(coefficients, X.ravel()) + np.random.randn(n_points) * noise

    # Plot the polynomial data
    plot_data(X, Y, title="Generated Polynomial Data", xlabel="X", ylabel="Y")

    # Linear and Polynomial Regression models
    if model_type == 'Linear Regression':
        model = LinearRegression()
        model.fit(X, Y)
        Y_pred = model.predict(X)
        st.write(f"### Linear Regression Equation: Y = {model.coef_[0]:.2f} * X + {model.intercept_:.2f}")

    elif model_type == 'Polynomial Regression':
        poly_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        poly_model.fit(X, Y)
        Y_pred = poly_model.predict(X)
        st.write(f"### Polynomial Regression (Degree {degree})")

    # Plot predictions
    plot_data(X, Y, Y_pred=Y_pred, title=f"{model_type} Fit", xlabel="X", ylabel="Y")

elif model_type == 'SVM':
    # Generate classification data
    X, Y = make_classification(n_samples=n_points, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, n_classes=2, n_clusters_per_class=1, flip_y=noise / 10, random_state=42)
    plot_data(X, labels=Y, title="Generated Classification Data", xlabel="Feature 1", ylabel="Feature 2")

    # Train SVM model
    svm_model = SVC(kernel='linear')
    svm_model.fit(X, Y)

    # Plot decision boundary
    plot_decision_boundary(svm_model, X, Y, "SVM Decision Boundary")

elif model_type == 'K-Means Clustering (3D)':
    # Generate random data for 3D clustering
    X, _ = make_blobs(n_samples=n_points, centers=4, n_features=3, cluster_std=noise, random_state=42)
    plot_3d_data(X, title="Generated 3D Clustering Data", xlabel="Feature 1", ylabel="Feature 2", zlabel="Feature 3")

    # K-Means clustering
    k_clusters = st.sidebar.slider('Number of Clusters', 2, 5, 3)
    kmeans_model = KMeans(n_clusters=k_clusters)
    kmeans_model.fit(X)
    labels = kmeans_model.predict(X)

    # Plot 3D clustering results
    plot_3d_data(X, labels=labels, title=f"K-Means Clustering (k={k_clusters}) in 3D", xlabel="Feature 1", ylabel="Feature 2", zlabel="Feature 3")

elif model_type == 'Perceptron':
    # Generate linearly separable data for perceptron
    X, Y = make_classification(n_samples=n_points, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, n_classes=2, n_clusters_per_class=1, flip_y=noise / 10, random_state=42)
    plot_data(X, labels=Y, title="Generated Perceptron Data", xlabel="Feature 1", ylabel="Feature 2")

    # Train perceptron model
    perceptron_model = Perceptron()
    perceptron_model.fit(X, Y)
    st.write("### Perceptron Model Trained")

    # Plot decision boundary
    plot_decision_boundary(perceptron_model, X, Y, "Perceptron Decision Boundary")

# Sidebar for making predictions (for regression models)
if model_type in ['Linear Regression', 'Polynomial Regression']:
    new_X = st.sidebar.number_input('Enter a new value for X:', value=5.0)
    if model_type == 'Linear Regression':
        new_Y_pred = model.predict([[new_X]])[0]
    else:
        new_Y_pred = poly_model.predict([[new_X]])[0]
    st.sidebar.write(f"Predicted value for X = {new_X}: Y = {new_Y_pred:.2f}")
