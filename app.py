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

# Function to plot data
def plot_data(X, Y=None, labels=None, title="Data Plot", xlabel="X", ylabel="Y"):
    fig = go.Figure()
    
    if labels is not None:
        fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', marker=dict(color=labels, colorscale='Viridis')))
    else:
        fig.add_trace(go.Scatter(x=X, y=Y, mode='markers', name="Data Points"))

    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel)
    st.plotly_chart(fig)

# Plot decision boundary for classification models
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

# Title and introduction
st.title('Interactive Machine Learning Models')
st.write("Explore different machine learning models interactively. Adjust the parameters and visualize the results.")

# Sidebar for model selection
st.sidebar.header("Model and Data Controls")
model_type = st.sidebar.selectbox('Choose Model', ('Linear Regression', 'Polynomial Regression', 'SVM', 'K-Means Clustering', 'Perceptron'))

# Number of data points and noise
n_points = st.sidebar.slider('Number of Data Points', 50, 300, 100)
noise = st.sidebar.slider('Noise Level', 0, 5, 2)

# Generate dataset based on selected model
if model_type in ['Linear Regression', 'Polynomial Regression']:
    # Generate random linear data
    X = np.random.rand(n_points, 1) * 10
    Y = 2 * X + 5 + np.random.randn(n_points, 1) * noise

    # Plot the data
    plot_data(X, Y, title="Generated Data", xlabel="X", ylabel="Y")

    # Linear and Polynomial Regression models
    if model_type == 'Linear Regression':
        model = LinearRegression()
        model.fit(X, Y)
        Y_pred = model.predict(X)
        st.write(f"### Linear Regression Equation: Y = {model.coef_[0][0]:.2f} * X + {model.intercept_[0]:.2f}")

    elif model_type == 'Polynomial Regression':
        degree = st.sidebar.slider('Degree of Polynomial', 2, 10, 2)
        poly_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        poly_model.fit(X, Y)
        Y_pred = poly_model.predict(X)
        st.write(f"### Polynomial Regression (Degree {degree})")

    # Plot predictions
    fig = px.scatter(x=X.ravel(), y=Y.ravel(), title=f"{model_type} Fit", labels={'x': 'X', 'y': 'Y'})
    fig.add_traces(go.Scatter(x=X.ravel(), y=Y_pred.ravel(), mode='lines', name=f'{model_type} Line'))
    st.plotly_chart(fig)

elif model_type == 'SVM':
    # Generate classification data
    X, Y = make_classification(n_samples=n_points, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, n_classes=2, n_clusters_per_class=1, flip_y=noise / 10, random_state=42)
    plot_data(X, labels=Y, title="Generated Classification Data", xlabel="Feature 1", ylabel="Feature 2")

    # Train SVM model
    svm_model = SVC(kernel='linear')
    svm_model.fit(X, Y)

    # Plot decision boundary
    plot_decision_boundary(svm_model, X, Y, "SVM Decision Boundary")

elif model_type == 'K-Means Clustering':
    # Generate random data for clustering
    X, _ = make_blobs(n_samples=n_points, centers=3, cluster_std=noise, random_state=42)
    plot_data(X, title="Generated Clustering Data", xlabel="Feature 1", ylabel="Feature 2")

    # K-Means clustering
    k_clusters = st.sidebar.slider('Number of Clusters', 2, 5, 3)
    kmeans_model = KMeans(n_clusters=k_clusters)
    kmeans_model.fit(X)
    labels = kmeans_model.predict(X)

    # Plot clustering results
    plot_data(X, labels=labels, title=f"K-Means Clustering (k={k_clusters})", xlabel="Feature 1", ylabel="Feature 2")

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
        new_Y_pred = model.predict([[new_X]])[0][0]
    else:
        new_Y_pred = poly_model.predict([[new_X]])[0][0]
    st.sidebar.write(f"Predicted value for X = {new_X}: Y = {new_Y_pred:.2f}")
