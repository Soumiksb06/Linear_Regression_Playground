import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
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
        # Sort X and Y_pred for smooth line plot
        sorted_indices = np.argsort(X.ravel())
        X_sorted = X.ravel()[sorted_indices]
        Y_pred_sorted = Y_pred.ravel()[sorted_indices]
        fig.add_trace(go.Scatter(x=X_sorted, y=Y_pred_sorted, mode='lines', name="Predicted Line"))

    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel)
    st.plotly_chart(fig)

# Load a pre-trained image classification model
@st.cache_resource
def load_model():
    model = MobileNetV2(weights='imagenet')
    return model

# Function to preprocess and classify an image
def classify_image(image, model):
    image = image.resize((224, 224))  # Resize to the model input size
    image = img_to_array(image)  # Convert image to array
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = preprocess_input(image)  # Preprocess for MobileNetV2
    preds = model.predict(image)
    return decode_predictions(preds, top=3)[0]  # Get top 3 predictions

# Title and introduction
st.title('Interactive Machine Learning Models with 3D Clustering and Image Classification')

# Sidebar for model selection
st.sidebar.header("Model and Data Controls")
model_type = st.sidebar.selectbox('Choose Model', ('Linear Regression', 'Polynomial Regression', 'SVM', 'K-Means Clustering (3D)', 'Perceptron', 'Image Classification'))

# Number of data points and noise (for non-image models)
if model_type != 'Image Classification':
    n_points = st.sidebar.slider('Number of Data Points', 50, 300, 100)
    noise = st.sidebar.slider('Noise Level', 0, 5, 2)

# Generate dataset based on selected model
if model_type in ['Linear Regression', 'Polynomial Regression']:
    X = np.random.rand(n_points, 1) * 10
    degree = st.sidebar.slider('Degree of Polynomial', 2, 5, 1)

    coefficients = np.random.randn(degree + 1)
    Y = np.polyval(coefficients, X.ravel()) + np.random.randn(n_points) * noise

    plot_data(X, Y, title="Generated Polynomial Data", xlabel="X", ylabel="Y")

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

    plot_data(X, Y, Y_pred=Y_pred, title=f"{model_type} Fit", xlabel="X", ylabel="Y")

elif model_type == 'SVM':
    X, Y = make_classification(n_samples=n_points, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, n_classes=2, n_clusters_per_class=1, flip_y=noise / 10, random_state=42)
    plot_data(X, labels=Y, title="Generated Classification Data", xlabel="Feature 1", ylabel="Feature 2")

    svm_model = SVC(kernel='linear')
    svm_model.fit(X, Y)

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

    plot_decision_boundary(svm_model, X, Y, "SVM Decision Boundary")

elif model_type == 'K-Means Clustering (3D)':
    X, _ = make_blobs(n_samples=n_points, centers=4, n_features=3, cluster_std=noise, random_state=42)
    plot_3d_data(X, title="Generated 3D Clustering Data", xlabel="Feature 1", ylabel="Feature 2", zlabel="Feature 3")

    k_clusters = st.sidebar.slider('Number of Clusters', 2, 5, 3)
    kmeans_model = KMeans(n_clusters=k_clusters)
    kmeans_model.fit(X)
    labels = kmeans_model.predict(X)

    def plot_3d_data(X, labels=None, title="3D Data Plot", xlabel="X", ylabel="Y", zlabel="Z"):
        fig = go.Figure()

        if labels is not None:
            fig.add_trace(go.Scatter3d(x=X[:, 0], y=X[:, 1], z=X[:, 2], mode='markers', marker=dict(color=labels, size=5, colorscale='Viridis', opacity=0.8)))
        else:
            fig.add_trace(go.Scatter3d(x=X[:, 0], y=X[:, 1], z=X[:, 2], mode='markers', marker=dict(size=5, opacity=0.8)))

        fig.update_layout(scene=dict(xaxis_title=xlabel, yaxis_title=ylabel, zaxis_title=zlabel), title=title)
        st.plotly_chart(fig)

    plot_3d_data(X, labels=labels, title=f"K-Means Clustering (k={k_clusters}) in 3D", xlabel="Feature 1", ylabel="Feature 2", zlabel="Feature 3")

elif model_type == 'Perceptron':
    X, Y = make_classification(n_samples=n_points, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, n_classes=2, n_clusters_per_class=1, flip_y=noise / 10, random_state=42)
    plot_data(X, labels=Y, title="Generated Perceptron Data", xlabel="Feature 1", ylabel="Feature 2")

    perceptron_model = Perceptron()
    perceptron_model.fit(X, Y)
    st.write("### Perceptron Model Trained")
    plot_decision_boundary(perceptron_model, X, Y, "Perceptron Decision Boundary")

elif model_type == 'Image Classification':
    st.sidebar.write("### Upload an Image to Classify")
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Load the pre-trained model and classify the image
        model = load_model()
        st.write("Classifying...")
        preds = classify_image(image, model)

        # Show the predictions
        st.write("### Top Predictions:")
        for i, (imagenet_id, label, score) in enumerate(preds):
            st.write(f"{i + 1}. **{label}**: {score * 100:.2f}%")

