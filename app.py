import streamlit as st
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Function to plot 2D data
def plot_data(X, Y, Y_pred=None, title="Data Plot", xlabel="X", ylabel="Y"):
    fig = px.scatter(x=X.ravel(), y=Y.ravel(), labels={'x': xlabel, 'y': ylabel}, title=title)
    
    if Y_pred is not None:
        fig.add_traces(px.line(x=X.ravel(), y=Y_pred.ravel(), labels={'x': xlabel, 'y': ylabel}).data)
        
    st.plotly_chart(fig)

# Sidebar for model selection
model_type = st.sidebar.selectbox('Choose Model', ('Linear Regression', 'Polynomial Regression'))

# Number of data points and noise
n_points = st.sidebar.slider('Number of Data Points', 50, 300, 100)
noise = st.sidebar.slider('Noise Level', 0.0, 5.0, 2.0)

# Generate random linear data
X = np.random.rand(n_points, 1) * 10  # Rescale X to [0, 10]
Y = 2 * X + 5 + np.random.randn(n_points, 1) * noise  # Linear function with noise

# Plot the data
plot_data(X, Y, title="Generated Data", xlabel="X", ylabel="Y")

# Linear and Polynomial Regression models
if model_type == 'Linear Regression':
    model = LinearRegression()
    model.fit(X, Y)
    Y_pred = model.predict(X)
    st.write(f"### Linear Regression Equation: Y = {model.coef_[0][0]:.2f} * X + {model.intercept_[0]:.2f}")
    plot_data(X, Y, Y_pred=Y_pred, title="Linear Regression Fit", xlabel="X", ylabel="Y")

elif model_type == 'Polynomial Regression':
    degree = st.sidebar.slider('Degree of Polynomial', 2, 10, 2)
    poly_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    poly_model.fit(X, Y)
    Y_pred = poly_model.predict(X)
    st.write(f"### Polynomial Regression (Degree {degree})")
    plot_data(X, Y, Y_pred=Y_pred, title=f"Polynomial Regression (Degree {degree})", xlabel="X", ylabel="Y")
