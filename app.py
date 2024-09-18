import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Title and Introduction
st.title('Interactive Regression: Linear vs Polynomial')
st.write("""
Welcome to the interactive regression model demo! You can create data points, select different models, and visualize 
how Linear and Polynomial Regression fit lines to the data in real-time.
""")

# Section to create data points
st.sidebar.header("Generate Data Points")

# User input for data points and noise level
n_points = st.sidebar.slider('Number of Data Points', min_value=5, max_value=100, value=50)
noise = st.sidebar.slider('Noise Level', min_value=0, max_value=10, value=3)
X = np.random.rand(n_points, 1) * 10  # Random X values
Y = 2 * X + 5 + np.random.randn(n_points, 1) * noise  # Linear relationship with noise

# Plot the generated data points
st.write("### Scatter Plot of Generated Data")
fig, ax = plt.subplots()
ax.scatter(X, Y, color='blue', label='Data Points')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Scatter Plot of Data")
st.pyplot(fig)

# User selects model type (Linear or Polynomial)
model_type = st.sidebar.selectbox('Select Model Type', ('Linear Regression', 'Polynomial Regression'))

# Linear Regression Model
if model_type == 'Linear Regression':
    model = LinearRegression()
    model.fit(X, Y)
    Y_pred = model.predict(X)

    # Display the linear regression equation
    st.write(f"### Linear Regression Equation: Y = {model.coef_[0][0]:.2f} * X + {model.intercept_[0]:.2f}")

# Polynomial Regression Model
elif model_type == 'Polynomial Regression':
    degree = st.sidebar.slider('Degree of Polynomial', min_value=2, max_value=10, value=2)
    
    poly_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    poly_model.fit(X, Y)
    Y_pred = poly_model.predict(X)

    # Display information about polynomial regression
    st.write(f"### Polynomial Regression (Degree {degree})")
    st.write("Polynomial regression allows the model to fit more complex curves.")

# Plot the regression line along with data points
st.write(f"### {model_type} Fit to Data")
fig, ax = plt.subplots()
ax.scatter(X, Y, color='blue', label='Data Points')
ax.plot(X, Y_pred, color='red', label=f'{model_type} Line')
plt.xlabel("X")
plt.ylabel("Y")
plt.title(f"{model_type} Fit")
ax.legend()
st.pyplot(fig)

# User input for prediction of new values
st.sidebar.header("Make Predictions")
new_X = st.sidebar.number_input('Enter a new value for X:', min_value=0.0, value=5.0, step=0.1)

if model_type == 'Linear Regression':
    new_Y_pred = model.predict([[new_X]])[0][0]
else:
    new_Y_pred = poly_model.predict([[new_X]])[0][0]

# Display the prediction
st.write(f"### Predicted Value for X = {new_X}: Y = {new_Y_pred:.2f}")
