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
