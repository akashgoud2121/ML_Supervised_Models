import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (r2_score, mean_squared_error, accuracy_score, 
                           classification_report, confusion_matrix)
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Model categories and parameters
MODEL_CATEGORIES = {
    "Regression": ["Linear Regression"],
    "Classification": ["Logistic Regression", "Decision Tree", "Random Forest", 
                      "Gradient Boosting", "XGBoost", "CatBoost", "LightGBM",
                      "KNN", "SVM"]
}

MODEL_PARAMS = {
    "Linear Regression": {
        "n_samples": {
            "desc": "Number of samples in dataset",
            "suggested": 100,
            "min": 10,
            "max": 1000,
            "warning": "Too few samples may lead to unstable results"
        },
        "noise": {
            "desc": "Amount of noise in data",
            "suggested": 0.5,
            "min": 0.0,
            "max": 2.0,
            "warning": "High noise makes patterns harder to detect"
        }
    },
    "Logistic Regression": {
        "n_samples": {
            "desc": "Number of samples",
            "suggested": 100,
            "min": 10,
            "max": 1000
        },
        "noise": {
            "desc": "Classification noise",
            "suggested": 0.1,
            "min": 0.0,
            "max": 1.0
        },
        "C": {
            "desc": "Regularization strength",
            "suggested": 1.0,
            "min": 0.01,
            "max": 10.0,
            "warning": "Lower values = stronger regularization"
        },
        "max_iter": {
            "desc": "Maximum iterations",
            "suggested": 100,
            "min": 100,
            "max": 1000,
            "warning": "Increase if not converging"
        }
    },
    "Decision Tree": {
        "criterion": {
            "desc": "Split criterion",
            "suggested": "gini",
            "options": ["gini", "entropy"],
            "warning": "Both methods usually give similar results"
        },
        "max_depth": {
            "desc": "Maximum depth of tree",
            "suggested": 3,
            "min": 1,
            "max": 10,
            "warning": "Deeper trees may overfit"
        },
        "min_samples_split": {
            "desc": "Minimum samples to split",
            "suggested": 2,
            "min": 2,
            "max": 20,
            "warning": "Larger values prevent overfitting"
        },
        "min_samples_leaf": {
            "desc": "Minimum samples in leaf",
            "suggested": 1,
            "min": 1,
            "max": 10,
            "warning": "Affects tree balance"
        }
    },
    "Random Forest": {
        "n_estimators": {
            "desc": "Number of trees",
            "suggested": 100,
            "min": 10,
            "max": 500,
            "warning": "More trees increase computation time"
        },
        "max_depth": {
            "desc": "Maximum depth of trees",
            "suggested": 5,
            "min": 1,
            "max": 20,
            "warning": "Deeper trees may overfit"
        },
        "bootstrap": {
            "desc": "Bootstrap samples",
            "suggested": True,
            "options": [True, False],
            "warning": "Controls sample diversity"
        },
        "max_features": {
            "desc": "Features to consider per split",
            "suggested": "sqrt",
            "options": ["sqrt", "log2", "auto"],
            "warning": "Affects randomness in feature selection"
        }
    },
    "Gradient Boosting": {
        "n_estimators": {
            "desc": "Number of boosting stages",
            "suggested": 100,
            "min": 50,
            "max": 500,
            "warning": "More trees increase training time"
        },
        "learning_rate": {
            "desc": "Learning rate",
            "suggested": 0.1,
            "min": 0.01,
            "max": 1.0,
            "warning": "Lower values need more trees"
        },
        "max_depth": {
            "desc": "Maximum tree depth",
            "suggested": 3,
            "min": 1,
            "max": 10,
            "warning": "Deeper trees may overfit"
        }
    },
    "XGBoost": {
        "n_estimators": {
            "desc": "Number of trees",
            "suggested": 100,
            "min": 50,
            "max": 500,
            "warning": "More trees increase training time"
        },
        "learning_rate": {
            "desc": "Learning rate",
            "suggested": 0.1,
            "min": 0.01,
            "max": 1.0,
            "warning": "Controls step size"
        },
        "max_depth": {
            "desc": "Maximum tree depth",
            "suggested": 6,
            "min": 1,
            "max": 15,
            "warning": "Controls model complexity"
        }
    },
    "CatBoost": {
        "iterations": {
            "desc": "Number of trees",
            "suggested": 100,
            "min": 50,
            "max": 500,
            "warning": "More iterations increase training time"
        },
        "learning_rate": {
            "desc": "Learning rate",
            "suggested": 0.1,
            "min": 0.01,
            "max": 1.0,
            "warning": "Affects convergence speed"
        },
        "depth": {
            "desc": "Tree depth",
            "suggested": 6,
            "min": 1,
            "max": 10,
            "warning": "Controls model complexity"
        }
    },
    "LightGBM": {
        "n_estimators": {
            "desc": "Number of trees",
            "suggested": 100,
            "min": 50,
            "max": 500,
            "warning": "More trees increase training time"
        },
        "learning_rate": {
            "desc": "Learning rate",
            "suggested": 0.1,
            "min": 0.01,
            "max": 1.0,
            "warning": "Controls training speed"
        },
        "num_leaves": {
            "desc": "Number of leaves",
            "suggested": 31,
            "min": 2,
            "max": 128,
            "warning": "Controls model complexity"
        }
    },
    "KNN": {
        "n_neighbors": {
            "desc": "Number of neighbors",
            "suggested": 5,
            "min": 1,
            "max": 30,
            "warning": "Too few = noisy, too many = underfit"
        },
        "weights": {
            "desc": "Weight function",
            "suggested": "uniform",
            "options": ["uniform", "distance"],
            "warning": "Use distance for varying density"
        },
        "p": {
            "desc": "Power parameter for Minkowski metric",
            "suggested": 2,
            "min": 1,
            "max": 2,
            "warning": "1=Manhattan, 2=Euclidean"
        }
    },
    "SVM": {
        "C": {
            "desc": "Regularization strength",
            "suggested": 1.0,
            "min": 0.01,
            "max": 10.0,
            "warning": "Lower values = stronger regularization"
        },
        "kernel": {
            "desc": "Kernel type",
            "suggested": "rbf",
            "options": ["rbf", "linear", "poly"],
            "warning": "RBF works well for most cases"
        },
        "gamma": {
            "desc": "Kernel coefficient",
            "suggested": "scale",
            "options": ["scale", "auto"],
            "warning": "Controls decision boundary complexity"
        }
    }
}

def plot_sigmoid_curve(model, X, y):
    """Plot sigmoid curve and decision function"""
    decision_function = model.decision_function(X)
    probabilities = model.predict_proba(X)[:, 1]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    
    sorted_idx = np.argsort(decision_function)
    decision_function = decision_function[sorted_idx]
    probabilities = probabilities[sorted_idx]
    y_sorted = y[sorted_idx]
    
    ax1.plot(range(len(decision_function)), decision_function, 
             'b-', label='Decision Function', alpha=0.5)
    ax1.set_xlabel('Sorted Sample Index')
    ax1.set_ylabel('Decision Function', color='b')
    
    ax2.plot(range(len(probabilities)), probabilities, 
             'r-', label='Probability', linewidth=2)
    ax2.set_ylabel('Probability', color='r')
    
    scatter = ax2.scatter(range(len(y_sorted)), y_sorted, 
                         c=y_sorted, cmap='coolwarm', 
                         alpha=0.4, label='Actual Classes')
    
    ax1.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='r')
    plt.title('Sigmoid Curve and Decision Function')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    return fig

@st.cache_data
def generate_linear_data(n_samples, noise):
    """Generate synthetic data for linear regression"""
    X = np.linspace(0, 10, n_samples).reshape(-1, 1)
    y = 2 * X + 1 + noise * np.random.randn(n_samples, 1)
    return X, y

# Add this function to plot actual vs predicted values for regression
def plot_actual_vs_predicted_regression(y_true, y_pred, title="Actual vs Predicted Values"):
    """Create scatter plot of actual vs predicted values for regression"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot scatter points
    ax.scatter(y_true, y_pred, alpha=0.5)
    
    # Plot perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title(title)
    ax.legend()
    
    # Add R² score
    r2 = r2_score(y_true, y_pred)
    ax.text(0.05, 0.95, f'R² = {r2:.3f}', 
            transform=ax.transAxes, 
            bbox=dict(facecolor='white', alpha=0.8))
    
    return fig

# Add this function to plot actual vs predicted values for classification
def plot_actual_vs_predicted_classification(y_true, y_pred_proba, title="Prediction Probabilities by Class"):
    """Create visualization of predicted probabilities for each class"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort by prediction probability
    if y_pred_proba.shape[1] == 2:  # Binary classification
        prob_class_1 = y_pred_proba[:, 1]
        sorted_indices = np.argsort(prob_class_1)
        
        ax.scatter(range(len(y_true)), 
                  prob_class_1[sorted_indices],
                  c=y_true[sorted_indices],
                  cmap='coolwarm',
                  alpha=0.6)
        
        ax.set_xlabel("Samples (sorted by prediction probability)")
        ax.set_ylabel("Predicted Probability (Class 1)")
        ax.set_title(title)
        
        # Add threshold line
        ax.axhline(y=0.5, color='r', linestyle='--', label='Decision Threshold')
        ax.legend()
        
    else:  # Multiclass classification
        sns.heatmap(y_pred_proba.T, 
                   cmap='YlOrRd',
                   ax=ax)
        ax.set_xlabel("Samples")
        ax.set_ylabel("Classes")
        ax.set_title(title)
    
    return fig

# Main app structure
def main():
    st.title("Machine Learning Models Playground")
    
    # Add reset button in sidebar
    if st.sidebar.button("Reset App"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.experimental_rerun()
    
    category = st.sidebar.selectbox("Select Category", list(MODEL_CATEGORIES.keys()))
    model_type = st.sidebar.selectbox("Select Model", MODEL_CATEGORIES[category])
    
    # Get data parameters
    n_samples = st.sidebar.number_input("Number of samples", 100, 1000, 500)
    noise = st.sidebar.number_input("Noise", 0.0, 1.0, 0.1)
    
    # Generate data only once and store in session state
    if "data_generated" not in st.session_state or st.session_state.get("category") != category:
        if category == "Regression":
            X, y = generate_linear_data(n_samples, noise)
        else:  # Classification
            X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
            # Scale features for better model performance
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Store in session state
        st.session_state.X = X
        st.session_state.y = y
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        st.session_state.data_generated = True
        st.session_state.category = category  # Store current category
    else:
        # Retrieve from session state
        X = st.session_state.X
        y = st.session_state.y
        X_train = st.session_state.X_train
        X_test = st.session_state.X_test
        y_train = st.session_state.y_train
        y_test = st.session_state.y_test
    
    # Visualize the dataset
    if category == "Classification":
        st.subheader("Dataset Visualization")
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='rainbow')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_title('Classification Dataset')
        legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
        ax.add_artist(legend1)
        st.pyplot(fig)
        plt.close(fig)
    
    if category == "Regression":
        show_regression_interface(X, y, X_train, X_test, y_train, y_test)
    else:
        model_params = MODEL_PARAMS.get(model_type, {})
        show_classification_interface(X, y, X_train, X_test, y_train, y_test, 
                                   model_type, model_params)

def show_regression_section(model_type):
    """Handle regression models (Linear Regression)"""
    st.sidebar.markdown(f"# {model_type} Parameters")
    params = MODEL_PARAMS[model_type]
    
    # Parameter inputs
    with st.sidebar:
        n_samples = st.number_input("Number of samples", 
                                  params["n_samples"]["min"],
                                  params["n_samples"]["max"],
                                  params["n_samples"]["suggested"])
        noise = st.number_input("Noise", 
                              params["noise"]["min"],
                              params["noise"]["max"],
                              params["noise"]["suggested"])
    
    # Generate data
    X, y = generate_linear_data(n_samples, noise)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Show regression interface
    show_regression_interface(X, y, X_train, X_test, y_train, y_test)

def show_classification_section(model_type):
    """Handle classification models"""
    st.sidebar.markdown(f"# {model_type} Parameters")
    params = MODEL_PARAMS[model_type]
    
    # Parameter inputs
    with st.sidebar:
        n_samples = st.number_input("Number of samples", 
                                  params["n_samples"]["min"],
                                  params["n_samples"]["max"],
                                  params["n_samples"]["suggested"])
        noise = st.number_input("Noise", 
                              params["noise"]["min"],
                              params["noise"]["max"],
                              params["noise"]["suggested"])
        C = st.number_input("Regularization (C)",
                           params["C"]["min"],
                           params["C"]["max"],
                           params["C"]["suggested"])
        max_iter = st.number_input("Max iterations",
                                 params["max_iter"]["min"],
                                 params["max_iter"]["max"],
                                 params["max_iter"]["suggested"])
    
    # Generate data using make_moons instead
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Show classification interface
    show_classification_interface(X, y, X_train, X_test, y_train, y_test, model_type, params)

def show_regression_interface(X, y, X_train, X_test, y_train, y_test):
    """Display regression model interface and plots"""
    # Create base plot container
    plot_container = st.empty()
    
    # Draw initial plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X_train, y_train, color='blue', alpha=0.5, label='Training Data')
    ax.scatter(X_test, y_test, color='green', alpha=0.5, label='Test Data')
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.legend()
    plot_container.pyplot(fig)
    plt.close(fig)

    # Train model button
    if st.sidebar.button("Train Model"):
        with st.spinner("Training model..."):
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Store model and training state
            st.session_state.model = model
            st.session_state.trained = True
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Update plot with regression line
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(X_train, y_train, color='blue', alpha=0.5, label='Training Data')
            ax.scatter(X_test, y_test, color='green', alpha=0.5, label='Test Data')
            
            X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
            y_line = model.predict(X_line)
            ax.plot(X_line, y_line, 'r-', label='Regression Line')
            
            ax.set_xlabel('X')
            ax.set_ylabel('y')
            ax.legend()
            plot_container.pyplot(fig)
            plt.close(fig)
            
            # Display metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Training R² Score", f"{r2_score(y_train, y_pred_train):.3f}")
                st.metric("Training MSE", f"{mean_squared_error(y_train, y_pred_train):.3f}")
            with col2:
                st.metric("Test R² Score", f"{r2_score(y_test, y_pred_test):.3f}")
                st.metric("Test MSE", f"{mean_squared_error(y_test, y_pred_test):.3f}")
            
            # Show model parameters
            st.sidebar.markdown("## Model Parameters")
            st.sidebar.info(f"Slope (w): {model.coef_[0][0]:.3f}")
            st.sidebar.info(f"Intercept (b): {model.intercept_[0]:.3f}")

    # Show prediction interface if model is trained
    if st.session_state.get("trained", False):
        show_prediction_interface(st.session_state.model, "regression")

def show_classification_interface(X, y, X_train, X_test, y_train, y_test, model_type, params):
    """Display classification model interface and plots"""
    
    if model_type == "Logistic Regression":
        handle_logistic_regression(X, y, X_train, X_test, y_train, y_test, params)
    
    elif model_type in ["Gradient Boosting", "XGBoost", "CatBoost", "LightGBM"]:
        handle_boosting_model(X, y, X_train, X_test, y_train, y_test, model_type, params)
    
    elif model_type == "Random Forest":
        handle_random_forest(X, y, X_train, X_test, y_train, y_test, params)
    
    elif model_type == "Decision Tree":
        handle_decision_tree(X, y, X_train, X_test, y_train, y_test, params)
    
    elif model_type == "KNN":
        handle_knn(X, y, X_train, X_test, y_train, y_test, params)
    
    elif model_type == "SVM":
        handle_svm(X, y, X_train, X_test, y_train, y_test, params)

def handle_logistic_regression(X, y, X_train, X_test, y_train, y_test, params):
    """Handle Logistic Regression specific interface and training"""
    C = st.sidebar.number_input(
        "Regularization strength (C)",
        min_value=params["C"]["min"],
        max_value=params["C"]["max"],
        value=params["C"]["suggested"],
        help=params["C"]["warning"],
        key="logistic_regression_C"
    )
    
    max_iter = st.sidebar.number_input(
        "Maximum iterations",
        min_value=params["max_iter"]["min"],
        max_value=params["max_iter"]["max"],
        value=params["max_iter"]["suggested"],
        help=params["max_iter"]["warning"],
        key="logistic_regression_max_iter"
    )

    if st.sidebar.button("Train Model", key="logistic_regression_train"):
        with st.spinner("Training Logistic Regression..."):
            model = LogisticRegression(C=C, max_iter=max_iter, random_state=42)
            model.fit(X_train, y_train)
            st.session_state.model = model
            
            y_pred = model.predict(X_test)
            
            # Plot decision boundary and ROC curve
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Decision boundary plot
            XX, YY = np.meshgrid(np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, 100),
                               np.linspace(X[:, 1].min()-1, X[:, 1].max()+1, 100))
            Z = model.predict(np.c_[XX.ravel(), YY.ravel()])
            Z = Z.reshape(XX.shape)
            
            ax1.contourf(XX, YY, Z, alpha=0.4, cmap='rainbow')
            ax1.scatter(X[:, 0], X[:, 1], c=y, cmap='rainbow')
            ax1.set_title("Decision Boundary")
            
            # ROC curve
            from sklearn.metrics import roc_curve, auc
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            
            ax2.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
            ax2.plot([0, 1], [0, 1], 'k--')
            ax2.set_xlabel('False Positive Rate')
            ax2.set_ylabel('True Positive Rate')
            ax2.set_title('Receiver Operating Characteristic (ROC) Curve')
            ax2.legend(loc="lower right")
            
            st.pyplot(fig)
            
            # Display metrics
            st.subheader("Model Performance")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Training Accuracy", 
                         f"{accuracy_score(y_train, model.predict(X_train)):.3f}")
            with col2:
                st.metric("Test Accuracy", 
                         f"{accuracy_score(y_test, y_pred):.3f}")
            
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))

            # Remove ax parameter from the function call
            show_classification_predictions(model, X, y, fig)

def handle_boosting_model(X, y, X_train, X_test, y_train, y_test, model_type, params):
    """Handle boosting models (GradientBoosting, XGBoost, CatBoost, LightGBM)"""
    param_name = "iterations" if model_type == "CatBoost" else "n_estimators"
    
    n_estimators = st.sidebar.number_input(
        "Number of trees",
        min_value=params[param_name]["min"],
        max_value=params[param_name]["max"],
        value=params[param_name]["suggested"],
        help=params[param_name]["warning"],
        key=f"{model_type}_trees"
    )
    
    learning_rate = st.sidebar.number_input(
        "Learning rate",
        min_value=params["learning_rate"]["min"],
        max_value=params["learning_rate"]["max"],
        value=params["learning_rate"]["suggested"],
        help=params["learning_rate"]["warning"],
        key=f"{model_type}_lr"
    )

    # Model-specific depth parameter
    if model_type == "LightGBM":
        depth_param = st.sidebar.number_input(
            "Number of leaves",
            min_value=params["num_leaves"]["min"],
            max_value=params["num_leaves"]["max"],
            value=params["num_leaves"]["suggested"],
            help=params["num_leaves"]["warning"],
            key="lightgbm_leaves"
        )
    elif model_type == "CatBoost":
        depth_param = st.sidebar.number_input(
            "Depth",
            min_value=params["depth"]["min"],
            max_value=params["depth"]["max"],
            value=params["depth"]["suggested"],
            help=params["depth"]["warning"],
            key="catboost_depth"
        )
    else:  # Gradient Boosting and XGBoost
        depth_param = st.sidebar.number_input(
            "Max depth",
            min_value=params["max_depth"]["min"],
            max_value=params["max_depth"]["max"],
            value=params["max_depth"]["suggested"],
            help=params["max_depth"]["warning"],
            key=f"{model_type}_depth"
        )

    if st.sidebar.button("Train Model", key=f"{model_type}_train"):
        with st.spinner(f"Training {model_type}..."):
            model = create_boosting_model(model_type, n_estimators, learning_rate, depth_param)
            model.fit(X_train, y_train)
            st.session_state.model = model
            
            y_pred = model.predict(X_test)
            
            # Plot decision boundary and feature importance
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Decision boundary plot
            XX, YY = np.meshgrid(np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, 100),
                               np.linspace(X[:, 1].min()-1, X[:, 1].max()+1, 100))
            Z = model.predict(np.c_[XX.ravel(), YY.ravel()])
            Z = Z.reshape(XX.shape)
            
            ax1.contourf(XX, YY, Z, alpha=0.4, cmap='rainbow')
            ax1.scatter(X[:, 0], X[:, 1], c=y, cmap='rainbow')
            ax1.set_title("Decision Boundary")
            
            # Feature importance plot
            importances = model.feature_importances_
            feature_names = [f"Feature {i+1}" for i in range(X.shape[1])]
            ax2.bar(feature_names, importances)
            ax2.set_title("Feature Importance")
            ax2.set_ylabel("Importance")
            plt.xticks(rotation=45)
            
            st.pyplot(fig)
            
            # Display metrics
            st.subheader("Model Performance")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Training Accuracy", 
                         f"{accuracy_score(y_train, model.predict(X_train)):.3f}")
            with col2:
                st.metric("Test Accuracy", 
                         f"{accuracy_score(y_test, y_pred):.3f}")
            
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))

            show_prediction_interface(model, "classification")

def create_boosting_model(model_type, n_estimators, learning_rate, depth_param):
    """Create appropriate boosting model based on type"""
    if model_type == "Gradient Boosting":
        return GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=depth_param,
            random_state=42
        )
    elif model_type == "XGBoost":
        return xgb.XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=depth_param,
            random_state=42
        )
    elif model_type == "CatBoost":
        return CatBoostClassifier(
            iterations=n_estimators,
            learning_rate=learning_rate,
            depth=depth_param,
            random_seed=42,
            verbose=False
        )
    else:  # LightGBM
        return lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=depth_param,
            random_state=42
        )

def handle_random_forest(X, y, X_train, X_test, y_train, y_test, params):
    """Handle Random Forest specific interface and training"""
    n_estimators = st.sidebar.number_input(
        "Number of trees",
        min_value=10,
        max_value=500,
        value=params["n_estimators"]["suggested"]
    )
    
    max_depth = st.sidebar.number_input(
        "Max Depth",
        min_value=1,
        max_value=20,
        value=params["max_depth"]["suggested"]
    )
    
    bootstrap = st.sidebar.selectbox(
        "Bootstrap",
        options=params["bootstrap"]["options"],
        index=0
    )
    
    max_features = st.sidebar.selectbox(
        "Max Features",
        options=params["max_features"]["options"],
        index=0
    )
    
    if st.sidebar.button("Train Model"):
        with st.spinner("Training Random Forest..."):
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                bootstrap=bootstrap,
                max_features=max_features,
                random_state=42
            )
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Plot decision boundary and feature importance
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Decision boundary plot
            XX, YY = np.meshgrid(np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, 100),
                               np.linspace(X[:, 1].min()-1, X[:, 1].max()+1, 100))
            Z = model.predict(np.c_[XX.ravel(), YY.ravel()])
            Z = Z.reshape(XX.shape)
            
            ax1.contourf(XX, YY, Z, alpha=0.4, cmap='rainbow')
            ax1.scatter(X[:, 0], X[:, 1], c=y, cmap='rainbow')
            ax1.set_title("Decision Boundary")
            
            # Feature importance plot
            importances = model.feature_importances_
            feature_names = [f"Feature {i+1}" for i in range(X.shape[1])]
            ax2.bar(feature_names, importances)
            ax2.set_title("Feature Importance")
            ax2.set_ylabel("Importance")
            plt.xticks(rotation=45)
            
            st.pyplot(fig)
            
            # Display metrics
            st.subheader("Model Performance")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Training Accuracy", 
                         f"{accuracy_score(y_train, model.predict(X_train)):.3f}")
            with col2:
                st.metric("Test Accuracy", 
                         f"{accuracy_score(y_test, y_pred):.3f}")
            
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))

            show_prediction_interface(model, "classification")

def handle_decision_tree(X, y, X_train, X_test, y_train, y_test, params):
    """Handle Decision Tree specific interface and training"""
    criterion = st.sidebar.selectbox("Criterion", 
                                   options=params["criterion"]["options"],
                                   index=0)
    
    max_depth = st.sidebar.number_input("Max Depth",
                                      min_value=params["max_depth"]["min"],
                                      max_value=params["max_depth"]["max"],
                                      value=params["max_depth"]["suggested"])
    
    min_samples_split = st.sidebar.number_input("Min Samples Split",
                                               min_value=params["min_samples_split"]["min"],
                                               max_value=params["min_samples_split"]["max"],
                                               value=params["min_samples_split"]["suggested"])
    
    min_samples_leaf = st.sidebar.number_input("Min Samples Leaf",
                                              min_value=params["min_samples_leaf"]["min"],
                                              max_value=params["min_samples_leaf"]["max"],
                                              value=params["min_samples_leaf"]["suggested"])
    
    # Create and train model
    if st.sidebar.button("Train Model"):
        with st.spinner("Training Decision Tree..."):
            model = DecisionTreeClassifier(
                criterion=criterion,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42
            )
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Plot decision boundary
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Decision boundary plot
            XX, YY = np.meshgrid(np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, 100),
                               np.linspace(X[:, 1].min()-1, X[:, 1].max()+1, 100))
            Z = model.predict(np.c_[XX.ravel(), YY.ravel()])
            Z = Z.reshape(XX.shape)
            
            ax1.contourf(XX, YY, Z, alpha=0.4, cmap='rainbow')
            ax1.scatter(X[:, 0], X[:, 1], c=y, cmap='rainbow')
            ax1.set_title("Decision Boundary")
            
            # Tree visualization
            tree_viz = export_graphviz(model, 
                                     feature_names=["Feature 1", "Feature 2"],
                                     class_names=["Class 0", "Class 1"],
                                     filled=True)
            st.graphviz_chart(tree_viz)
            
            # Display metrics
            st.subheader("Model Performance")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Training Accuracy", 
                         f"{accuracy_score(y_train, model.predict(X_train)):.3f}")
            with col2:
                st.metric("Test Accuracy", 
                         f"{accuracy_score(y_test, y_pred):.3f}")
            
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))

            show_prediction_interface(model, "classification")

def handle_knn(X, y, X_train, X_test, y_train, y_test, params):
    """Handle KNN specific interface and training"""
    n_neighbors = st.sidebar.number_input(
        "Number of neighbors (k)",
        min_value=params["n_neighbors"]["min"],
        max_value=params["n_neighbors"]["max"],
        value=params["n_neighbors"]["suggested"],
        help=params["n_neighbors"]["warning"],
        key="knn_n_neighbors"
    )
    
    weights = st.sidebar.selectbox(
        "Weights",
        options=params["weights"]["options"],
        index=0,
        help=params["weights"]["warning"],
        key="knn_weights"
    )
    
    p = st.sidebar.selectbox(
        "Distance Metric",
        options=[1, 2],
        index=1,
        help=params["p"]["warning"],
        key="knn_p"
    )

    if st.sidebar.button("Train Model", key="knn_train"):
        with st.spinner("Training KNN..."):
            model = KNeighborsClassifier(
                n_neighbors=n_neighbors,
                weights=weights,
                p=p
            )
            model.fit(X_train, y_train)
            st.session_state.model = model
            
            y_pred = model.predict(X_test)
            
            # Plot decision boundary
            fig, ax = plt.subplots(figsize=(10, 6))
            XX, YY = np.meshgrid(np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, 100),
                               np.linspace(X[:, 1].min()-1, X[:, 1].max()+1, 100))
            Z = model.predict(np.c_[XX.ravel(), YY.ravel()])
            Z = Z.reshape(XX.shape)
            
            ax.contourf(XX, YY, Z, alpha=0.4, cmap='rainbow')
            ax.scatter(X[:, 0], X[:, 1], c=y, cmap='rainbow')
            ax.set_title("Decision Boundary")
            st.pyplot(fig)
            
            # Display metrics
            st.subheader("Model Performance")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Training Accuracy", 
                         f"{accuracy_score(y_train, model.predict(X_train)):.3f}")
            with col2:
                st.metric("Test Accuracy", 
                         f"{accuracy_score(y_test, y_pred):.3f}")
            
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))

            show_prediction_interface(model, "classification")

def handle_svm(X, y, X_train, X_test, y_train, y_test, params):
    """Handle SVM specific interface and training"""
    C = st.sidebar.number_input(
        "Regularization strength (C)",
        min_value=params["C"]["min"],
        max_value=params["C"]["max"],
        value=params["C"]["suggested"],
        help=params["C"]["warning"],
        key="svm_C"
    )
    
    kernel = st.sidebar.selectbox(
        "Kernel",
        options=params["kernel"]["options"],
        index=0,
        help=params["kernel"]["warning"],
        key="svm_kernel"
    )
    
    gamma = st.sidebar.selectbox(
        "Gamma",
        options=params["gamma"]["options"],
        index=0,
        help=params["gamma"]["warning"],
        key="svm_gamma"
    )

    if st.sidebar.button("Train Model", key="svm_train"):
        with st.spinner("Training SVM..."):
            model = SVC(
                C=C,
                kernel=kernel,
                gamma=gamma,
                random_state=42,
                probability=True
            )
            model.fit(X_train, y_train)
            st.session_state.model = model
            
            y_pred = model.predict(X_test)
            
            # Plot decision boundary
            fig, ax = plt.subplots(figsize=(10, 6))
            XX, YY = np.meshgrid(np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, 100),
                               np.linspace(X[:, 1].min()-1, X[:, 1].max()+1, 100))
            Z = model.predict(np.c_[XX.ravel(), YY.ravel()])
            Z = Z.reshape(XX.shape)
            
            ax.contourf(XX, YY, Z, alpha=0.4, cmap='rainbow')
            ax.scatter(X[:, 0], X[:, 1], c=y, cmap='rainbow')
            ax.set_title("Decision Boundary")
            st.pyplot(fig)
            
            # Display metrics
            st.subheader("Model Performance")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Training Accuracy", 
                         f"{accuracy_score(y_train, model.predict(X_train)):.3f}")
            with col2:
                st.metric("Test Accuracy", 
                         f"{accuracy_score(y_test, y_pred):.3f}")
            
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))

            show_prediction_interface(model, "classification")

def show_prediction_interface(model, model_type):
    """Display prediction interface for both regression and classification models"""
    if model is not None:
        st.sidebar.markdown("## Predict New Value")
        
        # Initialize session state for predictions if not exists
        if 'predictions' not in st.session_state:
            st.session_state.predictions = []
            
        # Clear predictions button
        if st.sidebar.button("Clear Predictions", key=f"clear_predictions_{model_type}"):
            st.session_state.predictions = []
        
        if model_type == "regression":
            # Create a container for the plot
            plot_container = st.empty()
            
            x_new = st.sidebar.number_input("Enter X value", value=5.0)
            
            # Clear predictions button
            if st.sidebar.button("Clear Predictions", key="clear_predictions"):
                st.session_state.predictions = []
            
            # Store prediction button state
            predict_clicked = st.sidebar.button("Make Prediction", key="regression_predict")
            
            if predict_clicked:
                # Make prediction
                prediction = model.predict([[x_new]])[0]
                if isinstance(prediction, np.ndarray):
                    prediction = prediction[0]
                # Calculate actual value using the true relationship (y = 2x + 1)
                actual = 2 * x_new + 1
                
                # Store prediction in session state
                st.session_state.predictions.append({
                    'x': x_new,
                    'predicted': prediction,
                    'actual': actual
                })
                
                # Show results with desired format
                st.sidebar.success(f"✅ Predicted y: {round(prediction, 3)}")
                st.sidebar.info(f"📊 Actual y (true relationship): {round(actual, 3)}")
                st.sidebar.warning(f"📈 Difference: {round(abs(prediction - actual), 3)}")
            
            # Always show the plot with all predictions
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot regression line
            X_line = np.linspace(0, 10, 100).reshape(-1, 1)
            y_line = model.predict(X_line)
            ax.plot(X_line, y_line, 'r-', label='Regression Line', alpha=0.5)
            
            # Plot all stored predictions
            if st.session_state.predictions:
                for i, pred in enumerate(st.session_state.predictions):
                    ax.scatter(pred['x'], pred['predicted'], c="red", marker="X", s=150, 
                             label="Prediction" if i == 0 else "", alpha=0.6)
                    ax.scatter(pred['x'], pred['actual'], c="green", marker="*", s=150, 
                             label="True Value" if i == 0 else "", alpha=0.6)
                    
                    # Add connecting line between prediction and actual
                    ax.plot([pred['x'], pred['x']], [pred['predicted'], pred['actual']], 
                           'k--', alpha=0.3)
            
            ax.set_xlabel('X')
            ax.set_ylabel('y')
            ax.legend()
            
            # Update the plot in the container
            plot_container.pyplot(fig)
            plt.close(fig)
            
            # Show predictions table
            if st.session_state.predictions:
                st.subheader("Prediction History")
                pred_df = pd.DataFrame(st.session_state.predictions)
                pred_df.columns = ['X Value', 'Predicted Y', 'Actual Y']
                pred_df['Error'] = abs(pred_df['Predicted Y'] - pred_df['Actual Y'])
                st.dataframe(pred_df.style.format("{:.3f}"))
                
        else:  # classification
            # Create a container for the plot
            plot_container = st.empty()
            
            x1_new = st.sidebar.number_input("Enter X1 value", value=0.0)
            x2_new = st.sidebar.number_input("Enter X2 value", value=0.0)
            
            predict_clicked = st.sidebar.button("Make Prediction", key="classification_predict")
            
            if predict_clicked:
                x_new = np.array([[x1_new, x2_new]])
                prediction = model.predict(x_new)[0]
                
                # Store prediction in session state
                pred_info = {
                    'X1': x1_new,
                    'X2': x2_new,
                    'Predicted_Class': prediction
                }
                
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(x_new)[0]
                    for i, prob in enumerate(probabilities):
                        pred_info[f'Prob_Class_{i}'] = prob
                
                st.session_state.predictions.append(pred_info)
                
                # Show prediction result
                st.sidebar.success(f"✅ Predicted class: {prediction}")
                if hasattr(model, 'predict_proba'):
                    prob_str = ", ".join([f"Class {i}: {p:.3f}" 
                                        for i, p in enumerate(probabilities)])
                    st.sidebar.info(f"📊 Probabilities:\n{prob_str}")
            
            # Always show the plot with all predictions
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot decision boundary
            XX, YY = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
            Z = model.predict(np.c_[XX.ravel(), YY.ravel()])
            Z = Z.reshape(XX.shape)
            
            ax.contourf(XX, YY, Z, alpha=0.4, cmap='rainbow')
            
            # Plot original data points
            if hasattr(st.session_state, 'X') and hasattr(st.session_state, 'y'):
                X = st.session_state.X
                y = st.session_state.y
                scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='rainbow', alpha=0.6)
                ax.legend(*scatter.legend_elements(), title="Classes")
            
            # Plot all stored predictions
            if st.session_state.predictions:
                pred_df = pd.DataFrame(st.session_state.predictions)
                ax.scatter(pred_df['X1'], pred_df['X2'], c='red', marker='X', s=150, 
                         label='Predictions', alpha=0.8)
            
            ax.set_xlabel('X1')
            ax.set_ylabel('X2')
            if st.session_state.predictions:
                ax.legend()
            
            # Update the plot in the container
            plot_container.pyplot(fig)
            plt.close(fig)
            
            # Show predictions table
            if st.session_state.predictions:
                st.subheader("Prediction History")
                st.dataframe(pd.DataFrame(st.session_state.predictions))

def train_and_evaluate_model(model, X, y, X_train, X_test, y_train, y_test):
    """Common training and evaluation code for all models"""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Get prediction probabilities if available
    if hasattr(model, 'predict_proba'):
        y_pred_proba_train = model.predict_proba(X_train)
        y_pred_proba_test = model.predict_proba(X_test)
        
        # Add actual vs predicted plot
        st.subheader("Prediction Probabilities")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Training Data")
            fig_train = plot_actual_vs_predicted_classification(
                y_train, 
                y_pred_proba_train, 
                "Training: Prediction Probabilities"
            )
            st.pyplot(fig_train)
        
        with col2:
            st.write("Test Data")
            fig_test = plot_actual_vs_predicted_classification(
                y_test, 
                y_pred_proba_test, 
                "Test: Prediction Probabilities"
            )
            st.pyplot(fig_test)
    
    # Plot decision boundary and feature importance/confusion matrix
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Decision boundary
    XX, YY = np.meshgrid(np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, 100),
                       np.linspace(X[:, 1].min()-1, X[:, 1].max()+1, 100))
    Z = model.predict(np.c_[XX.ravel(), YY.ravel()])
    Z = Z.reshape(XX.shape)
    
    ax1.contourf(XX, YY, Z, alpha=0.4, cmap='rainbow')
    ax1.scatter(X[:, 0], X[:, 1], c=y, cmap='rainbow')
    ax1.set_title("Decision Boundary")
    
    # Feature importance or confusion matrix
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_names = [f"Feature {i+1}" for i in range(X.shape[1])]
        ax2.bar(feature_names, importances)
        ax2.set_title("Feature Importance")
        ax2.set_ylabel("Importance")
        plt.xticks(rotation=45)
    else:
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', ax=ax2, cmap='Blues')
        ax2.set_title('Confusion Matrix')
    
    st.pyplot(fig)
    
    # Display metrics
    st.subheader("Model Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Training Accuracy", 
                 f"{accuracy_score(y_train, model.predict(X_train)):.3f}")
    with col2:
        st.metric("Test Accuracy", 
                 f"{accuracy_score(y_test, y_pred):.3f}")
    
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

# Add this after show_classification_interface function:
def show_classification_predictions(model, X, y, fig):
    """Common prediction interface for all classification models"""
    if model is not None:
        st.sidebar.markdown("## Classify New Point")
        
        # Initialize session states if not exists
        if 'prediction_history' not in st.session_state:
            st.session_state.prediction_history = []
            
        # Clear predictions button
        if st.sidebar.button("Clear Predictions", key="clear_predictions"):
            st.session_state.prediction_history = []
            st.session_state.x1 = 0.0
            st.session_state.x2 = 0.0
            st.experimental_rerun()

        # Input features with persisted values
        x1 = st.sidebar.number_input(
            "Feature 1",
            value=st.session_state.get("x1", 0.0),  # Get value from session state
            key="x1_input"
        )
        x2 = st.sidebar.number_input(
            "Feature 2",
            value=st.session_state.get("x2", 0.0),  # Get value from session state
            key="x2_input"
        )
        
        # Save current values to session state
        st.session_state.x1 = x1
        st.session_state.x2 = x2
        
        if st.sidebar.button("Predict", key="predict_btn"):
            new_point = np.array([[x1, x2]])
            prediction = model.predict(new_point)[0]
            probability = model.predict_proba(new_point)[0]
            
            # Store prediction in history
            st.session_state.prediction_history.append({
                "x1": x1,
                "x2": x2,
                "prediction": prediction,
                "prob_0": probability[0],
                "prob_1": probability[1]
            })
            
            # Display results
            st.sidebar.success(f"✅ Predicted Class: {prediction}")
            st.sidebar.info(
                f"📊 Probabilities:\n"
                f"- Class 0: {probability[0]:.3f}\n"
                f"- Class 1: {probability[1]:.3f}"
            )
        
        # Update plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot original data points
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', alpha=0.6)
        
        # Plot decision boundary
        XX, YY = np.meshgrid(np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, 100),
                           np.linspace(X[:, 1].min()-1, X[:, 1].max()+1, 100))
        Z = model.predict(np.c_[XX.ravel(), YY.ravel()])
        Z = Z.reshape(XX.shape)
        ax.contourf(XX, YY, Z, alpha=0.3, cmap='coolwarm')
        
        # Plot prediction history
        if st.session_state.prediction_history:
            for p in st.session_state.prediction_history:
                color = 'red' if p['prediction'] == 1 else 'blue'
                ax.scatter(p['x1'], p['x2'], 
                          c=color, marker='X', s=150, 
                          label='Predictions' if p == st.session_state.prediction_history[0] else "")
        
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        if st.session_state.prediction_history:
            ax.legend()
        st.pyplot(fig)
        
        # Show prediction history table
        if st.session_state.prediction_history:
            st.subheader("Prediction History")
            df = pd.DataFrame(st.session_state.prediction_history)
            st.dataframe(df.style.format({
                "prob_0": "{:.3f}",
                "prob_1": "{:.3f}"
            }))

if __name__ == "__main__":
    main()