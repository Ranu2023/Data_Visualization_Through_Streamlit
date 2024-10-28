import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
import joblib
import os

# Set the page configuration
st.set_page_config(page_title="Enhanced Data Visualizer & Model Trainer", layout="wide", page_icon="📊", initial_sidebar_state='expanded')

# Title and Introduction
st.title("📊 Enhanced Data Visualizer & Model Trainer - Web App")
st.markdown("""
This app allows you to upload your data, visualize it through various plots, analyze descriptive statistics, and run multiple regression models with hyperparameter tuning to find the best one based on MSE, MAE, and R² Score.
""")

# Sidebar for uploading data
with st.sidebar:
    st.header("Upload and Select Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("File successfully uploaded!")

if uploaded_file is not None:
    # Data Preview Section
    st.subheader("Data Preview")
    preview_rows = st.slider("How many rows to display?", 5, 100, 20)
    st.dataframe(df.head(preview_rows))

    # Preprocess the dataset: Convert dates to numerical features and encode categorical variables
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
                df[f"{col}_year"] = df[col].dt.year
                df[f"{col}_month"] = df[col].dt.month
                df[f"{col}_day"] = df[col].dt.day
                df.drop(columns=[col], inplace=True)
            except Exception:
                df = pd.get_dummies(df, columns=[col], drop_first=True)

    # Data Analysis Section
    st.subheader("Data Analysis Tasks")
    analysis_options = ["Descriptive Statistics", "Missing Values Analysis", "Correlation Heatmap"]
    selected_analysis = st.multiselect("Select analysis tasks you want to perform:", analysis_options)

    if "Descriptive Statistics" in selected_analysis:
        st.write("### Descriptive Statistics")
        st.write(df.describe())

    if "Missing Values Analysis" in selected_analysis:
        st.write("### Missing Values Analysis")
        missing_values = df.isnull().sum()
        missing_values = missing_values[missing_values > 0]
        st.write(missing_values)

    if "Correlation Heatmap" in selected_analysis:
        st.write("### Correlation Heatmap")
        numeric_df = df.select_dtypes(include=[np.number])
        plt.figure(figsize=(10, 7))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
        st.pyplot(plt)

    # Data Visualization Section
    st.subheader("Data Visualization")
    plot_types = ["Line Plot", "Bar Plot", "Scatter Plot", "Histogram", "Interactive Plot", "Box Plot", "Pair Plot"]
    selected_plots = st.multiselect("Choose plot types:", plot_types)

    if selected_plots:
        columns = df.columns.tolist()
        x_axis = st.selectbox("Select the X-axis", options=columns, index=0)
        y_axis_options = ['None'] + columns
        y_axis = st.selectbox("Select the Y-axis", options=y_axis_options, index=0)

    for plot_type in selected_plots:
        st.write(f"### {plot_type}")
        if plot_type == "Interactive Plot":
            fig = px.scatter(df, x=x_axis, y=y_axis if y_axis != 'None' else None, title=f"{y_axis} vs {x_axis}")
            st.plotly_chart(fig, use_container_width=True)
        elif plot_type == "Pair Plot":
            sns.pairplot(df)
            st.pyplot(plt)
        else:
            fig, ax = plt.subplots()
            if plot_type == "Line Plot" and y_axis != 'None':
                sns.lineplot(x=x_axis, y=y_axis, data=df, ax=ax)
            elif plot_type == "Bar Plot" and y_axis != 'None':
                sns.barplot(x=x_axis, y=y_axis, data=df, ax=ax)
            elif plot_type == "Scatter Plot" and y_axis != 'None':
                sns.scatterplot(x=x_axis, y=y_axis, data=df, ax=ax)
            elif plot_type == "Histogram":
                sns.histplot(data=df, x=x_axis, kde=True, ax=ax)
            elif plot_type == "Box Plot" and y_axis != 'None':
                sns.boxplot(x=x_axis, y=y_axis, data=df, ax=ax)
            st.pyplot(fig)

    # Model Training and Selection
    st.subheader("Model Training & Selection")
    target_column = st.selectbox("Select Target Column", options=df.columns)
    
    if st.button("Train Models and Select Best"):
        feature_columns = [col for col in df.columns if col != target_column]
        X = df[feature_columns]
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define models and their hyperparameters
        models = {
            'Linear Regression': make_pipeline(SimpleImputer(strategy='mean'), LinearRegression()),
            'Ridge Regression': make_pipeline(SimpleImputer(strategy='mean'), Ridge()),
            'Lasso Regression': make_pipeline(SimpleImputer(strategy='mean'), Lasso()),
            'Random Forest': make_pipeline(SimpleImputer(strategy='mean'), RandomForestRegressor(random_state=42))
        }

        # Define hyperparameters for tuning
        hyperparameters = {
            'Ridge Regression': {'ridge__alpha': [0.1, 1.0, 10.0]},
            'Lasso Regression': {'lasso__alpha': [0.1, 1.0, 10.0]},
            'Random Forest': {'randomforestregressor__n_estimators': [50, 100, 200]}
        }

        mse_scores = {}
        mae_scores = {}
        r2_scores = {}
        best_models = {}

        for name, model in models.items():
            if name in hyperparameters:
                grid = GridSearchCV(model, hyperparameters[name], scoring='neg_mean_squared_error', cv=5)
                grid.fit(X_train, y_train)
                best_model = grid.best_estimator_
            else:
                model.fit(X_train, y_train)
                best_model = model

            predictions = best_model.predict(X_test)
            mse_scores[name] = mean_squared_error(y_test, predictions)
            mae_scores[name] = mean_absolute_error(y_test, predictions)
            r2_scores[name] = r2_score(y_test, predictions)
            best_models[name] = best_model

        # Display results in a table
        st.write("### Model Performance")
        performance_df = pd.DataFrame({
            "Model": mse_scores.keys(),
            "MSE": mse_scores.values(),
            "MAE": mae_scores.values(),
            "R2 Score": r2_scores.values()
        }).sort_values(by="MSE")
        st.dataframe(performance_df)

        # Identify and save the best model
        best_model_name = performance_df.iloc[0]['Model']
        best_model = best_models[best_model_name]
        joblib.dump(best_model, f"{best_model_name}_model.joblib")
        st.write(f"Best Model: {best_model_name} with MSE: {mse_scores[best_model_name]}")
        st.write("Model saved successfully.")

        # Load model section
        st.subheader("Load and Test Saved Model")
        saved_model_path = st.file_uploader("Upload a saved model", type=["joblib"])
        if saved_model_path:
            loaded_model = joblib.load(saved_model_path)
            st.write(f"Loaded Model: {loaded_model}")
            new_prediction = loaded_model.predict(X_test[:5])
            st.write("Sample Predictions:", new_prediction)

# Footer
st.markdown("---")
st.markdown("Developed By RANU SINGH - A versatile data visualization and model training app built with Streamlit.")
