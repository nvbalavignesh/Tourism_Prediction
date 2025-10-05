---
title: Tourism Package Prediction
emoji: 🧳
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.28.1
app_file: app.py
pinned: false
license: mit
---

# Tourism Package Prediction

This is a Streamlit app that predicts tourism package preferences using an XGBoost machine learning model.

## Features

- Interactive web interface for tourism package prediction
- Machine learning model trained on tourism preference data
- Real-time predictions based on user inputs

## Usage

The app provides input fields for various tourism-related features and returns predictions about package preferences.

## Model

The underlying model is an XGBoost classifier trained through an automated MLOps pipeline using:
- MLflow for experiment tracking
- Hugging Face Hub for model storage
- GitHub Actions for CI/CD automation

## Technology Stack

- **Frontend**: Streamlit
- **ML Model**: XGBoost
- **Deployment**: Hugging Face Spaces
- **CI/CD**: GitHub Actions
- **Model Tracking**: MLflow