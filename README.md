# Kolmogorov-Arnold Network (Recurrent)

## Introduction

This project implements a Kolmogorov-Arnold Network (Recurrent) for load forecasting. It uses Temporal Basis Functions and Temporal Spline Functions to model temporal data dependencies and produce forecasts for energy consumption based on past features. This architecture is particularly suited for load forecasting tasks and can be extended for other time-series forecasting applications.

### Key Features:
- **Temporal Basis Functions**: Captures both linear and non-linear relationships in the input data using SiLU (Sigmoid Linear Unit) activation.
- **B-Spline Temporal Spline Function**: Provides flexible and smooth modeling of temporal data using B-splines with adjustable grid sizes and spline degrees.
- **Recurrent Structure**: Allows the model to maintain hidden states across time steps, making it suitable for sequential and time-series data like energy load forecasting.
- **Multiple Evaluation Metrics**: The model tracks performance using various metrics, including RMSE, MAE, and SMAPE.


## Code Structure

The code is split into several key components, which include:
- **KARN Model**: This is the main model that combines temporal basis functions and spline functions to capture hidden state dependencies across time steps.
- **Data Handling**: A `SlidingWindowsDataset` class is used to process the time-series data into sliding windows for training, validation, and testing.
- **Training and Evaluation**: Functions to train and evaluate the KARN model on the provided datasets, tracking multiple metrics including RMSE, MAE, and SMAPE.
- **Temporal Basis Function**: Captures both linear and non-linear relationships in the input data using SiLU activation.
- **B-Spline Temporal Spline Function**: Provides flexible, smooth modeling of temporal data using B-splines with adjustable grid sizes and spline degrees.
- **Recurrent Structure**: The network updates hidden states at each time step, making it suitable for sequential data like time series.


## Installation and Setup
Install torch, pandas, numpy, and plotly
