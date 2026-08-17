# README: Adaptive Sales Forecasting Notebook

This notebook contains the full implementation for the Adaptive Sales Forecasting final year project. It reproduces the preprocessing, feature engineering, model training, adaptive meta-learner, statistical evaluation, and explainability outputs used in the dissertation.

    Runtime environment used: Google Colab with NVIDIA L4 GPU.
    Python notebook environment: Google Colab.
    Dataset: Walmart Sales Forecasting dataset downloaded using KaggleHub.
    Random seeds are fixed where supported by the model APIs.
    Main outputs are saved under walmart_forecasting/results/.
    The Temporal Fusion Transformer is trained only on the top 200 high-volume store-department series due to GPU memory limits.
    The Random Forest, XGBoost, Seasonal Naive, meta-learner, statistical testing, and SHAP pipelines are reproducible from this notebook.

# Before Running

The Kaggle API token has been removed for security reasons.

Before running the notebook, Kaggle authentication must be configured by the user.

The dataset is downloaded using KaggleHub, so the notebook requires a valid Kaggle account with access to the Walmart Sales Forecasting dataset.
How to Run the Notebook

Run the notebook from top to bottom

The notebook should be run in order because later sections depend on variables, models, files, and outputs created in earlier cells. Running individual cells out of order may cause missing-variable or missing-file errors.
Expected Outputs

The notebook generates:

    merged and preprocessed Walmart sales data
    engineered lag, rolling, cyclical, contextual, and macroeconomic features
    Seasonal Naive, Random Forest, XGBoost, and TFT model outputs
    adaptive meta-learner routing decisions
    per-department model performance files
    statistical significance test results
    SHAP explainability plots
    final model comparison tables

# Runtime Notes

Some cells may take a long time to run, especially:

    Random Forest training
    XGBoost training
    Temporal Fusion Transformer training
    SHAP value computation
    Optuna hyperparameter tuning

The TFT section is intentionally limited to the top 200 high-volume store-department series because training the full 3,326 series exceeded Google Colab GPU memory limits during development.
