
# Import metrices to check model performance 
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np                       # For mathematical operations (like square root)

# Functions to evaluate multiple trained models
def evaluate_models(trained_models):
    results = {}                         # Dictionary to store results for each models

    # Loop through each model in the dictionary
    # 'name' = model name (string)
    # 'model' = the trained ML model objects
    # 'X_test' = test features
    # 'Y_test' = true values of target
    for name, (model, X_test, Y_test) in trained_models.items():

        # Use the model to predict values on the test data
        Y_pred = model.predict(X_test)

        # Calculate R2 Score -> how well model explains variance (close to 1 = better)
        r2 = r2_score(Y_test, Y_pred)

        # Calculate RMSE -> error between actual and predicted (lower = better)
        rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))

        # Store results for this model inside dictionary
        results[name] = {"R2 Score": r2, "RMSE": rmse}

    # Return dictionary containing all models' scores
    return results