
import joblib        # joblib is a library used to save and load the python objects (like ML models)

def save_best_model(best_model, path="wine_quality_model.pkl"):
    # This function will save a trained model into a file for future use

    joblib.dump(best_model, path)
    # joblib.dump() takes the model (best_model) and saves it to the given path (default: "wine_quality_model.pkl")

    print(f"Model saved at {path}")
    # Print a confirmation message showing where the model file was saved  