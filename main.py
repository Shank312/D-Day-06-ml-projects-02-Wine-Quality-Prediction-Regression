
from data_loader import load_data
from eda import perform_eda
from model_trainer import train_models
from evaluator import evaluate_models
from model_exporter import save_best_model

# Step 1: Load data
data = load_data()

# Step 2: Perform EDA
perform_eda(data)

# Step 3: Train models
trained_models = train_models(data)

# Step 4: Evaluate
results = evaluate_models(trained_models)
print("\nModel Comparison:\n", results)

# Step 5: Save the best model
best_model_name = max(results, key=lambda x: results[x]["R2 Score"])
best_model = trained_models[best_model_name][0]
save_best_model(best_model)