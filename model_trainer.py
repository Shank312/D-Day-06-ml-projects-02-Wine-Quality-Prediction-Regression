
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def train_models(data):
    X = data.drop("quality", axis = 1)
    Y = data["quality"]

    # Train-Test Split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size = 0.2, random_state = 42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42)

    }

    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, Y_train)
        trained_models[name] = (model, X_test, Y_test)
    
    return trained_models