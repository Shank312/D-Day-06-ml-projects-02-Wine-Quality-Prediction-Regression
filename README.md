# 🍷 Wine Quality Prediction (Regression)

This project predicts the quality of red wine based on physicochemical features using Machine Learning. It demonstrates a complete ML pipeline: **data loading, exploratory data analysis (EDA), model training, evaluation, and saving the best model**.

---

## 📂 Project Structure

ml-projects/02_wine_quality/
│
├── data_loader.py # Load the dataset
├── visualize.py # Exploratory Data Analysis using Seaborn/Matplotlib
├── model_trainer.py # Train Linear Regression, Random Forest, Gradient Boosting models
├── evaluator.py # Evaluate models using R² Score and RMSE
├── model_exporter.py # Save the best model as wine_quality_model.pkl
├── main.py # Integrates all scripts into a single pipeline
├── winequality-red.csv # Dataset (Kaggle Red Wine dataset)
├── wine_quality_model.pkl # Saved best model
└── README.md


---

## 📝 Dataset

- Source: [Kaggle Wine Quality Dataset](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009)  
- Type: Regression  
- Features (12 numeric physicochemical attributes):
  - fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol
- Target:
  - `quality` (score between 0–10)

---

## 🔍 Exploratory Data Analysis (EDA)

- Checked for missing values → **No missing values** ✅  
- Visualizations:
  - Pairplot for feature relationships  
  - Correlation heatmap to find important features  

---

## 🤖 Models Trained

1. **Linear Regression**  
2. **Random Forest Regressor**  
3. **Gradient Boosting Regressor**

**Evaluation Metrics:**
- **R² Score**: Measures how much variance is explained by the model (higher = better)  
- **RMSE (Root Mean Squared Error)**: Measures average prediction error (lower = better)

**Results:**

| Model                  | R² Score | RMSE  |
|------------------------|----------|-------|
| Linear Regression      | 0.31     | 0.618 |
| Random Forest          | 0.46     | 0.546 |
| Gradient Boosting      | 0.40     | 0.575 |

> Random Forest is the best model for this dataset.

---

## 💾 Saving the Model

- The best-performing model (`Random Forest`) is saved as `wine_quality_model.pkl` using `joblib`.  
- This allows **loading and using the model later** without retraining.

---

## ⚡ How to Run

1. Clone the repository:
```bash
git clone <your-repo-url>
cd ml-projects/02_wine_quality/

Install dependencies:
pip install -r requirements.txt

Run the pipeline:
python main.py

This will execute the full ML pipeline: load data → visualize → train → evaluate → save best model.

🛠 Tools & Libraries

Python 3.x

pandas, numpy

scikit-learn

matplotlib, seaborn

joblib


📈 Next Steps / Improvements

Hyperparameter tuning for Random Forest & Gradient Boosting

Cross-validation for more robust evaluation

Feature importance analysis

Try additional regression models (XGBoost, LightGBM)


📌 Author

Shankar Kumar

GitHub: https://github.com/Shank312


