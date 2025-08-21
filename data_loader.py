
import pandas as pd

def load_data(path="WineQT.csv"):
    # Load CSV into a DataFrame
    data = pd.read_csv(path)
    return data