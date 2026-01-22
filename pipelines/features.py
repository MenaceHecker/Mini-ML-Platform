import pandas as pd
def build_features(input_path: str, output_path: str):
    df = pd.read_csv(input_path)

    #Basic cleaning here
    df = df.dropna()