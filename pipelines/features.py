import pandas as pd
def build_features(input_path: str, output_path: str):
    df = pd.read_csv(input_path)

    #Basic cleaning here
    df = df.dropna()

    # Example feature logic
    df["RoomsPerHousehold"] = df["AveRooms"] / df["AveOccup"]
    df["BedroomsPerRoom"] = df["AveBedrms"] / df["AveRooms"]

    df.to_csv(output_path, index=False)