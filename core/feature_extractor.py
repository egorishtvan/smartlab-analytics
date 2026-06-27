import pandas as pd

def extract_sensor_features(csv_path: str) -> dict:
    """Reads sensor CSV and extracts basic statistical metrics as a tsfresh placeholder"""
    try:
        df = pd.read_csv(csv_path)
        return {
            "mean": float(df["Value"].mean()),
            "max": float(df["Value"].max()),
            "min": float(df["Value"].min()),
            "std": float(df["Value"].std()),
            "last_value": float(df["Value"].iloc[-1])
        }
    except Exception as e:
        return {"error": str(e)}
