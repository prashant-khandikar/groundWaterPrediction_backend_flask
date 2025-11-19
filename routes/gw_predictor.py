import pandas as pd
import numpy as np
import joblib
import os

# -------------------------
# Paths to model & CSV
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # routes/
MODEL_PATH = os.path.join(BASE_DIR, "groundwater_village_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "merged_dataset.csv")

# -------------------------
# Load model & dataset
# -------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found at {MODEL_PATH}. Train the model first.")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ Dataset not found at {DATA_PATH}.")

model = joblib.load(MODEL_PATH)
merged = pd.read_csv(DATA_PATH)

# -------------------------
# Prediction function
# -------------------------
def predict_groundwater(state, district, tehsil, village, future_year, growth_rate=0.015):
    loc = merged[
        (merged["STATE/UT"].str.lower() == state.lower().strip()) &
        (merged["DISTRICT"].str.lower() == district.lower().strip()) &
        (merged["TEHSIL"].str.lower() == tehsil.lower().strip()) &
        (merged["VILLAGE"].str.lower() == village.lower().strip())
    ]

    if loc.empty:
        return {"predicted_gwl": None, "trend": "❌ Location not found"}

    base_gwl = loc["Groundwater_Level"].iloc[0]
    base_population = loc["Population"].iloc[0]
    base_rain = loc["Actual (mm)"].iloc[0]
    slope_rain = loc["Rainfall_Slope"].iloc[0]
    gwl_slope = loc["GWL_Slope"].iloc[0]

    year_diff = future_year - 2023

    # Project population
    projected_population = base_population * ((1 + growth_rate) ** year_diff)
    population_effect = 0.00005 * (projected_population - base_population)

    # Project rainfall
    projected_rain = base_rain + slope_rain * year_diff

    # Model input
    X_input = pd.DataFrame([{
        "Year": future_year,
        "Actual (mm)": projected_rain,
        "Population": projected_population,
        "elevation": loc["elevation"].iloc[0],
        "WELL DEPTH": loc["WELL DEPTH"].iloc[0],
        "pH": loc["pH"].iloc[0],
        "TDS (mg/l)": loc["TDS (mg/l)"].iloc[0]
    }])

    # Prediction
    pred = model.predict(X_input)[0]

    # Adjust
    pred += gwl_slope * year_diff
    pred += 0.01 * slope_rain * year_diff
    pred -= population_effect

    pred = np.clip(pred, 0, 20)
    
    # ✅ CORRECTED TREND LOGIC: 
    # Lower values = Better (shallower water table)
    # Higher values = Worse (deeper water table)
    # If pred > base_gwl: water table is deeper (worse) = "Decreasing" 
    # If pred < base_gwl: water table is shallower (better) = "Increasing"
    trend = "Decreasing" if pred > base_gwl else "Increasing"

    return {"predicted_gwl": round(pred, 2), "trend": trend}