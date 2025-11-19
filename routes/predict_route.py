from flask import Blueprint, request, jsonify
from .gw_predictor import predict_groundwater
import pandas as pd
import os

# Load your dataset (add this at the top with your existing imports)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "merged_dataset.csv")
merged = pd.read_csv(DATA_PATH)

bp = Blueprint("gw_predictor", __name__, url_prefix="/api")

@bp.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    state = data.get("state")
    district = data.get("district")
    tehsil = data.get("tehsil")
    village = data.get("village")
    year = data.get("year")

    if not all([state, district, tehsil, village, year]):
        return jsonify({"error": "❌ Missing required fields"}), 400

    # ✅ UPDATED: Get predictions for multiple years (3 years before and after)
    target_year = int(year)
    years_range = list(range(target_year - 3, target_year + 4))
    
    predictions = {}
    for y in years_range:
        result = predict_groundwater(state, district, tehsil, village, y)
        predictions[y] = result

    return jsonify(predictions)

# ✅ ADD THESE NEW ROUTES TO THE EXISTING predict_route.py
@bp.route("/states", methods=["GET"])
def get_states():
    try:
        states = merged["STATE/UT"].str.strip().unique().tolist()
        states = [state for state in states if state and state != "nan"]
        return jsonify(sorted(states))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@bp.route("/districts/<state>", methods=["GET"])
def get_districts(state):
    try:
        districts = merged[merged["STATE/UT"].str.lower() == state.lower().strip()]["DISTRICT"].str.strip().unique().tolist()
        districts = [district for district in districts if district and district != "nan"]
        return jsonify(sorted(districts))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@bp.route("/tehsils/<state>/<district>", methods=["GET"])
def get_tehsils(state, district):
    try:
        tehsils = merged[
            (merged["STATE/UT"].str.lower() == state.lower().strip()) &
            (merged["DISTRICT"].str.lower() == district.lower().strip())
        ]["TEHSIL"].str.strip().unique().tolist()
        tehsils = [tehsil for tehsil in tehsils if tehsil and tehsil != "nan"]
        return jsonify(sorted(tehsils))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@bp.route("/villages/<state>/<district>/<tehsil>", methods=["GET"])
def get_villages(state, district, tehsil):
    try:
        villages = merged[
            (merged["STATE/UT"].str.lower() == state.lower().strip()) &
            (merged["DISTRICT"].str.lower() == district.lower().strip()) &
            (merged["TEHSIL"].str.lower() == tehsil.lower().strip())
        ]["VILLAGE"].str.strip().unique().tolist()
        villages = [village for village in villages if village and village != "nan"]
        return jsonify(sorted(villages))
    except Exception as e:
        return jsonify({"error": str(e)}), 500