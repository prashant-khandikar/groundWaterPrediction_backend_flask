from flask import Flask
from flask_cors import CORS
from routes.predict_route import bp as gw_level_predictor_bp

app = Flask(__name__)
CORS(app)  # ✅ allow frontend requests

# Register blueprint
app.register_blueprint(gw_level_predictor_bp, url_prefix="/api")

@app.route("/", methods=["GET"])
def home():
    return "<h2>Groundwater Prediction API is running! Use /api/predict to get predictions.</h2>"

if __name__ == "__main__":
    app.run(debug=True)
