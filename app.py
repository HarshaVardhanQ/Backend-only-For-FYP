from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib
from scipy.stats.mstats import winsorize
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ---------------- Load Preprocessing Tools and Model ----------------
log_transform_cols = joblib.load("log_transform_cols.pkl")
winsorize_cols = joblib.load("winsorize_cols.pkl")
scaler = joblib.load("scaler.pkl")
voting_clf = joblib.load("voting_classifier.pkl")

# Normal reference ranges for generating explanations
NORMAL_RANGES = {
    "Total Bilirubin": (0.1, 1.2),
    "Alkaline Phosphatase": (40, 129),
    "Alanine Aminotransferase": (7, 55),
    "Aspartate Aminotransferase": (8, 48),
    "Albumin": (3.5, 5.0),
    "Total Proteins": (6.3, 7.9),
    "Prothrombin Time": (9.4, 12.5),
    "Platelets": (150, 450)  # in 10³/µL
}

# Stage mapping and corresponding explanations
stage_mapping = {
    0: "Healthy (No Liver Disease)",
    1: "Hepatitis (Liver Inflammation)",
    2: "Fibrosis (Scarring of the Liver)",
    3: "Cirrhosis (Severe Liver Damage)"
}

stage_explanations = {
    0: ("Liver function tests are within normal ranges: bilirubin (0.1–1.2 mg/dL), ALP (40–129 U/L), "
        "ALT (7–55 U/L), AST (8–48 U/L), albumin (3.5–5.0 g/dL) and PT (9.4–12.5 s). These findings indicate a healthy liver."),
    1: ("Mild to moderate enzyme elevations with slight bilirubin increase suggest liver inflammation typical of hepatitis. "
        "Further serological tests may help determine the cause."),
    2: ("Moderate increases in liver enzymes, a falling albumin level, and a modestly prolonged PT point toward fibrosis. "
        "Imaging and follow-up are advised."),
    3: ("Severely abnormal liver tests, combined with thrombocytopenia, strongly suggest advanced cirrhosis. "
        "Immediate specialist evaluation is recommended.")
}

def calculate_ast_alt_ratio(ast, alt):
    return round(ast / alt, 2) if alt > 0 else 0

def calculate_fib4_score(age, ast, alt, platelets):
    return round((age * ast) / (platelets * (alt ** 0.5)), 2) if alt > 0 else 0

def generate_feature_explanations(data, final_stage):
    """
    Generate detailed explanations for each lab feature.
    If the final stage is "Healthy", borderline abnormal values are explained in milder terms.
    """
    explanations = []
    # Unpack numeric values
    age = float(data["Age"])
    bilirubin = float(data["Total Bilirubin"])
    alk_phos = float(data["Alkaline Phosphatase"])
    alt = float(data["Alanine Aminotransferase"])
    ast = float(data["Aspartate Aminotransferase"])
    albumin = float(data["Albumin"])
    proteins = float(data["Total Proteins"])
    prothrombin = float(data["Prothrombin Time"])
    platelets = float(data["Platelets"])
    ascites = data["Ascites"]       # "Present"/"Absent"
    liver_firmness = data["LiverFirmness"]  # "Present"/"Absent"

    ast_alt_ratio = calculate_ast_alt_ratio(ast, alt)
    alb_glob_ratio = round(albumin / (proteins - albumin), 2) if proteins > albumin else 0
    isHealthy = "Healthy" in final_stage

    def explain_value(name, value, normal_range):
        low, high = normal_range
        if value < low:
            msg = f"{name} ({value}) is below the normal range ({low}–{high})."
            if isHealthy:
                msg += " Although classified as Healthy, consider rechecking this value."
            return msg
        elif value > high:
            msg = f"{name} ({value}) is elevated compared to the normal range ({low}–{high})."
            if isHealthy:
                msg += " Despite a Healthy classification, this borderline elevation may warrant further evaluation."
            return msg
        else:
            return f"{name} ({value}) is within the normal range ({low}–{high})."

    explanations.append(explain_value("Total Bilirubin", bilirubin, NORMAL_RANGES["Total Bilirubin"]))
    explanations.append(explain_value("Alkaline Phosphatase", alk_phos, NORMAL_RANGES["Alkaline Phosphatase"]))
    explanations.append(explain_value("ALT", alt, NORMAL_RANGES["Alanine Aminotransferase"]))
    explanations.append(explain_value("AST", ast, NORMAL_RANGES["Aspartate Aminotransferase"]))
    explanations.append(f"AST/ALT Ratio: {ast_alt_ratio}. Ratios above 2 may suggest alcoholic liver disease or advanced fibrosis; lower ratios are common in healthy livers or acute injury.")
    explanations.append(explain_value("Albumin", albumin, NORMAL_RANGES["Albumin"]))
    explanations.append(explain_value("Total Proteins", proteins, NORMAL_RANGES["Total Proteins"]))
    explanations.append(explain_value("Prothrombin Time", prothrombin, NORMAL_RANGES["Prothrombin Time"]))

    # Platelets require special handling
    platelets_range = NORMAL_RANGES["Platelets"]
    if platelets < platelets_range[0]:
        msg = f"Platelets ({platelets} ×10³/µL) are below normal ({platelets_range[0]}–{platelets_range[1]})."
        if isHealthy:
            msg += " Even in a Healthy prediction, this value should be rechecked."
        explanations.append(msg)
    elif platelets > platelets_range[1]:
        msg = f"Platelets ({platelets} ×10³/µL) are above normal ({platelets_range[0]}–{platelets_range[1]})."
        if isHealthy:
            msg += " Despite a Healthy classification, consider further evaluation."
        explanations.append(msg)
    else:
        explanations.append(f"Platelets ({platelets} ×10³/µL) are within the normal range ({platelets_range[0]}–{platelets_range[1]}).")

    explanations.append(f"Albumin/Globulin Ratio: {alb_glob_ratio}. A low ratio (<1.0) may be seen in chronic liver disease, while a normal or high ratio is generally reassuring.")

    if ascites == "Present":
        explanations.append("Ascites is reported as Present. This finding is concerning for advanced liver disease.")
    else:
        explanations.append("Ascites is reported as Absent, which is more typical of healthy or early-stage livers.")

    if liver_firmness == "Present":
        explanations.append("Liver Firmness is reported as Present, suggesting possible fibrosis or cirrhosis. Further imaging may be warranted.")
    else:
        explanations.append("Liver Firmness is reported as Absent, indicating no overt signs of advanced scarring.")

    return explanations

@app.route('/')
def home():
    return jsonify({"message": "Liver Disease Prediction API is running!"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Convert categorical inputs
        gender = 1 if str(data["Gender"]).strip().lower() == "male" else 0
        ascites = 1 if data["Ascites"].strip().lower() == "present" else 0
        liver_firmness = 1 if data["LiverFirmness"].strip().lower() == "present" else 0

        albumin_globulin_ratio = round(
            data["Albumin"] / (data["Total Proteins"] - data["Albumin"]), 2
        ) if data["Total Proteins"] > data["Albumin"] else 0

        ast_alt_ratio = calculate_ast_alt_ratio(
            data["Aspartate Aminotransferase"],
            data["Alanine Aminotransferase"]
        )
        fib4_score = calculate_fib4_score(
            data["Age"],
            data["Aspartate Aminotransferase"],
            data["Alanine Aminotransferase"],
            data["Platelets"]
        )
        afld_indicator = 1 if ast_alt_ratio >= 2 else 0

        # Prepare input DataFrame for preprocessing
        feature_names = [
            "Age", "Gender", "Total Bilirubin", "Alkaline Phosphatase",
            "Alanine Aminotransferase", "Aspartate Aminotransferase", "AST ALT Ratio",
            "Albumin", "Total Proteins", "Prothrombin Time", "Platelets",
            "Albumin Globulin Ratio", "FIB_4_Score", "Ascites", "LiverFirmness",
            "AFLD_Indicator"
        ]
        input_data = pd.DataFrame([[
            data["Age"],
            gender,
            data["Total Bilirubin"],
            data["Alkaline Phosphatase"],
            data["Alanine Aminotransferase"],
            data["Aspartate Aminotransferase"],
            ast_alt_ratio,
            data["Albumin"],
            data["Total Proteins"],
            data["Prothrombin Time"],
            data["Platelets"],
            albumin_globulin_ratio,
            fib4_score,
            ascites,
            liver_firmness,
            afld_indicator
        ]], columns=feature_names)

        # Preprocess the data
        input_data[log_transform_cols] = np.log1p(input_data[log_transform_cols])
        for col in winsorize_cols:
            input_data[col] = winsorize(input_data[col], limits=[0.05, 0.05])
        input_data = input_data[scaler.feature_names_in_]
        input_scaled = scaler.transform(input_data)

        # Model prediction
        predicted_class = voting_clf.predict(input_scaled)[0]
        final_stage = stage_mapping.get(predicted_class, "Unknown")

        # Generate detailed feature explanations, using the final stage for context
        feature_explanations = generate_feature_explanations(data, final_stage)

        response = {
            "Predicted Stage": final_stage,
            "Stage Explanation": stage_explanations.get(predicted_class, "No explanation available."),
            "Feature Explanations": feature_explanations,
            "Ascites": "Present" if ascites == 1 else "Absent",
            "LiverFirmness": "Present" if liver_firmness == 1 else "Absent",
            "Calculated Values": {
                "AST ALT Ratio": ast_alt_ratio,
                "FIB-4 Score": fib4_score,
                "Albumin Globulin Ratio": albumin_globulin_ratio
            }
        }

        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001, use_reloader=False)
