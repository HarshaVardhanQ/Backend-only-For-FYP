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

# Input validation max limits (used in frontend validation):
#   age: [18, 90],
#   bilirubin: [0.1, 15.0],
#   alk_phos: [40, 1200],
#   alt: [7, 500],
#   ast: [8, 500],
#   albumin: [1.5, 5.0],
#   proteins: [2.0, 7.9],
#   prothrombin: [9.4, 35.0],
#   platelets: [55, 450]

# Normal reference ranges for healthy individuals (for general reference)
NORMAL_RANGES = {
    "Total Bilirubin": (0.1, 1.2),
    "Alkaline Phosphatase": (40, 129),
    "ALT": (7, 55),
    "AST": (8, 48),
    "Albumin": (3.5, 5.0),
    "Total Proteins": (6.3, 7.9),
    "Prothrombin Time": (9.4, 12.5),
    "Platelets": (150, 450)
}

# Stage‑specific expected thresholds (illustrative values)
THRESHOLDS_BY_STAGE = {
    0: {  # Healthy
        "Total Bilirubin": (0.1, 1.2),
        "Alkaline Phosphatase": (40, 129),
        "ALT": (7, 55),
        "AST": (8, 48),
        "Albumin": (3.5, 5.0),
        "Total Proteins": (6.3, 7.9),
        "Prothrombin Time": (9.4, 12.5),
        "Platelets": (150, 450)
    },
    1: {  # Hepatitis (Liver Inflammation)
        "Total Bilirubin": (1.2, 3.5),
        "Alkaline Phosphatase": (40, 300),
        "ALT": (55, 250),
        "AST": (48, 250),
        "Albumin": (3.5, 5.0),
        "Total Proteins": (6.3, 7.9),
        "Prothrombin Time": (12.5, 16.0),
        "Platelets": (120, 450)
    },
    2: {  # Fibrosis (Scarring)
        "Total Bilirubin": (2.0, 6.0),
        "Alkaline Phosphatase": (80, 500),
        "ALT": (100, 300),
        "AST": (100, 300),
        "Albumin": (2.8, 3.8),
        "Total Proteins": (5.8, 7.5),
        "Prothrombin Time": (14.0, 20.0),
        "Platelets": (80, 140)
    },
    3: {  # Cirrhosis (Advanced Damage)
        "Total Bilirubin": (3.5, 15.0),
        "Alkaline Phosphatase": (120, 1200),
        "ALT": (50, 500),   # ALT may be lower in cirrhosis due to reduced hepatocyte mass
        "AST": (80, 500),
        "Albumin": (1.5, 3.0),
        "Total Proteins": (4.5, 7.0),
        "Prothrombin Time": (18.0, 35.0),
        "Platelets": (55, 100)
    }
}

# Overall stage mapping and summary explanations
stage_mapping = {
    0: "Healthy (No Liver Disease)",
    1: "Hepatitis (Liver Inflammation)",
    2: "Fibrosis (Scarring of the Liver)",
    3: "Cirrhosis (Severe Liver Damage)"
}

stage_explanations = {
    0: "Liver function tests are within normal limits, supporting a healthy liver.",
    1: "Moderate elevations in liver enzymes and bilirubin suggest liver inflammation typical of hepatitis.",
    2: "Further enzyme elevations, a moderate rise in bilirubin, reduced albumin, and prolonged clotting point toward fibrosis; further evaluation is advised.",
    3: "Markedly abnormal test values – significantly elevated bilirubin and alkaline phosphatase, low albumin, prolonged clotting time, and low platelets – strongly indicate advanced cirrhosis; immediate specialist evaluation is recommended."
}

def calculate_ast_alt_ratio(ast, alt):
    return round(ast / alt, 2) if alt > 0 else 0

def calculate_fib4_score(age, ast, alt, platelets):
    return round((age * ast) / (platelets * (alt ** 0.5)), 2) if alt > 0 else 0

def generate_feature_explanations(data, predicted_class):
    """
    Generate detailed explanations for each lab feature using stage‑specific thresholds.
    The output messages use descriptive language without revealing internal threshold values.
    """
    explanations = []
    # Extract patient values
    age = float(data["Age"])
    bilirubin = float(data["Total Bilirubin"])
    alk_phos = float(data["Alkaline Phosphatase"])
    alt = float(data["Alanine Aminotransferase"])
    ast = float(data["Aspartate Aminotransferase"])
    albumin = float(data["Albumin"])
    proteins = float(data["Total Proteins"])
    prothrombin = float(data["Prothrombin Time"])
    platelets = float(data["Platelets"])
    ascites = data["Ascites"]       # "Present" or "Absent"
    liver_firmness = data["LiverFirmness"]

    ast_alt_ratio = calculate_ast_alt_ratio(ast, alt)
    alb_glob_ratio = round(albumin / (proteins - albumin), 2) if proteins > albumin else 0

    # Get stage-specific thresholds; fallback to NORMAL_RANGES if not found
    thresholds = THRESHOLDS_BY_STAGE.get(predicted_class, NORMAL_RANGES)
    isHealthy = (predicted_class == 0)

    def explain_value(name, value):
        low, high = thresholds.get(name, (None, None))
        if low is None or high is None:
            return f"{name} is {value}."
        
        if value < low:
            msg = f"{name} is {value}, which is lower than what is typically observed. "
            msg += "This might be due to individual variation or a lab anomaly. "
            if isHealthy:
                msg += "Even with a generally healthy liver, it might be wise to recheck this value."
            else:
                msg += "In the setting of liver disease, lower-than-typical values can sometimes occur, but this should still be evaluated carefully."
            return msg
        elif value > high:
            msg = f"{name} is {value}, which is higher than what is normally seen. "
            msg += "This elevation may indicate liver injury or dysfunction. "
            if isHealthy:
                msg += "Even though the overall assessment is healthy, this abnormality might require further investigation."
            else:
                msg += "This finding is in line with the changes seen in this stage of liver disease and should be interpreted along with other clinical data."
            return msg
        else:
            return f"{name} is {value} and falls within normal limits. This is reassuring."

    explanations.append(explain_value("Total Bilirubin", bilirubin))
    explanations.append(explain_value("Alkaline Phosphatase", alk_phos))
    explanations.append(explain_value("ALT", alt))
    explanations.append(explain_value("AST", ast))
    explanations.append(f"AST/ALT Ratio is {ast_alt_ratio}. Ratios above 2 may suggest alcoholic injury or advanced fibrosis, while lower ratios are more common in healthy livers or acute inflammation.")
    explanations.append(explain_value("Albumin", albumin))
    explanations.append(explain_value("Total Proteins", proteins))
    explanations.append(explain_value("Prothrombin Time", prothrombin))
    
    # Special handling for Platelets
    plat_low, plat_high = thresholds.get("Platelets", (150, 450))
    if platelets < plat_low:
        msg = f"Platelets are {platelets} ×10³/µL, which is lower than typical values. "
        msg += "Low platelets can be associated with portal hypertension and splenic sequestration. "
        if isHealthy:
            msg += "Even with a healthy overall picture, this value might warrant a recheck."
        platelets_msg = msg
    elif platelets > plat_high:
        msg = f"Platelets are {platelets} ×10³/µL, which is higher than normal. "
        msg += "While this may be reactive, it is less common in liver disease. "
        if isHealthy:
            msg += "Further evaluation may still be appropriate."
        platelets_msg = msg
    else:
        platelets_msg = f"Platelets are {platelets} ×10³/µL and within normal limits, which is reassuring."
    explanations.append(platelets_msg)

    explanations.append(f"Albumin/Globulin Ratio is {alb_glob_ratio}. A lower ratio may indicate chronic inflammation or liver scarring.")
    
    if ascites == "Present":
        explanations.append("Ascites is reported as Present. This finding is concerning as it can indicate fluid accumulation from advanced liver disease.")
    else:
        explanations.append("Ascites is reported as Absent, which is a favorable sign and consistent with a healthy or early-stage liver.")
    
    if liver_firmness == "Present":
        explanations.append("Liver Firmness is reported as Present, suggesting possible fibrosis or cirrhosis. Further evaluation with imaging or elastography is recommended.")
    else:
        explanations.append("Liver Firmness is reported as Absent, indicating no overt signs of advanced scarring, although early fibrosis cannot be completely ruled out.")
    
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

        # Derived features
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

        # Preprocess data
        input_data[log_transform_cols] = np.log1p(input_data[log_transform_cols])
        for col in winsorize_cols:
            input_data[col] = winsorize(input_data[col], limits=[0.05, 0.05])
        input_data = input_data[scaler.feature_names_in_]
        input_scaled = scaler.transform(input_data)

        # Model prediction
        predicted_class = voting_clf.predict(input_scaled)[0]
        final_stage = stage_mapping.get(predicted_class, "Unknown")

        # Generate stage-specific feature explanations
        feature_explanations = generate_feature_explanations(data, predicted_class)

        response = {
            "Predicted Stage": final_stage,
            "Stage Explanation": stage_explanations.get(predicted_class, "No explanation available."),
            "Feature Explanations": feature_explanations,
            "AFLD Indicator": "Positive (Suggestive of AFLD)" if afld_indicator == 1 else "Negative",
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
