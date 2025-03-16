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

# Stage‑specific expected ranges (illustrative values based on literature and updated input limits)
# Typical trends:
# - Hepatitis: moderate enzyme elevations.
# - Fibrosis: further enzyme elevations with worsening synthetic function.
# - Cirrhosis: markedly high bilirubin/ALP, low albumin, prolonged PT, and low platelets.
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
        "ALT": (50, 500),   # ALT may be lower in cirrhosis due to decreased hepatocyte mass
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
    0: ("Liver function tests are within normal ranges, supporting a healthy liver."),
    1: ("Moderate elevations in liver enzymes and bilirubin suggest liver inflammation typical of hepatitis."),
    2: ("Further elevations in enzymes, a moderate rise in bilirubin, reduced albumin, and a prolonged PT point toward fibrosis. Further evaluation is advised."),
    3: ("Severely abnormal test values – markedly elevated bilirubin and ALP, low albumin, prolonged PT, and thrombocytopenia – strongly indicate advanced cirrhosis. Immediate specialist evaluation is recommended.")
}

def calculate_ast_alt_ratio(ast, alt):
    return round(ast / alt, 2) if alt > 0 else 0

def calculate_fib4_score(age, ast, alt, platelets):
    return round((age * ast) / (platelets * (alt ** 0.5)), 2) if alt > 0 else 0

def generate_feature_explanations(data, predicted_class):
    """
    Generate detailed explanations for each lab feature using stage‑specific thresholds.
    Each explanation compares the patient value with the expected range for the predicted stage,
    providing descriptive context on whether the value is low, within range, or elevated.
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

    # Get thresholds for the predicted stage; fallback to NORMAL_RANGES if not found
    thresholds = THRESHOLDS_BY_STAGE.get(predicted_class, NORMAL_RANGES)
    isHealthy = (predicted_class == 0)

    def explain_value(name, value):
        low, high = thresholds.get(name, (None, None))
        if low is None or high is None:
            return f"{name}: {value} (No specific threshold available.)"
        
        if value < low:
            msg = f"{name} is {value}, which is below the expected range of {low}–{high}. "
            msg += "A value this low might be due to individual variation or even a lab error. "
            if isHealthy:
                msg += "Although overall liver function appears healthy, this anomaly suggests it may be prudent to recheck the value."
            else:
                msg += "In the context of liver disease, lower-than-expected values may sometimes be observed; however, it still merits careful evaluation."
            return msg
        elif value > high:
            msg = f"{name} is {value}, which is elevated compared to the expected range of {low}–{high}. "
            msg += "This elevation is concerning as it can be indicative of liver injury or dysfunction. "
            if isHealthy:
                msg += "Even though the overall prediction is healthy, this abnormality may warrant further investigation."
            else:
                msg += "Such an elevated value is consistent with the expected changes in this stage of liver disease and should be interpreted alongside other clinical findings."
            return msg
        else:
            return f"{name} is {value}, which falls within the expected range of {low}–{high}. This is a reassuring finding."

    explanations.append(explain_value("Total Bilirubin", bilirubin))
    explanations.append(explain_value("Alkaline Phosphatase", alk_phos))
    explanations.append(explain_value("ALT", alt))
    explanations.append(explain_value("AST", ast))
    explanations.append(f"AST/ALT Ratio is {ast_alt_ratio}. Typically, ratios above 2 may indicate alcoholic injury or advanced fibrosis, whereas lower ratios are more common in healthy livers or acute inflammation.")
    explanations.append(explain_value("Albumin", albumin))
    explanations.append(explain_value("Total Proteins", proteins))
    explanations.append(explain_value("Prothrombin Time", prothrombin))
    
    # Special handling for Platelets (different thresholds may apply)
    plat_low, plat_high = thresholds.get("Platelets", (150, 450))
    if platelets < plat_low:
        msg = f"Platelets are {platelets} ×10³/µL, which is below the expected range of {plat_low}–{plat_high}. "
        msg += "Low platelet counts can be associated with portal hypertension and splenic sequestration, particularly in advanced liver disease. "
        if isHealthy:
            msg += "Even with an overall healthy prediction, this warrants a recheck."
        platelets_msg = msg
    elif platelets > plat_high:
        msg = f"Platelets are {platelets} ×10³/µL, which is above the expected range of {plat_low}–{plat_high}. "
        msg += "An elevated platelet count may be reactive, but it is less typical in liver disease. "
        if isHealthy:
            msg += "Further evaluation may still be needed despite a healthy overall classification."
        platelets_msg = msg
    else:
        platelets_msg = f"Platelets are {platelets} ×10³/µL, which falls within the expected range of {plat_low}–{plat_high}. This is reassuring."
    explanations.append(platelets_msg)

    explanations.append(f"Albumin/Globulin Ratio is {alb_glob_ratio}. A lower ratio (often below 1.0) may indicate chronic inflammation or liver scarring.")
    
    if ascites == "Present":
        explanations.append("Ascites is reported as Present. This is concerning as it often indicates fluid accumulation due to advanced liver disease (e.g., cirrhosis) and increased portal pressure.")
    else:
        explanations.append("Ascites is reported as Absent, which is a favorable sign and consistent with either a healthy liver or early-stage liver conditions.")
    
    if liver_firmness == "Present":
        explanations.append("Liver Firmness is reported as Present. This finding suggests that there may be significant fibrosis or cirrhosis, and further evaluation using imaging or elastography is recommended.")
    else:
        explanations.append("Liver Firmness is reported as Absent, indicating that there are no overt signs of advanced scarring; however, early fibrosis cannot be completely ruled out.")
    
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
