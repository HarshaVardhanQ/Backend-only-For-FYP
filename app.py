from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib
from scipy.stats.mstats import winsorize
from flask_cors import CORS

# ---------------- Backend Feature Limits (as per frontend and literature) ----------------
# These limits have been gathered from liver function test references:
# age: [18, 90]
# Total Bilirubin: [0.1, 15.0] mg/dL    | Normal: 0.1–1.2; Mild elevation: 1.2–3.5; Moderate: 3.5–8; Severe: >8
# Alkaline Phosphatase: [40, 1200] U/L  | Normal: 40–129; Mild: 130–300; Moderate: 300–600; Severe: >600
# Alanine Aminotransferase (ALT): [7, 500] U/L  | Normal: 7–55; Mild: 55–150; Moderate: 150–300; Severe: >300
# Aspartate Aminotransferase (AST): [8, 500] U/L   | Normal: 8–48; Mild: 48–150; Moderate: 150–300; Severe: >300
# Albumin: [1.5, 5.0] g/dL            | Normal: 3.5–5.0; Moderate decrease: 2.5–3.5; Severe decrease: <2.5
# Total Proteins: [2.0, 7.9] g/dL     | Normal: 6.3–7.9; Lower values suggest liver insufficiency
# Prothrombin Time: [9.4, 35.0] sec   | Normal: 9.4–12.5; Mild prolongation: 12.5–20; Severe: >20
# Platelets: [55, 450] (×10³/µL)       | Normal: 150–450; Thrombocytopenia: <150 (severe if <100)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load preprocessing tools and models
log_transform_cols = joblib.load("log_transform_cols.pkl")
winsorize_cols = joblib.load("winsorize_cols.pkl")
scaler = joblib.load("scaler.pkl")
voting_clf = joblib.load("voting_classifier.pkl")

@app.route('/')
def home():
    return jsonify({"message": "Liver Disease Prediction API is running!"})

# ---------------- Utility Functions ----------------

def calculate_ast_alt_ratio(ast, alt):
    """Calculate AST/ALT ratio."""
    return round(ast / alt, 2) if alt > 0 else 0

def calculate_fib4_score(age, ast, alt, platelets):
    """Calculate FIB-4 score."""
    return round((age * ast) / (platelets * (alt ** 0.5)), 2) if alt > 0 else 0

def generate_feature_explanations(data):
    """
    Generate detailed explanations for each feature based on updated reference ranges.
    """
    # Retrieve numeric values (all as float for consistency)
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

    explanations = []

    # --- Total Bilirubin ---
    if bilirubin < 0.1:
        explanations.append(
            f"Total Bilirubin ({bilirubin} mg/dL) is below the measurable range. This is unusual and might be due to lab variation."
        )
    elif bilirubin <= 1.2:
        explanations.append(
            f"Total Bilirubin ({bilirubin} mg/dL) is within the normal range (0.1–1.2 mg/dL). This supports healthy liver function."
        )
    elif bilirubin <= 3.5:
        explanations.append(
            f"Total Bilirubin ({bilirubin} mg/dL) is mildly elevated. Such levels are often seen in hepatitis or early inflammatory changes."
        )
    elif bilirubin <= 8:
        explanations.append(
            f"Total Bilirubin ({bilirubin} mg/dL) is moderately elevated, which may be indicative of progressing fibrosis."
        )
    else:
        explanations.append(
            f"Total Bilirubin ({bilirubin} mg/dL) is severely elevated. Levels above 8 mg/dL are commonly associated with advanced liver cirrhosis or severe bile duct obstruction."
        )

    # --- Alkaline Phosphatase (ALP) ---
    if alk_phos < 40:
        explanations.append(
            f"Alkaline Phosphatase ({alk_phos} U/L) is below the expected range. This could occur due to malnutrition or genetic conditions."
        )
    elif alk_phos <= 129:
        explanations.append(
            f"Alkaline Phosphatase ({alk_phos} U/L) is within the normal range (40–129 U/L), supporting normal liver and bone metabolism."
        )
    elif alk_phos <= 300:
        explanations.append(
            f"Alkaline Phosphatase ({alk_phos} U/L) is mildly elevated. This may indicate hepatic inflammation (hepatitis) or early signs of fibrosis."
        )
    elif alk_phos <= 600:
        explanations.append(
            f"Alkaline Phosphatase ({alk_phos} U/L) is moderately elevated. Such a level can be seen in progressing fibrosis with cholestatic features."
        )
    else:
        explanations.append(
            f"Alkaline Phosphatase ({alk_phos} U/L) is severely elevated, which is frequently observed in advanced cirrhosis or cholestatic liver disease."
        )

    # --- Alanine Aminotransferase (ALT) ---
    if alt < 7:
        explanations.append(
            f"ALT ({alt} U/L) is below the measurable range, usually not clinically significant."
        )
    elif alt <= 55:
        explanations.append(
            f"ALT ({alt} U/L) is within the normal range (7–55 U/L), consistent with healthy liver tissue."
        )
    elif alt <= 150:
        explanations.append(
            f"ALT ({alt} U/L) is moderately elevated. This degree of increase is often seen in hepatitis due to inflammation."
        )
    elif alt <= 300:
        explanations.append(
            f"ALT ({alt} U/L) is markedly elevated. Values in this range may suggest ongoing liver injury and early fibrosis."
        )
    else:
        explanations.append(
            f"ALT ({alt} U/L) is severely elevated, which is commonly associated with advanced liver damage or cirrhosis."
        )

    # --- Aspartate Aminotransferase (AST) ---
    if ast < 8:
        explanations.append(
            f"AST ({ast} U/L) is below the expected range, a finding that is typically not worrisome."
        )
    elif ast <= 48:
        explanations.append(
            f"AST ({ast} U/L) is within normal limits (8–48 U/L). This is typical of a healthy liver."
        )
    elif ast <= 150:
        explanations.append(
            f"AST ({ast} U/L) is moderately elevated. This level is often seen in hepatitis or mild liver injury."
        )
    elif ast <= 300:
        explanations.append(
            f"AST ({ast} U/L) is significantly elevated, which may indicate progression from hepatitis to fibrosis."
        )
    else:
        explanations.append(
            f"AST ({ast} U/L) is severely elevated, a pattern that is frequently associated with advanced liver cirrhosis."
        )

    # --- AST/ALT Ratio ---
    explanations.append(
        f"AST/ALT Ratio: {ast_alt_ratio}. Ratios above 2 can be suggestive of alcoholic liver disease or advanced fibrosis, while lower ratios are more typical of acute hepatitis."
    )

    # --- Albumin ---
    if albumin < 1.5:
        explanations.append(
            f"Albumin ({albumin} g/dL) is extremely low. This is a rare finding and may indicate severe liver dysfunction."
        )
    elif albumin < 2.5:
        explanations.append(
            f"Albumin ({albumin} g/dL) is severely decreased. Such low levels are often observed in advanced cirrhosis."
        )
    elif albumin < 3.5:
        explanations.append(
            f"Albumin ({albumin} g/dL) is moderately low. This may be seen in fibrosis or chronic liver inflammation."
        )
    else:
        explanations.append(
            f"Albumin ({albumin} g/dL) is within the normal range (3.5–5.0 g/dL), indicating good synthetic liver function."
        )

    # --- Total Proteins ---
    if proteins < 6.3:
        explanations.append(
            f"Total Proteins ({proteins} g/dL) are below the normal range (6.3–7.9 g/dL), suggesting possible liver insufficiency or malnutrition."
        )
    elif proteins <= 7.9:
        explanations.append(
            f"Total Proteins ({proteins} g/dL) are within normal limits (6.3–7.9 g/dL)."
        )
    else:
        explanations.append(
            f"Total Proteins ({proteins} g/dL) are elevated. Although uncommon in liver disease, this could indicate chronic inflammation or other systemic conditions."
        )

    # --- Prothrombin Time (PT) ---
    if prothrombin < 9.4:
        explanations.append(
            f"Prothrombin Time ({prothrombin}s) is below the expected range, which is rarely seen and usually not concerning."
        )
    elif prothrombin <= 12.5:
        explanations.append(
            f"Prothrombin Time ({prothrombin}s) is within the normal range (9.4–12.5s), suggesting adequate clotting factor synthesis."
        )
    elif prothrombin <= 20:
        explanations.append(
            f"Prothrombin Time ({prothrombin}s) is moderately prolonged. This can be an early indicator of liver dysfunction or fibrosis."
        )
    else:
        explanations.append(
            f"Prothrombin Time ({prothrombin}s) is severely prolonged. Values above 20 seconds are frequently observed in advanced cirrhosis due to impaired liver synthesis of clotting factors."
        )

    # --- Platelets ---
    if platelets < 55:
        explanations.append(
            f"Platelets ({platelets} ×10³/µL) are extremely low, which is concerning and can be seen in severe liver disease or bone marrow suppression."
        )
    elif platelets < 150:
        explanations.append(
            f"Platelets ({platelets} ×10³/µL) are below normal. Thrombocytopenia is common in liver disease, particularly in cirrhosis due to splenic sequestration."
        )
    else:
        explanations.append(
            f"Platelets ({platelets} ×10³/µL) are within the normal range (150–450 ×10³/µL), which is a positive sign regarding liver health."
        )

    # --- Albumin/Globulin Ratio ---
    explanations.append(
        f"Albumin/Globulin Ratio: {alb_glob_ratio}. A low ratio (<1.0) may be seen in chronic liver disease and inflammation, while a normal or high ratio is generally reassuring."
    )

    # --- Qualitative Findings: Ascites and Liver Firmness ---
    if ascites == "Present":
        explanations.append(
            "Ascites is reported as Present. This is often associated with advanced liver disease (fibrosis or cirrhosis) but may also occur with heart or kidney issues."
        )
    else:
        explanations.append(
            "Ascites is reported as Absent, which is more typical of healthy livers or early-stage disease."
        )

    if liver_firmness == "Present":
        explanations.append(
            "Liver Firmness is reported as Present. This may indicate fibrosis or cirrhosis; further imaging (like elastography) is recommended for confirmation."
        )
    else:
        explanations.append(
            "Liver Firmness is reported as Absent, suggesting no overt signs of advanced scarring, although early fibrosis might not be palpable."
        )

    return explanations

# ---------------- Prediction Endpoint ----------------

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("🔹 Received Data:", data)

        # Convert categorical inputs
        gender = 1 if str(data["Gender"]).strip().lower() == "male" else 0
        ascites = 1 if data["Ascites"].strip().lower() == "present" else 0
        liver_firmness = 1 if data["LiverFirmness"].strip().lower() == "present" else 0

        # Compute derived features
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
        afld_indicator = 1 if ast_alt_ratio >= 2 else 0  # NAFLD Indicator

        # Prepare input DataFrame
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

        print("✅ Processed Input Data:\n", input_data)

        # Apply log transformation to specified columns
        input_data[log_transform_cols] = np.log1p(input_data[log_transform_cols])

        # Apply winsorization
        for col in winsorize_cols:
            input_data[col] = winsorize(input_data[col], limits=[0.05, 0.05])

        # Ensure correct column order for the scaler
        input_data = input_data[scaler.feature_names_in_]

        # Apply scaling
        input_scaled = scaler.transform(input_data)

        # Make prediction using the loaded model
        predicted_class = voting_clf.predict(input_scaled)[0]

        # Map prediction to disease stage
        stage_mapping = {
            0: "Healthy (No Liver Disease)",
            1: "Hepatitis (Liver Inflammation)",
            2: "Fibrosis (Scarring of the Liver)",
            3: "Cirrhosis (Severe Liver Damage)"
        }

        # Enhanced explanations for each stage based on test value ranges:
        stage_explanation = {
            0: (
                "Liver function tests are within normal ranges: bilirubin (0.1–1.2 mg/dL), ALP (40–129 U/L), "
                "ALT (7–55 U/L), AST (8–48 U/L), albumin (3.5–5.0 g/dL) and PT (9.4–12.5 s). These findings indicate a healthy liver."
            ),
            1: (
                "Mild to moderate enzyme elevations (e.g., ALT and AST up to ~150 U/L) with slight bilirubin increase (1.2–3.5 mg/dL) "
                "suggest liver inflammation typical of hepatitis. Further serological tests may help determine the cause."
            ),
            2: (
                "Moderate increases in liver enzymes (ALT/AST in the 150–300 U/L range), bilirubin in the 3.5–8 mg/dL range, "
                "a falling albumin level, and a modestly prolonged PT point toward the development of fibrosis. Imaging and follow-up are advised."
            ),
            3: (
                "Severely abnormal liver tests—with ALT/AST >300 U/L, bilirubin >8 mg/dL, albumin <2.5 g/dL, and PT >20 s—combined with thrombocytopenia "
                "strongly suggest advanced cirrhosis. Immediate specialist evaluation and management are recommended."
            )
        }

        # Generate detailed feature-level explanations
        feature_explanations = generate_feature_explanations({
            "Age": data["Age"],
            "Total Bilirubin": data["Total Bilirubin"],
            "Alkaline Phosphatase": data["Alkaline Phosphatase"],
            "Alanine Aminotransferase": data["Alanine Aminotransferase"],
            "Aspartate Aminotransferase": data["Aspartate Aminotransferase"],
            "Albumin": data["Albumin"],
            "Total Proteins": data["Total Proteins"],
            "Prothrombin Time": data["Prothrombin Time"],
            "Platelets": data["Platelets"],
            "Ascites": "Present" if ascites == 1 else "Absent",
            "LiverFirmness": "Present" if liver_firmness == 1 else "Absent"
        })

        response = {
            "Predicted Stage": stage_mapping[predicted_class],
            "Stage Explanation": stage_explanation.get(predicted_class, "No explanation available."),
            "Feature Explanations": feature_explanations,
            "Ascites": "Present" if ascites == 1 else "Absent",
            "LiverFirmness": "Present" if liver_firmness == 1 else "Absent",
            "Calculated Values": {
                "AST ALT Ratio": ast_alt_ratio,
                "FIB-4 Score": fib4_score,
                "Albumin Globulin Ratio": albumin_globulin_ratio
            }
        }

        print("🔹 Response Sent:", response)
        return jsonify(response)

    except Exception as e:
        print("❌ Error:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001, use_reloader=False)
