from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib
from scipy.stats.mstats import winsorize
from flask_cors import CORS

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
    Generate feature-based explanations, including non-liver
    causes or reasons for abnormal results.
    """

    # We retrieve the user's actual numeric values:
    # (All float except Age, Platelets which are int, but we'll treat them as float for range checks.)
    age = float(data["Age"])
    bilirubin = float(data["Total Bilirubin"])
    alk_phos = float(data["Alkaline Phosphatase"])
    alt = float(data["Alanine Aminotransferase"])
    ast = float(data["Aspartate Aminotransferase"])
    albumin = float(data["Albumin"])
    proteins = float(data["Total Proteins"])
    prothrombin = float(data["Prothrombin Time"])
    platelets = float(data["Platelets"])
    ascites = data["Ascites"]   # "Present"/"Absent"
    liver_firmness = data["LiverFirmness"]  # "Present"/"Absent"
    ast_alt_ratio = calculate_ast_alt_ratio(ast, alt)
    alb_glob_ratio = round(albumin / (proteins - albumin), 2) if proteins > albumin else 0

    explanations = []

    # --- Total Bilirubin ---
    # Normal range ~ 0.1–1.2 mg/dL
    if bilirubin < 0.1:
        explanations.append(
            f"Total Bilirubin ({bilirubin} mg/dL) is slightly below normal, which is uncommon but may occur with certain genetic conditions or lab variations."
        )
    elif bilirubin <= 1.2:
        explanations.append(
            f"Total Bilirubin ({bilirubin} mg/dL) is within normal range (0.1–1.2). "
            "Mild variations can occur from hemolysis, medication, or Gilbert's syndrome."
        )
    elif bilirubin <= 3.5:
        explanations.append(
            f"Total Bilirubin ({bilirubin} mg/dL) is mildly elevated. "
            "Possible reasons include early liver inflammation, hemolysis, or gallbladder issues."
        )
    else:
        explanations.append(
            f"Total Bilirubin ({bilirubin} mg/dL) is significantly elevated. "
            "This can arise from advanced liver disease, bile duct obstruction, or other systemic factors."
        )

    # --- Alkaline Phosphatase (ALP) ---
    # Typical normal range for adults ~ 40–129 U/L (male), 35–104 U/L (female), but we keep a single approach
    if alk_phos < 40:
        explanations.append(
            f"Alkaline Phosphatase ({alk_phos} U/L) is slightly below typical adult range. "
            "Low ALP can occur with malnutrition or genetic factors, not necessarily liver disease."
        )
    elif alk_phos <= 129:
        explanations.append(
            f"Alkaline Phosphatase ({alk_phos} U/L) is within a typical range (~40–129). "
            "ALP can also be affected by bone growth, pregnancy, or certain medications."
        )
    else:
        explanations.append(
            f"Alkaline Phosphatase ({alk_phos} U/L) is elevated. "
            "This may reflect cholestatic liver conditions, bone disorders, or other causes."
        )

    # --- Alanine Aminotransferase (ALT) ---
    # Normal range roughly 7–55 U/L (male), 7–45 U/L (female); we'll use 7–55
    if alt < 7:
        explanations.append(
            f"ALT ({alt} U/L) is slightly below the usual lower bound. "
            "This is often clinically insignificant, but can sometimes reflect poor muscle mass or other factors."
        )
    elif alt <= 55:
        explanations.append(
            f"ALT ({alt} U/L) is within normal limits (7–55). "
            "ALT is mostly liver-specific but can vary with muscle injury or certain medications."
        )
    else:
        explanations.append(
            f"ALT ({alt} U/L) is elevated. "
            "This may indicate liver cell injury, but can also be influenced by muscle damage or medications."
        )

    # --- Aspartate Aminotransferase (AST) ---
    # Normal range roughly 8–48 U/L
    if ast < 8:
        explanations.append(
            f"AST ({ast} U/L) is slightly below the typical lower bound. "
            "Usually not a concern, but extremely low AST can occur with B6 deficiency or muscle issues."
        )
    elif ast <= 48:
        explanations.append(
            f"AST ({ast} U/L) is within normal range (8–48). "
            "Mild fluctuations can occur from exercise, medications, or minor muscle injury."
        )
    else:
        explanations.append(
            f"AST ({ast} U/L) is elevated. "
            "Causes include liver inflammation, cardiac/muscle injury, or other systemic conditions."
        )

    # --- AST/ALT Ratio ---
    explanations.append(
        f"AST/ALT Ratio: {ast_alt_ratio}. Ratios >2 often suggest advanced liver pathology (e.g., alcoholic liver disease), "
        "but other factors can affect this ratio."
    )

    # --- Albumin ---
    # Normal range ~ 3.5–5.0 g/dL
    if albumin < 3.5:
        explanations.append(
            f"Albumin ({albumin} g/dL) is below normal (3.5–5.0). "
            "Low albumin can indicate liver dysfunction, poor nutrition, or chronic illness."
        )
    elif albumin <= 5.0:
        explanations.append(
            f"Albumin ({albumin} g/dL) is within normal range (3.5–5.0). "
            "Variations can occur with hydration status, diet, and inflammation."
        )
    else:
        explanations.append(
            f"Albumin ({albumin} g/dL) is slightly above typical range. "
            "This is uncommon and may reflect dehydration or lab error. Rarely pathological."
        )

    # --- Total Proteins ---
    # Normal range ~ 6.3–7.9 g/dL
    if proteins < 6.3:
        explanations.append(
            f"Total Proteins ({proteins} g/dL) are below normal (6.3–7.9). "
            "Low levels can result from liver issues, malnutrition, or kidney losses."
        )
    elif proteins <= 7.9:
        explanations.append(
            f"Total Proteins ({proteins} g/dL) are within normal range (6.3–7.9). "
            "Slight variations can be due to hydration or diet."
        )
    else:
        explanations.append(
            f"Total Proteins ({proteins} g/dL) exceed normal range. "
            "Possible causes include chronic inflammation, infections, or certain blood disorders."
        )

    # --- Prothrombin Time (PT) ---
    # Normal range ~ 9.4–12.5 seconds
    if prothrombin < 9.4:
        explanations.append(
            f"Prothrombin Time ({prothrombin}s) is below the usual lower limit (9.4s). "
            "Faster clotting is less common but can occur with certain genetic factors or lab variation."
        )
    elif prothrombin <= 12.5:
        explanations.append(
            f"Prothrombin Time ({prothrombin}s) is within normal range (9.4–12.5s). "
            "Clotting function appears adequate; mild deviations can occur with vitamin K intake."
        )
    else:
        explanations.append(
            f"Prothrombin Time ({prothrombin}s) is prolonged. "
            "This may reflect decreased liver synthesis of clotting factors, vitamin K deficiency, or anticoagulant use."
        )

    # --- Platelets ---
    # Normal range ~ 150–450 ×10³/µL
    if platelets < 150:
        explanations.append(
            f"Platelets ({platelets} ×10³/µL) are below normal (150–450). "
            "Thrombocytopenia can occur with liver disease, immune conditions, or bone marrow issues."
        )
    elif platelets <= 450:
        explanations.append(
            f"Platelets ({platelets} ×10³/µL) are within normal range (150–450). "
            "Slight variations can occur from infection, inflammation, or pregnancy."
        )
    else:
        explanations.append(
            f"Platelets ({platelets} ×10³/µL) are above normal. "
            "Thrombocytosis may result from inflammation, iron deficiency, or reactive processes."
        )

    # --- Albumin/Globulin Ratio ---
    explanations.append(
        f"Albumin Globulin Ratio: {alb_glob_ratio}. "
        "A low A/G ratio (<1.0) may suggest chronic inflammation or liver issues, "
        "while a higher ratio is often normal but can vary with diet or dehydration."
    )

    # --- Ascites / Liver Firmness (Qualitative) ---
    if ascites == "Present":
        explanations.append(
            "Ascites is reported as Present. While often associated with advanced liver disease, "
            "it can also occur in heart failure, kidney problems, or certain cancers."
        )
    else:
        explanations.append(
            "Ascites is reported as Absent. This reduces the likelihood of severe fluid retention "
            "commonly seen in advanced liver or cardiac conditions."
        )

    if liver_firmness == "Present":
        explanations.append(
            "Liver Firmness is reported as Present, indicating possible fibrosis or cirrhosis. "
            "However, imaging or elastography is needed for confirmation."
        )
    else:
        explanations.append(
            "Liver Firmness is reported as Absent. This suggests no clinically detected stiffness, "
            "though early fibrosis may still be undetectable without imaging."
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

        # Apply log transformation
        input_data[log_transform_cols] = np.log1p(input_data[log_transform_cols])

        # Apply winsorization
        for col in winsorize_cols:
            input_data[col] = winsorize(input_data[col], limits=[0.05, 0.05])

        # Ensure correct column order
        input_data = input_data[scaler.feature_names_in_]

        # Apply scaling
        input_scaled = scaler.transform(input_data)

        # Make prediction
        predicted_class = voting_clf.predict(input_scaled)[0]

        # Map prediction to disease stage
        stage_mapping = {
            0: "Healthy (No Liver Disease)",
            1: "Hepatitis (Liver Inflammation)",
            2: "Fibrosis (Scarring of the Liver)",
            3: "Cirrhosis (Severe Liver Damage)"
        }

        # Basic explanation for each stage (optional)
        stage_explanation = {
            0: "Overall values suggest normal liver function. Minor variations may still occur.",
            1: "Liver inflammation indicated. Further tests may be needed to confirm cause (viral, autoimmune, etc.).",
            2: "Signs of liver scarring (fibrosis). Condition may be chronic; consult a specialist.",
            3: "Advanced liver damage (cirrhosis) detected. Urgent medical follow-up recommended."
        }

        # Generate feature-level explanations
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
