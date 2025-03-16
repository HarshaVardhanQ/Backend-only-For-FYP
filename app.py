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
    Generate feature-based explanations by comparing measured values to healthy ranges.
    The descriptions use phrases like “elevated,” “very high,” “low,” and “within healthy range”
    to indicate the severity of deviations and how they may correlate with different liver disease stages.
    
    Healthy Ranges:
      - Total Bilirubin: 1.2–3.5 mg/dL
      - Alkaline Phosphatase: 40–300 U/L
      - ALT: 55–250 U/L
      - AST: 48–250 U/L
      - Albumin: 3.5–5.0 g/dL
      - Total Proteins: 6.3–7.9 g/dL
      - Prothrombin Time: 12.5–16.0 s
      - Platelets: 120–450 ×10³/µL
    """
    age = float(data["Age"])
    bilirubin = float(data["Total Bilirubin"])
    alk_phos = float(data["Alkaline Phosphatase"])
    alt = float(data["Alanine Aminotransferase"])
    ast = float(data["Aspartate Aminotransferase"])
    albumin = float(data["Albumin"])
    proteins = float(data["Total Proteins"])
    prothrombin = float(data["Prothrombin Time"])
    platelets = float(data["Platelets"])
    ascites = data["Ascites"]        # "Present"/"Absent"
    liver_firmness = data["LiverFirmness"]  # "Present"/"Absent"
    
    # Calculate derived ratios
    ast_alt_ratio = calculate_ast_alt_ratio(ast, alt)
    alb_glob_ratio = round(albumin / (proteins - albumin), 2) if proteins > albumin else 0

    explanations = []

    # --- Total Bilirubin (Healthy: 1.2–3.5 mg/dL) ---
    if bilirubin < 1.2:
        explanations.append(
            f"Total Bilirubin is {bilirubin} mg/dL, which is lower than the healthy range (1.2–3.5 mg/dL). Although low bilirubin is rarely a primary concern, it could be due to lab variation or an atypical metabolic state."
        )
    elif bilirubin <= 3.5:
        explanations.append(
            f"Total Bilirubin is {bilirubin} mg/dL, falling within the healthy range of 1.2–3.5 mg/dL. This is generally reassuring, indicating normal bile metabolism and liver function."
        )
    else:
        if bilirubin <= 5.0:
            explanations.append(
                f"Total Bilirubin is {bilirubin} mg/dL, which is moderately elevated compared to the healthy range. This elevation may reflect early liver inflammation or bile flow disturbances, commonly seen in initial liver injury."
            )
        else:
            explanations.append(
                f"Total Bilirubin is {bilirubin} mg/dL, which is very high relative to the healthy range. Such a marked elevation strongly suggests significant liver dysfunction, possibly due to advanced liver disease or bile duct obstruction."
            )

    # --- Alkaline Phosphatase (Healthy: 40–300 U/L) ---
    if alk_phos < 40:
        explanations.append(
            f"Alkaline Phosphatase is {alk_phos} U/L, below the healthy range of 40–300 U/L. Low levels are uncommon and may be associated with certain metabolic or nutritional issues rather than direct liver injury."
        )
    elif alk_phos <= 300:
        explanations.append(
            f"Alkaline Phosphatase is {alk_phos} U/L, which is within the healthy range. This supports normal liver and bone activity and does not raise immediate concern for cholestatic injury."
        )
    else:
        explanations.append(
            f"Alkaline Phosphatase is {alk_phos} U/L, elevated above the healthy range. Elevated ALP can be indicative of cholestasis, bile duct blockage, or even bone-related pathology, and should be correlated with other findings."
        )

    # --- ALT (Healthy: 55–250 U/L) ---
    if alt < 55:
        explanations.append(
            f"ALT is {alt} U/L, which is lower than the healthy range of 55–250 U/L. A low ALT is less common and usually does not indicate liver injury, although it might reflect individual variations or reduced muscle mass."
        )
    elif alt <= 250:
        explanations.append(
            f"ALT is {alt} U/L, fitting well within the healthy range. This suggests that liver cell integrity is maintained and there is no overt hepatocellular injury."
        )
    else:
        explanations.append(
            f"ALT is {alt} U/L, elevated above the healthy range. Elevated ALT is a key indicator of liver cell injury and may point to conditions such as hepatitis or non-alcoholic fatty liver disease."
        )

    # --- AST (Healthy: 48–250 U/L) ---
    if ast < 48:
        explanations.append(
            f"AST is {ast} U/L, which is below the healthy range of 48–250 U/L. While low levels are generally not worrisome, they are uncommon and can sometimes reflect nutritional or metabolic factors."
        )
    elif ast <= 250:
        explanations.append(
            f"AST is {ast} U/L, which is within the healthy range. This suggests normal enzymatic activity in the liver, although AST levels can also be influenced by muscle metabolism."
        )
    else:
        explanations.append(
            f"AST is {ast} U/L, elevated above the healthy range. This elevation may indicate liver inflammation or damage, and when viewed together with ALT levels, it can help in assessing the nature of the liver injury."
        )

    # --- AST/ALT Ratio ---
    explanations.append(
        f"AST/ALT Ratio is {ast_alt_ratio}. Typically, a ratio above 2 raises concern for alcoholic liver injury or advanced fibrosis, whereas lower ratios are seen in acute liver inflammation or in healthy individuals."
    )

    # --- Albumin (Healthy: 3.5–5.0 g/dL) ---
    if albumin < 3.5:
        explanations.append(
            f"Albumin is {albumin} g/dL, which is below the healthy range. Low albumin levels are concerning as they may indicate chronic liver dysfunction, malnutrition, or prolonged inflammation, often associated with more advanced liver disease."
        )
    elif albumin <= 5.0:
        explanations.append(
            f"Albumin is {albumin} g/dL, which is within the healthy range. Normal albumin levels typically reflect good liver synthetic function and nutritional status, though monitoring is recommended if other markers are abnormal."
        )
    else:
        explanations.append(
            f"Albumin is {albumin} g/dL, slightly above the healthy range. This finding is less common and may be related to dehydration rather than an intrinsic liver problem."
        )

    # --- Total Proteins (Healthy: 6.3–7.9 g/dL) ---
    if proteins < 6.3:
        explanations.append(
            f"Total Proteins are {proteins} g/dL, which is below the healthy range. A low total protein level can be indicative of impaired liver synthetic function or malnutrition, often observed in chronic liver disease."
        )
    elif proteins <= 7.9:
        explanations.append(
            f"Total Proteins are {proteins} g/dL, well within the healthy range. This suggests that overall protein synthesis and balance are maintained."
        )
    else:
        explanations.append(
            f"Total Proteins are {proteins} g/dL, above the healthy range. Elevated protein levels may occur in chronic inflammation or certain hematologic conditions and should be interpreted in the context of other clinical findings."
        )

    # --- Prothrombin Time (Healthy: 12.5–16.0 s) ---
    if prothrombin < 12.5:
        explanations.append(
            f"Prothrombin Time is {prothrombin} s, which is shorter than the healthy range. While a reduced PT is less common, it may reflect a hypercoagulable state or lab variability."
        )
    elif prothrombin <= 16.0:
        explanations.append(
            f"Prothrombin Time is {prothrombin} s, falling within the healthy range. This is indicative of adequate liver production of clotting factors."
        )
    else:
        explanations.append(
            f"Prothrombin Time is {prothrombin} s, prolonged beyond the healthy range. A prolonged PT is a significant marker that can point to decreased liver synthetic function or vitamin K deficiency, frequently seen in advanced liver disease."
        )

    # --- Platelets (Healthy: 120–450 ×10³/µL) ---
    if platelets < 120:
        explanations.append(
            f"Platelet count is {platelets} ×10³/µL, which is below the healthy range. Low platelets (thrombocytopenia) are commonly associated with portal hypertension and splenic sequestration in advanced liver disease such as cirrhosis."
        )
    elif platelets <= 450:
        explanations.append(
            f"Platelet count is {platelets} ×10³/µL, within the healthy range. Normal platelet levels suggest that there is no significant splenic sequestration or bone marrow suppression."
        )
    else:
        explanations.append(
            f"Platelet count is {platelets} ×10³/µL, which is above the healthy range. Elevated platelets (thrombocytosis) are less frequently linked to liver disease and might indicate a reactive process, such as inflammation or iron deficiency."
        )

    # --- Albumin/Globulin Ratio ---
    explanations.append(
        f"The Albumin/Globulin Ratio is {alb_glob_ratio}. A lower ratio (often below 1.0) can indicate chronic inflammation or liver scarring, while a higher ratio is usually consistent with normal liver function."
    )

    # --- Ascites ---
    if ascites == "Present":
        explanations.append(
            "Ascites is reported as Present. The presence of ascites is a clinical sign often seen in advanced liver disease, particularly cirrhosis, due to increased portal pressure and fluid retention."
        )
    else:
        explanations.append(
            "Ascites is reported as Absent. This is a favorable finding, suggesting that there is no significant fluid accumulation in the abdomen typically associated with advanced liver dysfunction."
        )

    # --- Liver Firmness ---
    if liver_firmness == "Present":
        explanations.append(
            "Liver Firmness is reported as Present. A firm liver on examination or imaging may indicate the presence of fibrosis or cirrhosis, suggesting that the liver has undergone structural changes due to chronic damage."
        )
    else:
        explanations.append(
            "Liver Firmness is reported as Absent. This finding is consistent with a liver that does not show overt signs of advanced scarring, though early fibrosis cannot be completely ruled out without further imaging."
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
            0: "Overall values suggest normal liver function. Minor variations can occur due to transient conditions.",
            1: "The pattern of liver enzymes and synthetic function indicates inflammation. Further tests are recommended to pinpoint the cause.",
            2: "Signs of scarring and altered liver function are evident. This suggests a chronic process where fibrosis has developed.",
            3: "There is evidence of severe liver damage with advanced scarring. Immediate medical intervention is strongly recommended."
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
